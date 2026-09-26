"""Step 6: build the labelled pair tables the model trains and validates on.

    train: a sample of S1 records from folds 1-4, searched against an index
           WITHOUT fold-0 records -- so no validation record ever enters
           training, not even as a negative example.
    val:   the same 20k fold-0 S1 records used in the blocking experiments,
           searched against ALL train targets, exactly as test records will
           search the whole test pool.

Blocking uses the final Step-5 settings (cap 3000, 25 per source, token
rarity table counted outside fold 0).

Outputs (artifacts/, gitignored):
    pairs_{train,val}.parquet    one row per (S1, candidate): ids, label, features
    anchors_{train,val}.parquet  one row per S1: id, country, n_true -- n_true
                                 counts ALL true matches, including ones
                                 blocking missed, so later scores stay honest

Run from repo root:
    code/business_entity_resolution/experiments/run_capped.sh \\
        code/business_entity_resolution/experiments/build_pairs.py
"""
import argparse
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.features import FEATURE_NAMES, compute_features, prepare  # noqa: E402
from business_er.io import iter_source, read_source  # noqa: E402
from business_er.retrieve import (  # noqa: E402
    SOURCE_BASE, TokenFreq, build_index, compute_keys, generate_candidates, target_code,
)
from business_er.splits import load_split, load_truth_csr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train-anchors", type=int, default=100_000)
ap.add_argument("--val-anchors", type=int, default=20_000)
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--k", type=int, default=25)
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()

D = ROOT / "amazon_ml_dataset/student_resource/dataset/train"
OUT = ROOT / "artifacts"
SRC = {2: D / "train_source2.tsv", 3: D / "train_source3.tsv"}
t0 = time.time()


def mark(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:6.1f}s {rss:4.1f}GB] {m}", flush=True)


sp = load_split(OUT / "split_5fold.npz")
truth = load_truth_csr(D / "train_ground_truth.tsv")
freq = TokenFreq.load(OUT / "token_freq_excl_fold0.npz")   # built by blocking_fullscale.py --freq

# ---- anchors ---------------------------------------------------------------
random.seed(11)   # same val sample as blocking_fullscale.py
val_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold == 0).tolist(), args.val_anchors)))
random.seed(12)
train_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold != 0).tolist(), args.train_anchors)))
sets = {"train": train_idx, "val": val_idx}

s1 = read_source(D / "train_source1.tsv", "S1-")
s1_vals = s1["entity_id"].str.slice(3).astype(np.int64).to_numpy()
order = np.argsort(s1_vals)
anchors = {}
for name, idx in sets.items():
    rows = order[np.searchsorted(s1_vals[order], truth.s1_values[idx])]
    a = s1.iloc[rows].reset_index(drop=True)
    a["truth_row"] = idx
    anchors[name] = a
del s1, s1_vals, order
mark(f"anchors: {len(train_idx):,} train, {len(val_idx):,} val")

# fold-0 targets, excluded from the TRAINING index
fold0 = {s: np.sort(sp.tgt_values[s][sp.tgt_fold[s] == 0].astype(np.int64)) for s in (2, 3)}
not_fold0 = lambda s, v: ~np.isin(v, fold0[s])

# ---- blocking (cached: a later failure should not cost another 12 minutes) --
cache = OUT / (f"cands_build_pairs_t{args.train_anchors}_v{args.val_anchors}"
               f"_c{args.cap}_k{args.k}.npz")
if cache.exists():
    z = np.load(cache)
    pairs = {n: {k: z[f"{n}__{k}"] for k in ("anchor", "code", "score", "rank", "channels")}
             for n in sets}
    mark(f"loaded cached candidates from {cache.name}")
else:
    cand = {name: [] for name in sets}   # lists of dicts of arrays, per country
    for name in ("train", "val"):
        a = anchors[name]
        for ctry in sorted(a["country"].unique()):
            sel = np.flatnonzero((a["country"] == ctry).to_numpy())
            index = build_index(SRC, keep=not_fold0 if name == "train" else None, country=ctry,
                                freq=freq, workers=args.workers)
            sub = a.iloc[sel]
            ak = compute_keys(sub["business_name"].tolist(), sub["business_address"].tolist(),
                              sub["country"].tolist(), workers=args.workers, freq=freq)
            c = generate_candidates(index, ak, len(sel), cap=args.cap, k=args.k,
                                    batch=args.batch, workers=args.workers)
            cand[name].append({
                "anchor": sel[c.anchor].astype(np.int32),        # row in anchors[name]
                "code": index.codes[c.target],
                "score": c.score, "rank": c.rank, "channels": c.channels,
            })
            mark(f"[{name}/{ctry}] {len(sel):,} anchors -> {len(c):,} pairs "
                 f"(index {index.n_targets:,} targets)")
            del index, ak, c

    pairs = {}
    for name, parts in cand.items():
        p = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
        o = np.argsort(p["anchor"], kind="stable")                 # group by anchor
        pairs[name] = {k: v[o] for k, v in p.items()}
    np.savez(cache, **{f"{n}__{k}": v for n, p in pairs.items() for k, v in p.items()})
    mark(f"saved candidates to {cache.name}")

# ---- labels ------------------------------------------------------------------
for name, p in pairs.items():
    a = anchors[name]
    tr = a["truth_row"].to_numpy()
    true_keys = []
    n_true = np.zeros(len(a), np.int64)
    for i, r in enumerate(tr):
        v, s = truth.targets_of(int(r))
        n_true[i] = len(v)
        true_keys.extend((i * 10**10 + target_code(s.astype(np.int64), v.astype(np.int64))).tolist())
    p["label"] = np.isin(p["anchor"].astype(np.int64) * 10**10 + p["code"],
                         np.asarray(true_keys, np.int64)).astype(np.int8)
    a["n_true"] = n_true
    found = np.bincount(p["anchor"], weights=p["label"], minlength=len(a))
    print(f"  {name}: {len(p['label']):,} pairs, {int(p['label'].sum()):,} positives "
          f"({p['label'].mean():.3%}), blocking recall {found.sum() / n_true.sum():.4f}")

# leak check: no fold-0 target in any training pair
tc = pairs["train"]["code"]
for s in (2, 3):
    m = tc // SOURCE_BASE == s
    assert not np.isin(tc[m] % SOURCE_BASE, fold0[s]).any(), "fold-0 record leaked into training"
print("  leak check passed: no fold-0 record in training pairs")

# ---- candidate text ----------------------------------------------------------
need = np.unique(np.concatenate([p["code"] for p in pairs.values()]))
names_, addrs_, ctry_, codes_ = [], [], [], []
for s, path in SRC.items():
    for df in iter_source(path):
        cd = target_code(s, df["entity_id"].str.slice(3).astype(np.int64).to_numpy())
        m = np.isin(cd, need)
        codes_.append(cd[m])
        names_ += df["business_name"][m].tolist()
        addrs_ += df["business_address"][m].tolist()
        ctry_ += df["country"][m].tolist()
codes_ = np.concatenate(codes_)
assert len(codes_) == len(need), "some candidate text not found"
mark(f"fetched text of {len(need):,} candidate records")
T = prepare(names_, addrs_, workers=args.workers, countries=ctry_)
del names_, addrs_, ctry_
t_order = np.argsort(codes_)
mark("prepared candidate text")

# ---- features ------------------------------------------------------------------
for name, p in pairs.items():
    a = anchors[name]
    A = prepare(a["business_name"].tolist(), a["business_address"].tolist(), workers=args.workers,
                countries=a["country"].tolist())
    t_pos = t_order[np.searchsorted(codes_[t_order], p["code"])]
    meta = {"anchor": p["anchor"], "score": p["score"], "rank": p["rank"],
            "channels": p["channels"], "source": (p["code"] // SOURCE_BASE).astype(np.int8)}
    X = compute_features(A, T, p["anchor"], t_pos, a["country"].to_numpy()[p["anchor"]],
                         meta, freq=freq, workers=args.workers)
    mark(f"[{name}] features {X.shape}")

    out = pd.DataFrame(X, columns=list(FEATURE_NAMES))
    out.insert(0, "label", p["label"])
    out.insert(0, "target_code", p["code"])
    out.insert(0, "s1_value", truth.s1_values[a["truth_row"].to_numpy()[p["anchor"]]].astype(np.int64))
    out.insert(0, "anchor", p["anchor"])
    out.to_parquet(OUT / f"pairs_{name}.parquet", index=False)
    pd.DataFrame({
        "anchor": np.arange(len(a)), "s1_value": truth.s1_values[a["truth_row"].to_numpy()],
        "country": a["country"].to_numpy(), "n_true": a["n_true"].to_numpy(),
    }).to_parquet(OUT / f"anchors_{name}.parquet", index=False)
    mark(f"[{name}] saved pairs_{name}.parquet, anchors_{name}.parquet")
    del X, out, A

mark("done")
