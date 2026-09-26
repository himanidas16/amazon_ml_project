"""Round 2: build a MUCH larger training table that still fits in memory.

Differences from build_pairs.py (v1, 100k training anchors, every pair kept):

* Many more training anchors (default 500k from folds 1-4).
* Easy negatives are SUBSAMPLED after their features are computed:
    kept always   all positives, and "hard" negatives -- pairs that look like
                  a match (top-5 among rivals, or high name/address similarity,
                  or blocking rank < 3)
    kept at rate r  every other negative, with weight 1/r so the model's
                  probabilities keep their natural scale (see
                  test_weights_keep_probabilities_calibrated)
  Context features are computed on ALL candidates first, so subsampling
  never changes what a kept pair's features say about its rivals.
* One country at a time, features in chunks of anchors, each chunk written
  to its own parquet file -- the same memory pattern as predict.py.
* Validation: the same 20k fold-0 anchors as v1, all pairs kept, searched
  against the full pool, rebuilt with the CURRENT code so v1 and v2 models
  can be compared on identical features.

Leak guard as in v1: the training index excludes every fold-0 record.

Outputs in artifacts/pairs_v2/:
    train_<country>_<chunk>.parquet   anchor, s1_value, target_code, label, weight, features
    val_<country>_<chunk>.parquet     same columns, weight 1
    anchors_train.parquet, anchors_val.parquet   (anchor, s1_value, country, n_true)

Run from repo root (close Firefox; ~40-60 min):
    code/business_entity_resolution/experiments/run_capped.sh \\
        code/business_entity_resolution/experiments/build_pairs_v2.py --train-anchors 500000
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

from business_er.features import F_INDEX, FEATURE_NAMES, compute_features, prepare  # noqa: E402
from business_er.io import iter_source  # noqa: E402
from business_er.predict import _prepare_targets  # noqa: E402
from business_er.retrieve import (  # noqa: E402
    SOURCE_BASE, TokenFreq, build_index, compute_keys, generate_candidates, target_code,
)
from business_er.splits import load_split, load_truth_csr  # noqa: E402
from business_er.labels import label_candidates  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train-anchors", type=int, default=500_000)
ap.add_argument("--val-anchors", type=int, default=20_000)
ap.add_argument("--easy-rate", type=float, default=0.1, help="share of easy negatives kept")
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--k", type=int, default=25)
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--anchors-per-chunk", type=int, default=20_000)
ap.add_argument("--out", default="artifacts/pairs_v2")
args = ap.parse_args()
if not 0 < args.easy_rate <= 1:
    ap.error("--easy-rate must be in (0, 1]")

D = ROOT / "amazon_ml_dataset/student_resource/dataset/train"
ART = ROOT / "artifacts"
OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)
if list(OUT.glob("*.parquet")):
    raise RuntimeError(f"{OUT} already contains pair data; use a new --out to avoid mixing runs")
SRC = {2: D / "train_source2.tsv", 3: D / "train_source3.tsv"}
t0 = time.time()


def mark(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:7.1f}s {rss:4.1f}GB] {m}", flush=True)


sp = load_split(ART / "split_5fold.npz")
truth = load_truth_csr(D / "train_ground_truth.tsv")
freq = TokenFreq.load(ART / "token_freq_excl_fold0.npz")

# ---- anchors (val sample identical to v1 / blocking experiments) --------------
random.seed(11)
val_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold == 0).tolist(), args.val_anchors)))
random.seed(12)
train_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold != 0).tolist(), args.train_anchors)))
want = {"train": train_idx, "val": val_idx}

# S1 text for exactly these anchors, streamed (never the whole file at once)
want_vals = {n: np.sort(truth.s1_values[i].astype(np.int64)) for n, i in want.items()}
parts = {n: [] for n in want}
for df in iter_source(D / "train_source1.tsv"):
    v = df["entity_id"].str.slice(3).astype(np.int64).to_numpy()
    for n in want:
        m = np.isin(v, want_vals[n])
        if m.any():
            sub = df.loc[m, ["business_name", "business_address", "country"]].copy()
            sub["s1_value"] = v[m]
            parts[n].append(sub)
anchors = {}
for n, idx in want.items():
    a = pd.concat(parts[n], ignore_index=True)
    # truth row of each anchor, via its id
    order = np.argsort(truth.s1_values)
    a["truth_row"] = order[np.searchsorted(truth.s1_values[order], a["s1_value"].to_numpy())]
    a["n_true"] = np.diff(truth.indptr)[a["truth_row"].to_numpy()]
    anchors[n] = a.reset_index(drop=True)
    anchors[n][["s1_value", "country", "n_true"]].rename_axis("anchor").reset_index() \
        .to_parquet(OUT / f"anchors_{n}.parquet", index=False)
del parts
mark(f"anchors: {len(anchors['train']):,} train, {len(anchors['val']):,} val")

fold0 = {s: np.sort(sp.tgt_values[s][sp.tgt_fold[s] == 0].astype(np.int64)) for s in (2, 3)}
not_fold0 = lambda s, v: ~np.isin(v, fold0[s])


countries = sorted(set(anchors["train"]["country"]) | set(anchors["val"]["country"]))
rng = np.random.default_rng(20260926)
stats = {"train_pairs_all": 0, "train_pairs_kept": 0, "train_pos": 0, "val_pairs": 0}

for ctry in countries:
    cands = {}
    for n in ("train", "val"):
        a = anchors[n]
        sel = np.flatnonzero((a["country"] == ctry).to_numpy())
        if not len(sel):
            continue
        index = build_index(SRC, keep=not_fold0 if n == "train" else None, country=ctry,
                            freq=freq, workers=args.workers)
        sub = a.iloc[sel]
        ak = compute_keys(sub["business_name"].tolist(), sub["business_address"].tolist(),
                          sub["country"].tolist(), workers=args.workers, freq=freq)
        c = generate_candidates(index, ak, len(sel), cap=args.cap, k=args.k,
                                batch=args.batch, workers=args.workers)
        cands[n] = dict(sel=sel, anchor=c.anchor.astype(np.int64), code=index.codes[c.target],
                        score=c.score, rank=c.rank, channels=c.channels)
        mark(f"[{ctry}/{n}] {len(sel):,} anchors -> {len(c):,} pairs")
        del index, ak, c

    # candidate text once per country, for train and val together
    need = np.unique(np.concatenate([c["code"] for c in cands.values()]))
    tcodes, T = _prepare_targets(SRC, ctry, need, args.workers)
    t_order = np.argsort(tcodes)
    mark(f"[{ctry}] prepared text of {len(need):,} candidates")

    for n, c in cands.items():
        a = anchors[n]
        sel = c["sel"]
        sub = a.iloc[sel]
        A = prepare(sub["business_name"].tolist(), sub["business_address"].tolist(),
                    workers=args.workers, countries=sub["country"].tolist())
        anchor, code = c["anchor"], c["code"]
        t_pos = t_order[np.searchsorted(tcodes[t_order], code)]
        label = label_candidates(truth, a["truth_row"].to_numpy()[sel], anchor, code)
        bounds = np.searchsorted(anchor, np.arange(0, len(sel) + args.anchors_per_chunk,
                                                   args.anchors_per_chunk))
        for ci, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            if lo == hi:
                continue
            sl = slice(lo, hi)
            meta = {"anchor": anchor[sl], "score": c["score"][sl], "rank": c["rank"][sl],
                    "channels": c["channels"][sl], "source": (code[sl] // SOURCE_BASE).astype(np.int8)}
            X = compute_features(A, T, anchor[sl], t_pos[sl], [ctry] * (hi - lo), meta,
                                 freq=freq, workers=args.workers)
            y = label[sl]
            w = np.ones(hi - lo, np.float32)
            keep = np.ones(hi - lo, bool)
            if n == "train":
                hard = ((X[:, F_INDEX["combo_rank"]] < 5) | (X[:, F_INDEX["ret_rank"]] < 3)
                        | (np.nan_to_num(X[:, F_INDEX["name_tset"]]) >= 0.8)
                        | (np.nan_to_num(X[:, F_INDEX["addr_tset"]]) >= 0.8))
                easy_neg = (y == 0) & ~hard
                keep = ~easy_neg | (rng.random(hi - lo) < args.easy_rate)
                w[easy_neg] = 1.0 / args.easy_rate
                stats["train_pairs_all"] += hi - lo
                stats["train_pairs_kept"] += int(keep.sum())
                stats["train_pos"] += int(y.sum())
            else:
                stats["val_pairs"] += hi - lo
            df = pd.DataFrame(X[keep], columns=list(FEATURE_NAMES))
            df.insert(0, "weight", w[keep])
            df.insert(0, "label", y[keep])
            df.insert(0, "target_code", code[sl][keep])
            df.insert(0, "s1_value", a["s1_value"].to_numpy()[sel][anchor[sl][keep]])
            # anchor = row in anchors_<n>.parquet (global within the set)
            df.insert(0, "anchor", sel[anchor[sl][keep]].astype(np.int64))
            df.to_parquet(OUT / f"{n}_{ctry}_{ci:03d}.parquet", index=False)
            del X, df
        mark(f"[{ctry}/{n}] features written")
        del A
    del T, tcodes, cands

s = stats
print(f"\n  train pairs: {s['train_pairs_all']:,} -> kept {s['train_pairs_kept']:,} "
      f"({s['train_pairs_kept'] / max(s['train_pairs_all'], 1):.1%}); positives {s['train_pos']:,}")
print(f"  val pairs: {s['val_pairs']:,} (all kept)")
mark("done")
