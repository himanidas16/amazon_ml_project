"""Block 2 probe: does a learned selection over WIDE candidate lists beat
blocking's own top-25?

    1. Blocking keeps the top --k-wide per source (instead of 25).
    2. Cheap features (features.cheap_features) for every wide pair.
    3. A small LightGBM ranker, trained on TRAINING businesses (index without
       any fold-0 record), scores the wide pairs.
    4. On the 20k VALIDATION businesses (same sample as every experiment),
       compare the candidate sets:
         baseline  retrieval rank < 25 per source      (what we use today)
         ranker    top-K per source by ranker score
         union     baseline  OR  ranker top-K
       reporting pair recall, candidates per business and the oracle F0.5
       ceiling (score of a perfect matcher on that candidate set).

Ground truth is used only for training the ranker (train businesses) and for
measuring (validation) -- never to build validation candidates.

Run from repo root (after nothing else heavy is running; ~30-40 min):
    code/business_entity_resolution/experiments/run_capped.sh \\
        code/business_entity_resolution/experiments/selection_probe.py
"""
import argparse
import json
import random
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.features import CHEAP_NAMES, cheap_features, prepare  # noqa: E402
from business_er.io import iter_source  # noqa: E402
from business_er.labels import label_candidates  # noqa: E402
from business_er.metrics import entity_f05_counts  # noqa: E402
from business_er.predict import _prepare_targets  # noqa: E402
from business_er.retrieve import (SOURCE_BASE, TokenFreq, build_index, compute_keys,  # noqa: E402
                                  generate_candidates, pair_counts)
from business_er.splits import load_split, load_truth_csr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--train-anchors", type=int, default=30_000)
ap.add_argument("--val-anchors", type=int, default=20_000)
ap.add_argument("--k-wide", type=int, default=150)
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--out", default="artifacts/selection_probe")
args = ap.parse_args()

D = ROOT / "amazon_ml_dataset/student_resource/dataset/train"
ART = ROOT / "artifacts"
OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)
SRC = {2: D / "train_source2.tsv", 3: D / "train_source3.tsv"}
t0 = time.time()


def mark(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:7.1f}s {rss:4.1f}GB] {m}", flush=True)


sp = load_split(ART / "split_5fold.npz")
truth = load_truth_csr(D / "train_ground_truth.tsv")
freq = TokenFreq.load(ART / "token_freq_excl_fold0.npz")

random.seed(11)
val_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold == 0).tolist(), args.val_anchors)))
random.seed(13)   # a different training sample from the matcher's (seed 12)
train_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold != 0).tolist(), args.train_anchors)))
want = {"train": train_idx, "val": val_idx}
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
order = np.argsort(truth.s1_values)
anchors = {}
for n in want:
    a = pd.concat(parts[n], ignore_index=True)
    a["truth_row"] = order[np.searchsorted(truth.s1_values[order], a["s1_value"].to_numpy())]
    a["n_true"] = np.diff(truth.indptr)[a["truth_row"].to_numpy()]
    anchors[n] = a
del parts
mark(f"businesses: {len(anchors['train']):,} train, {len(anchors['val']):,} val")

fold0 = {s: np.sort(sp.tgt_values[s][sp.tgt_fold[s] == 0].astype(np.int64)) for s in (2, 3)}
not_fold0 = lambda s, v: ~np.isin(v, fold0[s])

res = {n: {"X": [], "y": [], "anchor": [], "src": [], "rank": []} for n in want}
for ctry in sorted(set(anchors["val"]["country"])):
    cands = {}
    for n in ("train", "val"):
        a = anchors[n]
        sel = np.flatnonzero((a["country"] == ctry).to_numpy())
        index = build_index(SRC, keep=not_fold0 if n == "train" else None, country=ctry,
                            freq=freq, workers=args.workers)
        sub = a.iloc[sel]
        ak = compute_keys(sub["business_name"].tolist(), sub["business_address"].tolist(),
                          sub["country"].tolist(), workers=args.workers, freq=freq)
        c = generate_candidates(index, ak, len(sel), cap=args.cap, k=args.k_wide,
                                batch=100, workers=args.workers)
        cands[n] = dict(sel=sel, anchor=c.anchor.astype(np.int64), code=index.codes[c.target],
                        score=c.score, rank=c.rank, channels=c.channels,
                        counts=pair_counts(index, ak, c.anchor, c.target, len(sel)))
        mark(f"[{ctry}/{n}] {len(sel):,} businesses -> {len(c):,} wide pairs")
        del index, ak, c
    need = np.unique(np.concatenate([c["code"] for c in cands.values()]))
    tcodes, T = _prepare_targets(SRC, ctry, need, args.workers)
    t_order = np.argsort(tcodes)
    for n, c in cands.items():
        a, sel = anchors[n], c["sel"]
        sub = a.iloc[sel]
        A = prepare(sub["business_name"].tolist(), sub["business_address"].tolist(),
                    workers=args.workers, countries=sub["country"].tolist())
        t_pos = t_order[np.searchsorted(tcodes[t_order], c["code"])]
        src = (c["code"] // SOURCE_BASE).astype(np.int8)
        X = cheap_features(A, T, c["anchor"], t_pos,
                           {"anchor": c["anchor"], "score": c["score"], "rank": c["rank"],
                            "channels": c["channels"], "source": src, **c["counts"]}, threads=6)
        y = label_candidates(truth, a["truth_row"].to_numpy()[sel], c["anchor"], c["code"])
        r = res[n]
        r["X"].append(X); r["y"].append(y); r["anchor"].append(sel[c["anchor"]])
        r["src"].append(src); r["rank"].append(c["rank"])
        mark(f"[{ctry}/{n}] cheap features {X.shape}")
        del A
    del T, tcodes, cands

for n in res:
    res[n] = {k: np.concatenate(v) for k, v in res[n].items()}
    o = np.lexsort((res[n]["src"], res[n]["anchor"]))       # group by anchor, then source
    res[n] = {k: v[o] for k, v in res[n].items()}

# ---- ranker ----------------------------------------------------------------
tr = res["train"]
ranker = lgb.train(
    {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 100,
     "feature_fraction": 0.9, "num_threads": 6, "seed": 7, "deterministic": True,
     "force_col_wise": True, "verbosity": -1},
    lgb.Dataset(tr["X"], label=tr["y"], feature_name=list(CHEAP_NAMES)), num_boost_round=400)
ranker.save_model(str(OUT / "ranker.lgb"))
imp = ranker.feature_importance("gain"); imp = imp / imp.sum()
mark("ranker trained; top gain: " + ", ".join(f"{CHEAP_NAMES[i]} {imp[i]:.0%}"
                                              for i in np.argsort(-imp)[:6]))

# ---- evaluate candidate sets on validation ----------------------------------
va = res["val"]
s = ranker.predict(va["X"], num_threads=6)
n_true = anchors["val"]["n_true"].to_numpy()
n = len(n_true)
country = anchors["val"]["country"].to_numpy()
grp = va["anchor"] * 8 + va["src"]
o = np.lexsort((-s, grp))
rrank = np.empty(len(s), np.int64)
starts = np.flatnonzero(np.r_[True, grp[o][1:] != grp[o][:-1]])
rrank[o] = np.arange(len(s)) - np.repeat(starts, np.diff(np.r_[starts, len(s)]))


def evaluate(mask):
    found = np.bincount(va["anchor"][mask], weights=va["y"][mask], minlength=n)
    f = entity_f05_counts(found, np.zeros(n), n_true)
    out = {"recall": float(found.sum() / n_true.sum()), "cand_per_s1": float(mask.sum() / n),
           "oracle": float(f.mean())}
    for c in sorted(set(country)):
        out[f"oracle_{c}"] = float(f[country == c].mean())
    return out


rows = {"wide (all)": evaluate(np.ones(len(s), bool)),
        "baseline: retrieval rank<25": evaluate(va["rank"] < 25)}
for k in (10, 15, 20, 25, 30, 40, 50):
    rows[f"ranker top-{k}/source"] = evaluate(rrank < k)
for k in (5, 10, 15):
    rows[f"union: baseline + ranker top-{k}"] = evaluate((va["rank"] < 25) | (rrank < k))


def rank_by(score):
    o = np.lexsort((-score, grp))
    r = np.empty(len(score), np.int64)
    r[o] = np.arange(len(score)) - np.repeat(starts, np.diff(np.r_[starts, len(score)]))
    return r


# FIXED formulas (no learning): these would count as part of blocking, so the
# exported candidate file stays at the final selection (organizers: the file
# must hold the input to the FIRST scoring model).
from business_er.features import C_INDEX  # noqa: E402
Xv = va["X"]
nm, ad = Xv[:, C_INDEX["c_name_tset"]], Xv[:, C_INDEX["c_addr_tset"]]
with np.errstate(invalid="ignore"):
    avail = np.nan_to_num(np.nanmean(np.stack([nm, ad]), axis=0), nan=0.0)
rs = Xv[:, C_INDEX["ret_score"]]
rules = {
    "rule A: avail-mean sim": avail + 1e-6 * rs,
    "rule B: avail-mean + 0.01*ret": avail + 0.01 * rs,
    "rule C: max(name,addr) + avail": np.nan_to_num(np.fmax(nm, ad)) + avail + 1e-6 * rs,
}
for name_, sc in rules.items():
    rk = rank_by(sc)
    for k in (25,):
        rows[f"{name_} top-{k}/source"] = evaluate(rk < k)
    rows[f"union: baseline + {name_.split(':')[0]} top-10"] = evaluate((va["rank"] < 25) | (rk < 10))
print(f"\n  {'candidate set':<34}{'recall':>9}{'cand/S1':>9}{'oracle':>9}{'India':>9}{'US':>9}")
for name, r in rows.items():
    print(f"  {name:<34}{r['recall']:>9.4f}{r['cand_per_s1']:>9.1f}{r['oracle']:>9.4f}"
          f"{r.get('oracle_India', float('nan')):>9.4f}{r.get('oracle_US', float('nan')):>9.4f}")
(OUT / "report.json").write_text(json.dumps(rows, indent=2))
np.savez(OUT / "val_wide.npz", anchor=va["anchor"], src=va["src"], rank=va["rank"],
         y=va["y"], ranker_score=s.astype(np.float32))
mark("done")
