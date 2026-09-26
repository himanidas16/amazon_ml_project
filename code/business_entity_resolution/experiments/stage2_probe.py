"""Stage-2 probe: do features that compare COMPETING claims help?

Every S2/S3 record belongs to at most one S1 business (all 7.6M training
pairs).  Stage 1 judges each pair alone.  Stage 2 adds, per pair:

  record side    how many businesses list this record as a candidate, the best
                 OTHER business's stage-1 probability for it, this pair's rank
                 among the record's claims, how many other claims are > 0.5
  business side  this business's best probability, gap to it, rank within the
                 business, how many of its candidates are > 0.5 / > 0.9, sum

These only make sense when all competing businesses are present, so this runs
on a COMPLETE world (build-pairs --val-world: every fold-0 business vs every
fold-0 record).  Stage 1 never trained on fold 0, so its probabilities there
are honest.  Stage 2 is trained on one half of the businesses and scored on
the other half, both ways round; thresholds are chosen on the training half.

Run from repo root:
    code/business_entity_resolution/experiments/run_capped.sh \\
        code/business_entity_resolution/experiments/stage2_probe.py \\
        --world artifacts/world_v12 --model artifacts/models/matcher_v12_300k.lgb
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.evaluate import one_owner, score_keep, threshold  # noqa: E402
from business_er.features import FEATURE_NAMES  # noqa: E402
from business_er.metrics import entity_f05_counts  # noqa: E402
from business_er.train import Matcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--world", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--min-leaf", type=int, default=200)
args = ap.parse_args()
W = ROOT / args.world if not Path(args.world).is_absolute() else Path(args.world)
t0 = time.time()


def mark(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:7.1f}s {rss:4.1f}GB] {m}", flush=True)


from business_er.stage2 import AGG, KEEP, STAGE2_NAMES, competition_features  # noqa: E402

m1 = Matcher.load(ROOT / args.model if not Path(args.model).is_absolute() else args.model)

# ---- stage-1 probabilities, part by part (never all features in memory) ------
cols = {k: [] for k in ["anchor", "target_code", "label", "p", *KEEP]}
for f in sorted(W.glob("val_*.parquet")):
    pf = pq.ParquetFile(f)
    for batch in pf.iter_batches(batch_size=500_000):
        df = batch.to_pandas()
        p = m1.predict(df[list(FEATURE_NAMES)].to_numpy(np.float32), FEATURE_NAMES, num_threads=12)
        cols["p"].append(p.astype(np.float32))
        for k in ["anchor", "target_code", "label", *KEEP]:
            cols[k].append(df[k].to_numpy())
d = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
del cols
d = d.iloc[np.argsort(d["anchor"].to_numpy(), kind="stable")].reset_index(drop=True)
anchors = pd.read_parquet(W / "anchors_val.parquet")
n_true = anchors["n_true"].to_numpy()
n = len(n_true)
mark(f"stage-1 scored {len(d):,} pairs of {n:,} businesses")

# ---- cross-business features ---------------------------------------------------
p = d["p"].to_numpy(np.float64)
comp = competition_features(d["anchor"].to_numpy(), d["target_code"].to_numpy(), p)
for k, v in comp.items():
    d[k] = v
S2 = list(STAGE2_NAMES)
mark("cross-business features done")

y = d["label"].to_numpy(np.int8)
anchor = d["anchor"].to_numpy(np.int64)
target = d["target_code"].to_numpy(np.int64)
half = np.random.default_rng(9).random(n) < 0.5
in_a = half[anchor]
ts = np.round(np.arange(0.40, 0.96, 0.02), 2)


def best_t(score, sel_pairs, sel_anchor):
    return max(ts, key=lambda t: score_keep(one_owner(threshold(score, t) & sel_pairs, score, target, anchor),
                                            y, anchor, n_true, sel_anchor)["macro_f05"])


params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": args.min_leaf,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "num_threads": 12,
          "seed": 11, "deterministic": True, "force_col_wise": True, "verbosity": -1}
X = d[S2].to_numpy(np.float32)
res = {"stage1": [], "stage2": []}
p2 = np.zeros(len(d))
for train_half in (True, False):
    tr = in_a == train_half
    te = ~tr
    bst = lgb.train(params, lgb.Dataset(X[tr], label=y[tr], feature_name=S2), num_boost_round=600)
    p2[te] = bst.predict(X[te], num_threads=12)
    p2_tr = bst.predict(X[tr], num_threads=12)   # only to choose ITS threshold on its own half
    sel_tr, sel_te = half == train_half, half != train_half
    t1 = best_t(p, tr, sel_tr)
    s1 = score_keep(one_owner(threshold(p, t1) & te, p, target, anchor), y, anchor, n_true, sel_te)
    tmp = np.zeros(len(d)); tmp[tr] = p2_tr
    t2 = best_t(tmp, tr, sel_tr)
    s2 = score_keep(one_owner(threshold(p2, t2) & te, p2, target, anchor), y, anchor, n_true, sel_te)
    res["stage1"].append((float(t1), s1)); res["stage2"].append((float(t2), s2))
    mark(f"half {'A' if train_half else 'B'} as train: stage1 {s1['macro_f05']:.4f} (t={t1}) "
         f"-> stage2 {s2['macro_f05']:.4f} (t={t2})")
imp = bst.feature_importance("gain"); imp = imp / imp.sum()

found = np.bincount(anchor, weights=y, minlength=n)
summary = {
    "world_businesses": n, "pairs": int(len(d)),
    "ceiling": float(entity_f05_counts(found, np.zeros(n), n_true).mean()),
    "stage1_heldout": float(np.mean([r[1]["macro_f05"] for r in res["stage1"]])),
    "stage2_heldout": float(np.mean([r[1]["macro_f05"] for r in res["stage2"]])),
    "stage2_top_features": {S2[i]: round(float(imp[i]), 4) for i in np.argsort(-imp)[:12]},
}
print(json.dumps(summary, indent=2))
(W / "stage2_report.json").write_text(json.dumps({**summary, "detail": res}, indent=2, default=str))
mark("done")
