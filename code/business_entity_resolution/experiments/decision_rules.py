"""Step 8: compare decision rules on the saved validation scores.

Overfitting guard: the 20k validation S1 records are split in two halves by
record.  Each rule's settings are tuned on one half and scored on the other,
both ways round; the reported number is the average of the two HELD-OUT
scores.  A rule that only wins on the half it was tuned on is noise.

Run from repo root (small: ~0.5 GB):
    code/business_entity_resolution/experiments/run_capped.sh \\
        code/business_entity_resolution/experiments/decision_rules.py --tag v1
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.evaluate import expected_f05, one_owner, rescue, score_keep, threshold  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v1")
args = ap.parse_args()
OUT = ROOT / "artifacts"

v = pd.read_parquet(OUT / "pairs_val.parquet", columns=["anchor", "target_code", "label"])
anchor = v["anchor"].to_numpy(np.int64)
target = v["target_code"].to_numpy(np.int64)
label = v["label"].to_numpy(np.int8)
p = np.load(OUT / f"val_scores_{args.tag}.npy").astype(np.float64)
n_true = pd.read_parquet(OUT / "anchors_val.parquet")["n_true"].to_numpy()
n = len(n_true)

rng = np.random.default_rng(5)
half = np.zeros(n, bool); half[rng.choice(n, n // 2, replace=False)] = True
halves = {"A": half, "B": ~half}

# ---- calibration: are the probabilities honest? ---------------------------
print("  calibration (predicted vs actual match rate):")
for lo, hi in [(0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, .97), (.97, 1.01)]:
    m = (p >= lo) & (p < hi)
    if m.any():
        print(f"    p in [{lo:.2f},{min(hi, 1):.2f}): {m.sum():>8,} pairs  predicted {p[m].mean():.3f}"
              f"  actual {label[m].mean():.3f}")


def cross_fit(name, grid, make_keep):
    """Tune on one half, score on the other, both ways; return held-out mean."""
    held = []
    for tune, test in (("A", "B"), ("B", "A")):
        best = max(grid, key=lambda g: score_keep(make_keep(g), label, anchor, n_true,
                                                  halves[tune])["macro_f05"])
        s = score_keep(make_keep(best), label, anchor, n_true, halves[test])
        held.append((best, s))
    mean = {k: np.mean([h[1][k] for h in held]) for k in held[0][1]}
    print(f"  {name:<34}{mean['macro_f05']:>10.4f}{mean['non_singleton']:>12.4f}"
          f"{mean['singleton']:>11.4f}{mean['pred_per_s1']:>9.2f}   chosen {[h[0] for h in held]}")
    return mean["macro_f05"], held


ts = [round(x, 2) for x in np.arange(0.50, 0.96, 0.01)]
print(f"\n  {'rule (held-out score)':<34}{'macro F0.5':>10}{'non-single':>12}{'singleton':>11}{'pred/S1':>9}")
res = {}
res["A threshold"] = cross_fit("A  global threshold", ts, lambda t: threshold(p, t))
res["B +one_owner"] = cross_fit("B  threshold + one owner", ts,
                                lambda t: one_owner(threshold(p, t), p, target, anchor))
grid_c = [(t, t2) for t in ts[::2] for t2 in (0.2, 0.3, 0.4, 0.5, 0.6) if t2 < t]
res["C +rescue"] = cross_fit("C  threshold + rescue", grid_c,
                             lambda g: rescue(threshold(p, g[0]), p, anchor, g[1]))
mfs = [1.0, 1.02, 1.05, 1.1, 1.2]
res["D expected"] = cross_fit("D  expected F0.5", mfs, lambda mf: expected_f05(p, anchor, mf))
res["D +one_owner"] = cross_fit("D  expected F0.5 + one owner", mfs,
                                lambda mf: one_owner(expected_f05(p, anchor, mf), p, target, anchor))

# how often do val anchors fight over a target at all?
keep_a = threshold(p, 0.75)
tk = target[keep_a]
print(f"\n  targets kept by >1 val anchor at t=0.75: {int((np.unique(tk, return_counts=True)[1] > 1).sum())} "
      f"(only 20k of 441k fold-0 anchors present, so test will have far more)")

# ---- where the remaining points go, for the best rule ----------------------
best_rule = max(res, key=lambda k: res[k][0])
print(f"\n  best rule: {best_rule}")
t_fix = 0.75
keep = threshold(p, t_fix)
tp = np.bincount(anchor[keep], weights=label[keep], minlength=n)
fp = np.bincount(anchor[keep], weights=1 - label[keep], minlength=n)
found = np.bincount(anchor, weights=label, minlength=n)
from business_er.metrics import entity_f05_counts  # noqa: E402
s = entity_f05_counts(tp, fp, n_true)
loss = 1 - s
cats = {
    "singleton, wrongly given a match": (n_true == 0) & (fp > 0),
    "has matches, predicted nothing": (n_true > 0) & (tp + fp == 0),
    "has matches, some wrong ones added": (n_true > 0) & (fp > 0) & (tp + fp > 0),
    "has matches, only missed some": (n_true > 0) & (fp == 0) & (tp > 0) & (tp < n_true),
}
print(f"  where the lost points are (threshold {t_fix}, total loss {loss.mean():.4f}):")
for k, m in cats.items():
    print(f"    {k:<38}{m.sum():>6,} S1   loss {loss[m].sum() / n:.4f}")
bm = (n_true > found)
print(f"    (of all loss, S1 records with a blocking miss: {loss[bm].sum() / n:.4f})")
