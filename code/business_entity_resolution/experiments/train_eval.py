"""Step 7: train the matcher on pairs_train, measure the real score on pairs_val.

The score is the competition metric: F0.5 per S1 record, averaged over EVERY
validation S1 record -- singletons included, and true matches that blocking
never proposed counted as misses (n_true comes from anchors_val.parquet).

The threshold sweep here is a first look; choosing the decision rule
properly is Step 8.

Run from repo root:
    code/business_entity_resolution/experiments/run_capped.sh \\
        code/business_entity_resolution/experiments/train_eval.py
"""
import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.features import FEATURE_NAMES  # noqa: E402
from business_er.metrics import entity_f05_counts, macro_f05_at_threshold  # noqa: E402
from business_er.train import Matcher, train_matcher  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="v1")
args = ap.parse_args()

OUT = ROOT / "artifacts"
(OUT / "models").mkdir(exist_ok=True)
t0 = time.time()


def mark(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:6.1f}s {rss:4.1f}GB] {m}", flush=True)


def load(name):
    df = pd.read_parquet(OUT / f"pairs_{name}.parquet")
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    return X, df["label"].to_numpy(np.int8), df["anchor"].to_numpy(np.int64)


# ---- train -----------------------------------------------------------------
X, y, anchor = load("train")
mark(f"train pairs {X.shape}, positives {y.mean():.3%}")
m = train_matcher(X, y, anchor)
model_path = OUT / "models" / f"matcher_{args.tag}.lgb"
m.save(model_path)
mark(f"trained: best iteration {m.meta['best_iteration']}, dev logloss "
     f"{m.meta['dev_logloss']:.4f} -> {model_path.name}")
del X, y, anchor

print("\n  top features by gain:")
for f, g in list(m.importance().items())[:15]:
    print(f"    {f:<18}{g:6.1%}")

# ---- validate --------------------------------------------------------------
X, y, anchor = load("val")
p = m.predict(X)
del X
n_true = pd.read_parquet(OUT / "anchors_val.parquet")["n_true"].to_numpy()
n = len(n_true)
mark(f"scored {len(p):,} val pairs")

found = np.bincount(anchor, weights=y, minlength=n)
oracle = entity_f05_counts(found, np.zeros(n), n_true).mean()
print(f"\n  validation: {n:,} S1 records, singleton rate {(n_true == 0).mean():.3f}")
print(f"  all-empty baseline  {(n_true == 0).mean():.4f}")
print(f"  blocking ceiling    {oracle:.4f}   (perfect matcher on these candidates)")

print(f"\n  {'threshold':>9}{'macro F0.5':>12}{'non-single':>12}{'singleton':>11}"
      f"{'pair prec':>11}{'pair rec':>10}{'pred/S1':>9}")
best = (0, None)
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    keep = p >= t
    tp = np.bincount(anchor[keep], weights=y[keep], minlength=n)
    fp = np.bincount(anchor[keep], weights=1 - y[keep], minlength=n)
    s = entity_f05_counts(tp, fp, n_true)
    ns, sg = s[n_true > 0].mean(), s[n_true == 0].mean()
    prec = tp.sum() / max(tp.sum() + fp.sum(), 1)
    rec = tp.sum() / n_true.sum()
    print(f"  {t:>9.2f}{s.mean():>12.4f}{ns:>12.4f}{sg:>11.4f}{prec:>11.4f}{rec:>10.4f}"
          f"{keep.sum() / n:>9.2f}")
    if s.mean() > best[0]:
        best = (s.mean(), t)
assert abs(macro_f05_at_threshold(anchor, y, p, n_true, best[1]) - best[0]) < 1e-12
print(f"\n  best so far: macro F0.5 {best[0]:.4f} at threshold {best[1]}"
      f"  (ceiling {oracle:.4f})")
np.save(OUT / f"val_scores_{args.tag}.npy", p.astype(np.float32))
mark("done")
