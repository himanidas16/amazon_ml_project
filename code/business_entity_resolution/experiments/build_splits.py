"""Build and verify the train/validation splits, and save them to artifacts/.

Run from the repository root:
    python3 code/business_entity_resolution/experiments/build_splits.py

Produces (all gitignored, all reproducible from the fixed seed):
    artifacts/split_5fold.npz    -- the real validation split (5 folds)
    artifacts/split_25fold.npz   -- a finer split; one fold (~88k S1) is a cheap
                                    dev slice for fast blocking experiments
    artifacts/countries.npz      -- country per record, so experiments need not
                                    reload the 500 MB source files just for that
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.ids import IdIndex, encode  # noqa: E402
from business_er.io import read_source  # noqa: E402
from business_er.splits import check_split, load_truth_csr, make_split, save_split  # noqa: E402

DATA = ROOT / "amazon_ml_dataset/student_resource/dataset/train"
OUT = ROOT / "artifacts"
SEED = 20260925

t0 = time.time()


def mark(msg):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:6.1f}s  peak {rss:5.2f} GB]  {msg}", flush=True)


OUT.mkdir(exist_ok=True)
truth = load_truth_csr(DATA / "train_ground_truth.tsv")
mark(f"labels: {len(truth):,} S1, {len(truth.tgt_values):,} true matches (exclusivity OK)")

s1 = read_source(DATA / "train_source1.tsv", "S1-", usecols=["entity_id", "country"])
pos = IdIndex(s1["entity_id"].tolist(), 1).positions(truth.s1_values)
if (pos < 0).any():
    raise SystemExit(f"{int((pos < 0).sum())} ground-truth S1 ids missing from source1")
s1_country = s1["country"].to_numpy()[pos]
del s1

tv, tc = {}, {}
for src in (2, 3):
    df = read_source(DATA / f"train_source{src}.tsv", f"S{src}-", usecols=["entity_id", "country"])
    tv[src] = encode(df["entity_id"].tolist(), src)
    tc[src] = df["country"].to_numpy()
    del df
mark("sources loaded")

counts = truth.match_counts
for n_folds in (5, 25):
    split = make_split(truth, s1_country, tv, tc, n_folds=n_folds, seed=SEED)
    problems = check_split(split, truth)
    if problems:
        raise SystemExit(f"{n_folds}-fold split leaks: {problems}")
    save_split(OUT / f"split_{n_folds}fold.npz", split)
    mark(f"{n_folds}-fold split built, verified leak-free, saved")
    if n_folds == 5:
        print(split.summary())
        print(f"  {'fold':<6}{'S1':>10}{'singleton%':>12}{'mean matches':>14}{'US%':>8}")
        for f in range(n_folds):
            m = split.s1_mask(f)
            c, ct = counts[m], s1_country[m]
            print(f"  {f:<6}{m.sum():>10,}{100 * (c == 0).mean():>11.2f}%"
                  f"{c[c > 0].mean():>14.3f}{100 * (ct == 'US').mean():>7.1f}%")
    else:
        n1 = int(split.s1_mask(0).sum())
        n2, n3 = (int(split.target_mask(s, 0).sum()) for s in (2, 3))
        print(f"  dev slice (fold 0 of 25): S1={n1:,} S2={n2:,} S3={n3:,} "
              f"targets/S1={(n2 + n3) / n1:.2f}")

np.savez_compressed(
    OUT / "countries.npz",
    s1_values=truth.s1_values, s1_country=s1_country.astype("U8"),
    t2_values=tv[2], t2_country=tc[2].astype("U8"),
    t3_values=tv[3], t3_country=tc[3].astype("U8"),
)
mark("countries saved -- done")
