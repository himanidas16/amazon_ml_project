"""Stage 2: re-judge each pair knowing the COMPETING claims.

Every S2/S3 record belongs to at most one S1 business (verified on all 7.6M
training pairs).  Stage 1 scores each (business, record) pair on its own.
Stage 2 adds what the other pairs say:

  record side    how many businesses list the record, the best OTHER claim's
                 stage-1 probability, this claim's margin over it and its rank,
                 how many other claims exceed 0.5
  business side  the business's best probability, the gap to it, this pair's
                 rank, how many candidates exceed 0.5 / 0.9, their sum

These are only meaningful when every competing business is present, which is
true for the test set and for a complete validation world
(build-pairs --val-world), and NOT for a sample of businesses.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

STAGE2_VERSION = "1.1.0"

# stage-1 features carried into stage 2 (cheap to store per test pair)
KEEP: Tuple[str, ...] = (
    "ret_score", "ret_rank", "avail_mean", "avail_rank", "name_tset", "addr_tset", "combo_rank",
    "n_strong_rivals", "t_addr_cnt", "a_addr_cnt", "t_name_cnt", "a_name_cnt",
    "numj_first_equal", "num_first_equal", "addr_empty_t", "n_channels", "is_s3",
)
AGG: Tuple[str, ...] = (
    "a_pmax", "a_gap", "a_rank", "a_n50", "a_n90", "a_psum",
    "t_nclaims", "t_rank", "t_best_other", "t_margin", "t_n50_other",
)
STAGE2_NAMES: Tuple[str, ...] = ("p", *KEEP, *AGG)


def _segments(key: np.ndarray, p: np.ndarray):
    """Sort by (key, p descending); return order, group starts and sizes."""
    o = np.lexsort((-p, key))
    ks = key[o]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]]) if len(o) else np.empty(0, np.int64)
    sizes = np.diff(np.r_[starts, len(o)])
    return o, starts, sizes


def competition_features(anchor: np.ndarray, target: np.ndarray, p: np.ndarray,
                         selected=None) -> Dict[str, np.ndarray]:
    """Global competition features, optionally only for selected output rows.

    All candidates remain in the competing population. Selection avoids
    materializing eleven full-pool columns when only ~8% are re-judged.
    """
    from .competition import selected_competition_features
    return selected_competition_features(anchor, target, p, selected)


# Stage 2 re-judges only pairs stage 1 does not already rule out; the rest keep
# their stage-1 probability.  Their probabilities still enter the competition
# features above (which always use every pair).
P_MIN = 0.005


def stage2_matrix(p: np.ndarray, keep_cols: Dict[str, np.ndarray],
                  comp: Dict[str, np.ndarray]) -> np.ndarray:
    """Columns in STAGE2_NAMES order."""
    cols = [np.asarray(p, np.float32)] + [np.asarray(keep_cols[k], np.float32) for k in KEEP] \
        + [comp[k] for k in AGG]
    return np.column_stack(cols).astype(np.float32)


# --------------------------------------------------------------------------
# training on a complete world
# --------------------------------------------------------------------------

STAGE2_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63,
                 "min_data_in_leaf": 200, "feature_fraction": 0.9, "bagging_fraction": 0.8,
                 "bagging_freq": 1, "num_threads": 12, "seed": 11, "deterministic": True,
                 "force_col_wise": True, "verbosity": -1}


def load_world(world_dir, stage1_model, log=print):
    """Stage-1 probabilities (every pair) and the KEEP features, read part by
    part into preallocated arrays.  Returns a dict of arrays + n_true."""
    from pathlib import Path

    import pyarrow.parquet as pq

    from .features import FEATURE_NAMES
    from .train import Matcher

    W = Path(world_dir)
    m1 = Matcher.load(stage1_model)
    files = [pq.ParquetFile(f) for f in sorted(W.glob("val_*.parquet"))]
    n = sum(f.metadata.num_rows for f in files)
    # KEEP features are stored only for rows stage 2 will re-judge (p >= P_MIN,
    # ~8% of pairs): storing them for every pair is what exhausted memory
    w = {"anchor": np.empty(n, np.int64), "target": np.empty(n, np.int64),
         "label": np.empty(n, np.int8), "p": np.empty(n, np.float32)}
    keep_rows, keep_parts = [], []
    pos = 0
    for f in files:
        for batch in f.iter_batches(batch_size=250_000):
            df = batch.to_pandas()
            e = pos + len(df)
            w["p"][pos:e] = m1.predict(df[list(FEATURE_NAMES)].to_numpy(np.float32),
                                       FEATURE_NAMES, num_threads=12)
            w["anchor"][pos:e] = df["anchor"].to_numpy()
            w["target"][pos:e] = df["target_code"].to_numpy()
            w["label"][pos:e] = df["label"].to_numpy()
            m = w["p"][pos:e] >= P_MIN
            keep_rows.append(np.flatnonzero(m) + pos)
            keep_parts.append(df.loc[m, list(KEEP)].to_numpy(np.float32))
            pos = e
            del df
        log(f"stage 1 scored {pos:,}/{n:,} pairs")
    w["keep_idx"] = np.concatenate(keep_rows)
    w["keep"] = np.concatenate(keep_parts)
    n_true = pd.read_parquet(W / "anchors_val.parquet")["n_true"].to_numpy()
    return w, n_true


def train_stage2(world_dir, stage1_model, out_path, rounds: int = 600, log=print) -> Dict:
    """Fit stage 2 on a complete world and measure it honestly.

    Businesses are split in two halves.  Out-of-fold stage-2 probabilities
    come from a model trained on the other half.  For the REPORTED score,
    each half's threshold is chosen on the other half; the one-owner rule
    always sees every claim (both halves).  Stage 1 is measured the same way,
    so the comparison is like for like.  The saved model is refit on all
    businesses; its threshold is the best on all out-of-fold predictions."""
    import json
    from pathlib import Path

    import lightgbm as lgb

    from .evaluate import one_owner, score_keep, threshold

    w, n_true = load_world(world_dir, stage1_model, log)
    p1 = w["p"].astype(np.float64)
    anchor, target, y = w["anchor"], w["target"], w["label"]
    idx = w["keep_idx"]                              # rows stage 2 re-judges (p >= P_MIN)
    comp = competition_features(anchor, target, p1, selected=idx)
    X = stage2_matrix(p1[idx], {k: w["keep"][:, i] for i, k in enumerate(KEEP)}, comp)
    del comp, w["keep"]
    log(f"stage 2 rows: {len(idx):,} of {len(p1):,} pairs (p >= {P_MIN})")

    n = len(n_true)
    half = np.random.default_rng(9).random(n) < 0.5
    oof = p1.copy()
    for h in (True, False):
        tr = half[anchor[idx]] == h
        b = lgb.train(STAGE2_PARAMS, lgb.Dataset(X[tr], label=y[idx][tr]), num_boost_round=rounds)
        oof[idx[~tr]] = b.predict(X[~tr], num_threads=12)
    ts = np.round(np.arange(0.30, 0.96, 0.02), 2)

    def score_on(prob, t, subset):
        return score_keep(one_owner(threshold(prob, t), prob, target, anchor),
                          y, anchor, n_true, subset)["macro_f05"]

    def crossed(prob):
        """each half scored with the threshold chosen on the other half"""
        res = []
        for h in (True, False):
            t = max(ts, key=lambda t: score_on(prob, t, half == h))
            res.append(score_on(prob, t, half != h))
        return float(np.mean(res))

    s1_held, s2_held = crossed(p1), crossed(oof)
    t_best = float(max(ts, key=lambda t: score_on(oof, t, None)))
    t1_best = float(max(ts, key=lambda t: score_on(p1, t, None)))
    final = lgb.train(STAGE2_PARAMS, lgb.Dataset(X, label=y[idx], feature_name=list(STAGE2_NAMES)),
                      num_boost_round=rounds)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out.with_suffix(".lgb")))
    report = {"stage2_version": STAGE2_VERSION, "p_min": P_MIN, "threshold": t_best,
              "stage1_threshold": t1_best, "stage1_heldout": s1_held, "stage2_heldout": s2_held,
              "gain": s2_held - s1_held, "rows_rejudged": int(len(idx)), "pairs": int(len(p1))}
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    log(f"held-out macro F0.5: stage 1 {s1_held:.4f} -> stage 2 {s2_held:.4f} "
        f"(gain {s2_held - s1_held:+.4f}); thresholds t1={t1_best} t2={t_best}")
    return report
