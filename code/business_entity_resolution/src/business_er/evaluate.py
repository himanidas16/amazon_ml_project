"""Step 8: turn match probabilities into the final lists of ids.

The matcher scores each (S1, candidate) pair independently.  The competition
scores whole LISTS per S1 record, so the decision rule matters:

  threshold      keep every candidate with p >= t
  one_owner      a target record never belongs to two S1 records (verified
                 on all 7.6M training pairs), so when several S1 records keep
                 the same target, only the highest-probability claim survives
  rescue         an S1 record with nothing kept, whose best candidate has
                 p >= t2 (< t), keeps that one candidate: an empty answer for
                 a record that does have matches scores 0
  expected_f05   per S1 record, output the top-k candidates for the k that
                 maximises the EXPECTED F0.5 under the model's probabilities

Every function takes flat per-pair arrays and returns a boolean keep mask,
so validation (Step 8) and test inference (Step 9) share one code path.
Pairs must be grouped by anchor (as build_pairs / predict produce them).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .metrics import entity_f05_counts


def _groups(anchor: np.ndarray):
    """Start index and size of each run of equal anchors (pairs are grouped)."""
    if len(anchor) and np.any(anchor[1:] < anchor[:-1]):
        raise ValueError("pairs must be grouped (sorted) by anchor")
    starts = np.flatnonzero(np.r_[True, anchor[1:] != anchor[:-1]]) if len(anchor) else np.empty(0, int)
    sizes = np.diff(np.r_[starts, len(anchor)])
    return starts, sizes


def threshold(p: np.ndarray, t: float) -> np.ndarray:
    return np.asarray(p) >= t


def one_owner(keep: np.ndarray, p: np.ndarray, target: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Among KEPT pairs, each target stays only with its highest-p anchor.
    Ties go to the lower anchor index so the result is deterministic."""
    keep = keep.copy()
    idx = np.flatnonzero(keep)
    if not len(idx):
        return keep
    order = np.lexsort((anchor[idx], -p[idx], target[idx]))   # by target, best p first
    tg = target[idx][order]
    first = np.r_[True, tg[1:] != tg[:-1]]
    keep[idx[order[~first]]] = False
    return keep


def rescue(keep: np.ndarray, p: np.ndarray, anchor: np.ndarray, t2: float) -> np.ndarray:
    """For anchors with nothing kept, keep their best candidate if p >= t2."""
    keep = keep.copy()
    starts, sizes = _groups(anchor)
    if not len(starts):
        return keep
    any_kept = np.add.reduceat(keep.astype(np.int64), starts) > 0
    # position of the best candidate in each group
    order = np.lexsort((-p, anchor))
    best = order[starts]
    ok = (~any_kept) & (p[best] >= t2)
    keep[best[ok]] = True
    return keep


def expected_f05(p: np.ndarray, anchor: np.ndarray, miss_factor: float = 1.0,
                 max_k: int = 20) -> np.ndarray:
    """Choose, per anchor, how many top candidates to output.

    With probabilities p_i (sorted descending) and S_k = p_1 + ... + p_k:
      expected true matches  N  = miss_factor * sum(p)   (>1 allows for matches
                                   blocking never proposed)
      k >= 1:  E[F] ~ 1.25 S_k / (1.25 S_k + (k - S_k) + 0.25 (N - S_k))
      k == 0:  E[F] = P(no match at all) ~ prod(1 - p_i)  (singleton rule)
    This is a ratio-of-expectations approximation -- good when p is roughly
    calibrated, which binary log-loss on the natural class mix encourages.
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    n = len(p)
    keep = np.zeros(n, bool)
    if n == 0:
        return keep
    starts, sizes = _groups(anchor)
    order = np.lexsort((-p, anchor))                 # within anchor, best first
    ps = p[order]
    grp = np.repeat(np.arange(len(starts)), sizes)
    rank = np.arange(n) - np.repeat(starts, sizes)   # 0-based
    csum = np.cumsum(ps)
    base = np.repeat(csum[starts] - ps[starts], sizes)
    S = csum - base                                  # S_k for k = rank+1
    N = miss_factor * np.repeat(np.add.reduceat(ps, starts), sizes)
    k = rank + 1
    with np.errstate(invalid="ignore", divide="ignore"):
        ef = 1.25 * S / (1.25 * S + (k - S) + 0.25 * np.maximum(N - S, 0.0))
    ef = np.nan_to_num(ef, nan=0.0)
    ef[rank >= max_k] = -1.0
    # k = 0: probability that nothing matches
    log1m = np.log(np.maximum(1.0 - ps, 1e-12))
    p_none = np.exp(np.add.reduceat(log1m, starts))
    # best k per group: the smallest k reaching the group's highest E[F]
    best_ef = np.maximum.reduceat(ef, starts)
    rows = np.flatnonzero(ef == np.repeat(best_ef, sizes))   # every group has one
    first_best = np.full(len(starts), n, dtype=np.int64)
    np.minimum.at(first_best, grp[rows], rows)
    # output nothing unless some k beats "no match at all" (ties -> empty)
    choose_k = np.where(best_ef > p_none, rank[first_best] + 1, 0)
    keep_sorted = rank < np.repeat(choose_k, sizes)
    keep[order] = keep_sorted
    return keep


def score_keep(keep: np.ndarray, label: np.ndarray, anchor: np.ndarray,
               n_true: np.ndarray, subset: np.ndarray | None = None) -> Dict[str, float]:
    """Competition score of a keep mask (plus its singleton / non-singleton
    halves).  n_true covers every S1 row, so blocking misses count.
    subset: optional boolean mask over S1 rows to score only those."""
    n = len(n_true)
    tp = np.bincount(anchor[keep], weights=label[keep], minlength=n)
    fp = np.bincount(anchor[keep], weights=1 - label[keep], minlength=n)
    s = entity_f05_counts(tp, fp, n_true)
    nt = n_true
    if subset is not None:
        s, nt = s[subset], n_true[subset]
    return {
        "macro_f05": float(s.mean()),
        "non_singleton": float(s[nt > 0].mean()) if (nt > 0).any() else float("nan"),
        "singleton": float(s[nt == 0].mean()) if (nt == 0).any() else float("nan"),
        "pred_per_s1": float(keep.sum() / n),
    }
