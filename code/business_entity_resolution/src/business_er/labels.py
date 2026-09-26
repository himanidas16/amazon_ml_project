"""Label retrieved pairs using consistent local anchor coordinates."""
import numpy as np
from .retrieve import target_code
PAIR_BASE = 10_000_000_000

def label_candidates(truth, truth_rows, anchor, code):
    """truth_rows[local anchor] selects its global ground-truth row.

    Handles reordered/non-contiguous country shards without adding truth edges.
    """
    rows = np.asarray(truth_rows, dtype=np.int64)
    anchor = np.asarray(anchor, dtype=np.int64)
    code = np.asarray(code, dtype=np.int64)
    if anchor.shape != code.shape or anchor.ndim != 1:
        raise ValueError("anchor and code must be aligned one-dimensional arrays")
    if len(anchor) and (anchor.min() < 0 or anchor.max() >= len(rows)):
        raise ValueError("candidate anchor outside local truth rows")
    keys = []
    for local, row in enumerate(rows):
        values, sources = truth.targets_of(int(row))
        keys.extend((local * PAIR_BASE + target_code(sources, values)).tolist())
    return np.isin(anchor * PAIR_BASE + code, np.asarray(keys, np.int64)).astype(np.int8)
