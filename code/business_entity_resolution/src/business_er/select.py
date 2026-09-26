"""Wide retrieval + a FIXED-formula re-selection (part of blocking).

Blocking ranks candidates by key rarity and keeps the top 25 per source; 4.4%
of true matches are found but ranked lower and lost.  Measured on 20k
validation businesses against the full 10.3M-record pool
(experiments/selection_probe.py):

    candidate set                                   recall   ceiling
    blocking top-25/source (before)                 0.943    0.9791
    blocking top-25  UNION  formula top-10/source    0.965    0.9880   <- used
    learned ranker top-25/source                    0.967    0.9886

The formula is NOT a learned model: the average token-set similarity of the
name and address fields that are present (a missing field is skipped, not
counted as 0), ties broken by the blocking score.  Because nothing here is
learned, this stays part of candidate generation -- candidate_pairs.tsv is
the selected set, which is exactly the input to the (first) scoring model.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .retrieve import (EXTRA_CHANNELS, SOURCE_BASE, Candidates, KeyChunk, KeyIndex,
                       generate_candidates, pair_counts)

SELECT_VERSION = "1.1.0"   # 1.1: optional per-channel quotas for the extra channels


def slice_keys(akeys: KeyChunk, lo: int, hi: int) -> KeyChunk:
    """Anchor keys of anchors lo..hi-1, renumbered from 0 (rows are sorted)."""
    out = {}
    for ch, (h, r) in akeys.items():
        a0, a1 = np.searchsorted(r, [lo, hi])
        out[ch] = (h[a0:a1], r[a0:a1] - np.int32(lo))
    return out


def _tset(a: List[str], t: List[str], threads: int) -> np.ndarray:
    s = process.cpdist(a, t, scorer=fuzz.token_set_ratio, workers=threads).astype(np.float32) / 100
    s[np.fromiter((not x or not y for x, y in zip(a, t)), bool, len(a))] = np.nan
    return s


def formula_score(a_name: pd.Series, a_addr: pd.Series, t_name: pd.Series, t_addr: pd.Series,
                  a_pos: np.ndarray, t_pos: np.ndarray, ret_score: np.ndarray,
                  chunk: int = 500_000, threads: int = 6) -> np.ndarray:
    """Mean of the PRESENT name/address token-set similarities (0 if both
    missing), plus a tiny blocking-score tiebreak."""
    n = len(a_pos)
    out = np.empty(n, np.float64)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ap, tp = a_pos[s:e], t_pos[s:e]
        nm = _tset(a_name.iloc[ap].tolist(), t_name.iloc[tp].tolist(), threads)
        ad = _tset(a_addr.iloc[ap].tolist(), t_addr.iloc[tp].tolist(), threads)
        with np.errstate(invalid="ignore"):
            avail = np.nanmean(np.stack([nm, ad]), axis=0)
        out[s:e] = np.nan_to_num(avail, nan=0.0)
    return out + 1e-6 * np.asarray(ret_score, np.float64)


def select_candidates(
    index: KeyIndex, akeys: KeyChunk, n_anchors: int,
    a_name: pd.Series, a_addr: pd.Series, t_name: pd.Series, t_addr: pd.Series,
    cap: int = 3000, k_keep: int = 25, k_wide: int = 150, k_formula: int = 10,
    batch: int = 100, workers: int = 6, threads: int = 6, anchors_per_range: int = 25_000,
    extra_k: int = 0, log=lambda m: None,
):
    """Selected candidates + their pool counts.

    For each (anchor, source): keep blocking's top k_keep, plus the formula's
    top k_formula among blocking's top k_wide.  t_name/t_addr are aligned
    with index rows; a_name/a_addr with anchor positions.  Work is done in
    ranges of anchors so the wide lists never all sit in memory.

    Returns (Candidates, counts dict), sorted by anchor, source, blocking score.
    """
    src_all = (index.codes // SOURCE_BASE).astype(np.int64)
    parts: List[tuple] = []
    for lo in range(0, n_anchors, anchors_per_range):
        hi = min(lo + anchors_per_range, n_anchors)
        ak = slice_keys(akeys, lo, hi)
        c = generate_candidates(index, ak, hi - lo, cap=cap, k=k_wide, batch=batch, workers=workers)
        if not len(c):
            continue
        a_glob = c.anchor.astype(np.int64) + lo
        fs = formula_score(a_name, a_addr, t_name, t_addr, a_glob, c.target, c.score, threads=threads)
        grp = c.anchor.astype(np.int64) * 8 + src_all[c.target]
        o = np.lexsort((c.target, -fs, grp))
        starts = np.flatnonzero(np.r_[True, grp[o][1:] != grp[o][:-1]])
        frank = np.empty(len(o), np.int64)
        frank[o] = np.arange(len(o)) - np.repeat(starts, np.diff(np.r_[starts, len(o)]))
        keep = (c.rank < k_keep) | (frank < k_formula)
        anc, tgt = c.anchor[keep], c.target[keep]
        sco, chb, rnk = c.score[keep], c.channels[keep], c.rank[keep]
        if extra_k and all(ch in index.keys for ch in EXTRA_CHANNELS):
            # each extra channel adds its own top extra_k per source (as measured)
            n_t = index.n_targets
            code = anc.astype(np.int64) * n_t + tgt
            for ch in EXTRA_CHANNELS:
                e = generate_candidates(index, ak, hi - lo, cap=cap, k=extra_k, batch=batch,
                                        workers=workers, channels=[ch])
                ecode = e.anchor.astype(np.int64) * n_t + e.target
                pos = np.searchsorted(np.sort(code), ecode)
                sc = np.sort(code)
                dup = (pos < len(sc)) & (sc[np.minimum(pos, len(sc) - 1)] == ecode)
                # already selected: just add this channel's flag
                if dup.any():
                    o = np.argsort(code)
                    chb = chb.copy()
                    chb[o[pos[dup]]] |= e.channels[dup]
                new = ~dup
                anc = np.r_[anc, e.anchor[new]]
                tgt = np.r_[tgt, e.target[new]]
                sco = np.r_[sco, e.score[new]]
                chb = np.r_[chb, e.channels[new]]
                rnk = np.r_[rnk, np.full(int(new.sum()), k_wide, rnk.dtype)]   # "beyond the wide list"
                code = anc.astype(np.int64) * n_t + tgt
            src = (index.codes[tgt] // SOURCE_BASE).astype(np.int64)
            o = np.lexsort((tgt, -sco, src, anc))
            anc, tgt, sco, chb, rnk = anc[o], tgt[o], sco[o], chb[o], rnk[o]
        cnt = pair_counts(index, ak, anc, tgt, hi - lo)
        parts.append(((anc.astype(np.int64) + lo).astype(np.int32), tgt.astype(np.int32),
                      sco.astype(np.float32), chb, rnk,
                      {k: v.astype(np.float32) for k, v in cnt.items()}))
        log(f"selected anchors {hi:,}/{n_anchors:,}: {len(c):,} wide -> {len(anc):,} kept")
    if not parts:
        e = np.empty(0, np.int32)
        return (Candidates(e, e, np.empty(0, np.float32), np.empty(0, np.uint16), np.empty(0, np.int16)),
                {k: np.empty(0, np.float32) for k in ("t_name_cnt", "t_addr_cnt", "a_name_cnt", "a_addr_cnt")})
    # Join one column at a time and free each range's piece as soon as it is
    # copied: joining everything at once briefly doubled memory (India, 48M
    # selected pairs, ran out of memory there).
    parts = [list(p[:5]) + [dict(p[5])] for p in parts]

    def take(i, key=None):
        out = np.concatenate([p[i] if key is None else p[i][key] for p in parts])
        for p in parts:
            if key is None:
                p[i] = None
            else:
                del p[i][key]
        return out

    count_keys = list(parts[0][5])
    fields = [take(i) for i in range(5)]
    counts = {k: take(5, k) for k in count_keys}
    cand = Candidates(anchor=fields[0], target=fields[1], score=fields[2], channels=fields[3],
                      rank=fields[4])
    return cand, counts


def prepare_pool(source_paths: Dict[int, str], index: KeyIndex, country=None, keep=None,
                 workers: int = 6, read_chunk: int = 400_000) -> pd.DataFrame:
    """Prepared name/address of EVERY index row, in index row order (the same
    streaming and filters as build_index).  Only the two columns the formula
    needs are kept, to save memory.  Verified against index.codes.

    If the index was built with keep_text=True the text is already there and
    no second pass over the source files is needed."""
    if index.text is not None:
        if len(index.text) != index.n_targets:
            raise RuntimeError("index text is not aligned with the index rows")
        return index.text
    import os

    from .features import prepare_min
    from .io import iter_source
    from .retrieve import target_code

    # text preparation is CPU-bound and light on memory per worker, so it may
    # use more cores than feature computation
    workers = max(workers, min(16, (os.cpu_count() or 4) - 4))

    parts, codes = [], []
    for source in sorted(source_paths):
        for df in iter_source(source_paths[source], chunksize=read_chunk):
            values = df["entity_id"].str.slice(3).astype(np.int64).to_numpy()
            m = np.ones(len(df), dtype=bool)
            if country is not None:
                m &= (df["country"] == country).to_numpy()
            if keep is not None:
                m &= keep(source, values)
            if not m.all():
                df, values = df[m], values[m]
            if not len(df):
                continue
            parts.append(prepare_min(df["business_name"].tolist(), df["business_address"].tolist(),
                                     df["country"].tolist(), workers=workers))
            codes.append(target_code(source, values))
    codes = np.concatenate(codes) if codes else np.empty(0, np.int64)
    if not np.array_equal(codes, index.codes):
        raise RuntimeError("pool text is not aligned with the index rows")
    return pd.concat(parts, ignore_index=True)
