"""Step 9: complete inference -- data in, both submission files out.

    blocking -> pair features -> matcher probabilities -> decision rule
             -> output/matching_results.tsv + output/candidate_pairs.tsv

Everything runs ONE COUNTRY AT A TIME.  Blocking keys are country-scoped, so
this is exact, and it keeps peak memory near the largest country's share.
Countries are whatever labels appear in source 1 -- France (absent from
training) goes through the same code as US and India.

Each stage saves its result in work_dir, keyed by split and country:
    <split>_<country>_cands.npz    blocking output (the scored-pair universe)
    <split>_<country>_scores.npy   matcher probability per pair
A rerun skips finished stages, so a stopped run resumes instead of restarting.
Delete work_dir to recompute from scratch.

candidate_pairs.tsv is written from exactly the pairs the matcher scored, and
matching_results.tsv from the subset the decision rule kept, so matches are a
subset of candidates by construction.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .evaluate import one_owner, threshold
from .features import FEATURE_NAMES, compute_features, prepare
from .io import iter_source, read_source, write_id_lists_by_group
from .retrieve import SOURCE_BASE, TokenFreq, build_index, compute_keys, generate_candidates, target_code
from .train import Matcher

PREDICT_VERSION = "1.0.0"


def _safe(label: str) -> str:
    """Country label -> file-name-safe token ("" becomes "_blank_")."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", label) or "_blank_"


def _prepare_targets(paths: Dict[int, Path], country: str, need: np.ndarray, workers: int):
    """Prepared text of the needed target codes, built chunk by chunk.

    Each ~400k-row chunk of a source file is filtered and prepared at once,
    so raw strings for millions of candidates never sit in memory together
    (holding them all is what stalled the first India run).
    Returns (codes, prepared DataFrame) aligned row by row.
    """
    codes, parts = [], []
    for s, path in paths.items():
        for df in iter_source(path):
            df = df[df["country"] == country]
            cd = target_code(s, df["entity_id"].str.slice(3).astype(np.int64).to_numpy())
            m = np.isin(cd, need)
            if not m.any():
                continue
            codes.append(cd[m])
            parts.append(prepare(df["business_name"][m].tolist(),
                                 df["business_address"][m].tolist(), workers=workers))
            del df
    codes = np.concatenate(codes) if codes else np.empty(0, np.int64)
    if len(codes) != len(need):
        raise RuntimeError(f"text found for {len(codes):,} of {len(need):,} candidates")
    return codes, pd.concat(parts, ignore_index=True)


def _anchor_text(path: Path, country: str) -> pd.DataFrame:
    """Name/address of one country's S1 records, in file order."""
    parts = [df.loc[df["country"] == country, ["business_name", "business_address", "country"]]
             for df in iter_source(path)]
    return pd.concat(parts, ignore_index=True)


def run(
    data_dir: str | Path, split: str, model_path: str | Path, freq_path: str | Path,
    out_dir: str | Path, work_dir: str | Path,
    cap: int = 3000, k: int = 25, t: float = 0.80, use_one_owner: bool = True,
    workers: int = 6, batch: int = 100, anchors_per_chunk: int = 20_000,
    predict_threads: int = 12, log: Callable[[str], None] = print,
) -> Dict[str, dict]:
    d = Path(data_dir) / split
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    src = {2: d / f"{split}_source2.tsv", 3: d / f"{split}_source3.tsv"}
    matcher = Matcher.load(model_path)
    freq = TokenFreq.load(freq_path)
    t0 = time.time()
    say = lambda m: log(f"[{time.time() - t0:7.1f}s] {m}")

    # ids + country only; each country's text is loaded when its turn comes
    s1 = read_source(d / f"{split}_source1.tsv", "S1-", usecols=["entity_id", "country"])
    s1_ids: List[str] = s1["entity_id"].tolist()
    s1_country = s1["country"].to_numpy()
    del s1
    countries = sorted(set(s1_country.tolist()))
    say(f"{split}: {len(s1_ids):,} S1 records; countries {countries}")

    groups = []   # per country: (S1 rows, anchor, code, keep) -- never concatenated
    summary: Dict[str, dict] = {}
    for ctry in countries:
        tag = f"{split}_{_safe(ctry)}"
        sel = np.flatnonzero(s1_country == ctry)
        a = _anchor_text(d / f"{split}_source1.tsv", ctry)
        if len(a) != len(sel):
            raise RuntimeError(f"{ctry}: {len(a)} text rows for {len(sel)} S1 ids")

        # ---- 1. blocking ---------------------------------------------------
        cpath = work_dir / f"{tag}_cands.npz"
        if cpath.exists():
            z = np.load(cpath)
            anchor, code = z["anchor"], z["code"]
            score = rank = chans = None          # loaded only if scoring is needed
            say(f"[{ctry}] loaded blocking from {cpath.name}")
        else:
            index = build_index(src, country=ctry, freq=freq, workers=workers)
            ak = compute_keys(a["business_name"].tolist(), a["business_address"].tolist(),
                              a["country"].tolist(), workers=workers, freq=freq)
            c = generate_candidates(index, ak, len(sel), cap=cap, k=k, batch=batch, workers=workers)
            anchor, code = c.anchor.astype(np.int32), index.codes[c.target]
            score, rank, chans = c.score, c.rank, c.channels
            n_targets = index.n_targets
            del index, ak, c
            np.savez(cpath, anchor=anchor, code=code, score=score, rank=rank, channels=chans)
            say(f"[{ctry}] blocking: {len(sel):,} S1 vs {n_targets:,} targets -> {len(anchor):,} pairs")
        if len(anchor) and np.any(anchor[1:] < anchor[:-1]):
            raise RuntimeError("candidates are not grouped by anchor")

        # ---- 2. features + probabilities, chunk by chunk --------------------
        spath = work_dir / f"{tag}_scores.npy"
        if spath.exists():
            p = np.load(spath)
            say(f"[{ctry}] loaded scores from {spath.name}")
        else:
            if score is None:
                score, rank, chans = z["score"], z["rank"], z["channels"]
            need = np.unique(code)
            tcodes, T = _prepare_targets(src, ctry, need, workers)
            t_order = np.argsort(tcodes)
            t_pos = t_order[np.searchsorted(tcodes[t_order], code)]
            A = prepare(a["business_name"].tolist(), a["business_address"].tolist(), workers=workers)
            say(f"[{ctry}] prepared text: {len(A):,} S1, {len(T):,} candidates")
            p = np.empty(len(anchor), np.float32)
            bounds = np.searchsorted(anchor, np.arange(0, len(sel) + anchors_per_chunk, anchors_per_chunk))
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                if lo == hi:
                    continue
                sl = slice(lo, hi)
                meta = {"anchor": anchor[sl], "score": score[sl], "rank": rank[sl],
                        "channels": chans[sl], "source": (code[sl] // SOURCE_BASE).astype(np.int8)}
                X = compute_features(A, T, anchor[sl], t_pos[sl], [ctry] * (hi - lo), meta,
                                     freq=freq, workers=workers)
                p[sl] = matcher.predict(X, FEATURE_NAMES, num_threads=predict_threads)
                del X
                say(f"[{ctry}] scored {hi:,}/{len(anchor):,} pairs")
            np.save(spath, p)
            del A, T

        # ---- 3. decision ---------------------------------------------------
        keep = threshold(p, t)
        if use_one_owner:
            keep = one_owner(keep, p, code, anchor.astype(np.int64))
        groups.append((sel, anchor, code, keep))
        del p, score, rank, chans
        n_matched = len(np.unique(anchor[keep]))
        summary[ctry] = {
            "s1": int(len(sel)), "pairs": int(len(anchor)), "kept": int(keep.sum()),
            "s1_with_match": n_matched, "share_with_match": n_matched / max(len(sel), 1),
            "kept_per_s1": float(keep.sum() / max(len(sel), 1)),
        }
        say(f"[{ctry}] kept {keep.sum():,} matches; {n_matched / max(len(sel), 1):.1%} of S1 "
            f"get >=1 match; {keep.sum() / max(len(sel), 1):.2f} per S1")

    write_id_lists_by_group(out_dir / "candidate_pairs.tsv", s1_ids,
                            [(sel, anc, cd) for sel, anc, cd, _ in groups], "candidate_entity_ids")
    write_id_lists_by_group(out_dir / "matching_results.tsv", s1_ids,
                            [(sel, anc[kp], cd[kp]) for sel, anc, cd, kp in groups], "matched_entity_ids")
    say(f"wrote {out_dir / 'matching_results.tsv'} and {out_dir / 'candidate_pairs.tsv'}")
    return summary
