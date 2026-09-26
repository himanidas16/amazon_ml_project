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

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .evaluate import one_owner, threshold
from .features import F_INDEX, FEATURES_VERSION, FEATURE_NAMES, compute_features, prepare
from .select import SELECT_VERSION, prepare_pool, select_candidates
from .stage2 import KEEP, P_MIN, competition_features, stage2_matrix
from .normalize import NORMALIZE_VERSION
from .io import iter_source, read_source, write_id_lists_by_group
from .retrieve import (RETRIEVE_VERSION, SOURCE_BASE, TokenFreq, build_index, compute_keys,
                       generate_candidates, pair_counts, target_code)
from .train import Matcher

PREDICT_VERSION = "1.1.0"   # 1.1: work_dir manifest guards against stale stages


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
                                 df["business_address"][m].tolist(), workers=workers,
                                 countries=[country] * int(m.sum())))
            del df
    codes = np.concatenate(codes) if codes else np.empty(0, np.int64)
    if len(codes) != len(need):
        raise RuntimeError(f"text found for {len(codes):,} of {len(need):,} candidates")
    return codes, pd.concat(parts, ignore_index=True)


COUNT_KEYS = ("t_name_cnt", "t_addr_cnt", "a_name_cnt", "a_addr_cnt")


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_manifest(work_dir: Path, manifest: dict) -> None:
    """Refuse to mix saved stages computed under different settings.

    Saved blocking/scores are only valid for the exact model, token table,
    code versions and blocking parameters that produced them.  Reusing them
    after any of those change would silently mix old and new results.
    """
    path = work_dir / "manifest.json"
    stages = list(work_dir.glob("*_cands.npz")) + list(work_dir.glob("*_scores.npy"))
    if path.exists():
        old = json.loads(path.read_text())
        # a work dir that also stored stage-2 inputs holds everything a
        # stage-1-only run needs (same blocking, same stage-1 scores)
        if old.get("stores_stage2_inputs") and not manifest.get("stores_stage2_inputs"):
            manifest = {**manifest, "stores_stage2_inputs": True}
        if old != manifest:
            diff = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
            raise RuntimeError(f"{work_dir} holds stages built with different settings "
                               f"({', '.join(diff)}); use a new --work-dir or delete this one")
    elif stages:
        raise RuntimeError(f"{work_dir} holds saved stages but no manifest, so their settings "
                           f"are unknown; use a new --work-dir or delete this one")
    else:
        path.write_text(json.dumps(manifest, indent=2))


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
    stage2_path: str | Path | None = None, t2: float | None = None,
    k_wide: int = 0, k_formula: int = 0, extra_k: int = 0,
) -> Dict[str, dict]:
    """stage2_path: optional stage-2 LightGBM model (see stage2.py).  When
    given, the stage-1 features stage 2 needs are saved per pair, and the
    final decision is  stage-2 probability >= t2, then one owner."""
    d = Path(data_dir) / split
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    src = {2: d / f"{split}_source2.tsv", 3: d / f"{split}_source3.tsv"}
    matcher = Matcher.load(model_path)
    freq = TokenFreq.load(freq_path)
    stage2 = None
    if stage2_path is not None:
        import lightgbm as lgb
        stage2 = lgb.Booster(model_file=str(stage2_path))
        if t2 is None:
            raise ValueError("a stage-2 model needs its threshold t2")
    check_manifest(work_dir, {
        "split": split, "cap": cap, "k": k,
        "model_sha256": _sha256(model_path), "freq_sha256": _sha256(freq_path),
        "normalize_version": NORMALIZE_VERSION, "retrieve_version": RETRIEVE_VERSION,
        "features_version": FEATURES_VERSION, "predict_version": PREDICT_VERSION,
        "stores_stage2_inputs": stage2 is not None,
        "k_wide": k_wide, "k_formula": k_formula, "select_version": SELECT_VERSION,
        "extra_k": extra_k,
    })
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
            score = rank = chans = counts = None   # loaded only if scoring is needed
            say(f"[{ctry}] loaded blocking from {cpath.name}")
        else:
            index = build_index(src, country=ctry, freq=freq, workers=workers,
                                keep_text=bool(k_wide > k and k_formula > 0), extra=extra_k > 0)
            say(f"[{ctry}] index built: {index.n_targets:,} targets")
            ak = compute_keys(a["business_name"].tolist(), a["business_address"].tolist(),
                              a["country"].tolist(), workers=workers, freq=freq, extra=extra_k > 0)
            if k_wide > k and k_formula > 0:
                # wide retrieval + fixed-formula re-selection (select.py)
                sa = prepare(a["business_name"].tolist(), a["business_address"].tolist(),
                             workers=workers, countries=a["country"].tolist())
                pool = prepare_pool(src, index, country=ctry, workers=workers)
                c, counts = select_candidates(index, ak, len(sel), sa["name"], sa["addr"],
                                              pool["name"], pool["addr"], cap=cap, k_keep=k,
                                              k_wide=k_wide, k_formula=k_formula, batch=batch,
                                              workers=workers, threads=predict_threads,
                                              extra_k=extra_k, log=say)
                del sa, pool
            else:
                c = generate_candidates(index, ak, len(sel), cap=cap, k=k, batch=batch, workers=workers)
                counts = pair_counts(index, ak, c.anchor, c.target, len(sel))
            anchor, code = c.anchor.astype(np.int32), index.codes[c.target]
            score, rank, chans = c.score, c.rank, c.channels
            n_targets = index.n_targets
            del index, ak, c
            np.savez(cpath, anchor=anchor, code=code, score=score, rank=rank, channels=chans, **counts)
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
                counts = {k: z[k] for k in COUNT_KEYS if k in z.files}
            need = np.unique(code)
            tcodes, T = _prepare_targets(src, ctry, need, workers)
            t_order = np.argsort(tcodes)
            t_pos = t_order[np.searchsorted(tcodes[t_order], code)].astype(np.int32)
            del t_order, tcodes
            A = prepare(a["business_name"].tolist(), a["business_address"].tolist(), workers=workers,
                        countries=a["country"].tolist())
            say(f"[{ctry}] prepared text: {len(A):,} S1, {len(T):,} candidates")
            # Scores go to an on-disk array, with a progress marker saved after
            # every chunk: a stop mid-country (the India run was OOM-killed
            # after 6M of 40M pairs) resumes at the last finished chunk.
            ppath = work_dir / f"{tag}_scores.partial.npy"
            done_path = work_dir / f"{tag}_scores.partial.done"
            if ppath.exists() and done_path.exists():
                p = np.load(ppath, mmap_mode="r+")
                done = int(done_path.read_text())
                say(f"[{ctry}] resuming scoring at pair {done:,}")
            else:
                p = np.lib.format.open_memmap(ppath, mode="w+", dtype=np.float32, shape=(len(anchor),))
                done = 0
            kpath = work_dir / f"{tag}_keep.partial.npy"
            kcols = None
            if stage2 is not None:
                kcols = (np.load(kpath, mmap_mode="r+") if done else
                         np.lib.format.open_memmap(kpath, mode="w+", dtype=np.float32,
                                                   shape=(len(anchor), len(KEEP))))
            bounds = np.searchsorted(anchor, np.arange(0, len(sel) + anchors_per_chunk, anchors_per_chunk))
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                if lo == hi or hi <= done:
                    continue
                sl = slice(lo, hi)
                meta = {"anchor": anchor[sl], "score": score[sl], "rank": rank[sl],
                        "channels": chans[sl], "source": (code[sl] // SOURCE_BASE).astype(np.int8),
                        **{k: v[sl] for k, v in counts.items()}}
                X = compute_features(A, T, anchor[sl], t_pos[sl], [ctry] * (hi - lo), meta,
                                     freq=freq, workers=workers)
                p[sl] = matcher.predict(X, FEATURE_NAMES, num_threads=predict_threads)
                if kcols is not None:
                    kcols[sl] = X[:, [F_INDEX[f] for f in KEEP]]
                    kcols.flush()
                del X
                p.flush()
                done_path.write_text(str(hi))
                say(f"[{ctry}] scored {hi:,}/{len(anchor):,} pairs")
            p = np.asarray(p, dtype=np.float32).copy()
            if kcols is not None:
                np.save(work_dir / f"{tag}_keep.npy", np.asarray(kcols))
                del kcols
                kpath.unlink()
            np.save(spath, p)
            ppath.unlink(); done_path.unlink()
            del A, T, t_pos

        # ---- 3. decision ---------------------------------------------------
        if stage2 is not None:
            # competition features over EVERY pair of the country; stage 2
            # re-judges only rows with p >= P_MIN, read from a memory map
            idx = np.flatnonzero(p >= P_MIN)
            comp = competition_features(anchor, code, p, selected=idx)
            kc = np.load(work_dir / f"{tag}_keep.npy", mmap_mode="r")
            p2 = p.astype(np.float32).copy()
            for s0 in range(0, len(idx), 2_000_000):
                ii = idx[s0:s0 + 2_000_000]
                X2 = stage2_matrix(p[ii], {f: kc[ii, i] for i, f in enumerate(KEEP)},
                                   {k: v[s0:s0 + len(ii)] for k, v in comp.items()})
                p2[ii] = stage2.predict(X2, num_threads=predict_threads)
            del kc, comp
            p = p2
            keep = threshold(p, t2)
        else:
            keep = threshold(p, t)
        if use_one_owner:
            keep = one_owner(keep, p, code, anchor.astype(np.int64))
        groups.append((sel, anchor, code, keep))
        del p, score, rank, chans, counts
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
