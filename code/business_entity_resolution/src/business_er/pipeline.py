"""The training half of the pipeline, as functions behind the CLI.

    split        train/validation folds, grouped by business
    token-freq   word counts (train records outside the validation fold)
    build-pairs  labelled (S1, candidate) pairs with features, train + val
    train        LightGBM matcher, then the competition score on validation

Together with `predict` these regenerate both submission files from the raw
data using only this package.  The logic is the one measured in experiments/
(build_splits.py, build_pairs_v2.py, train_eval.py), moved here so that every
step of the final submission lives under src/.

Everything is deterministic: fixed seeds for the split, the anchor samples,
negative subsampling and LightGBM.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd

from .evaluate import one_owner, score_keep, threshold
from .features import F_INDEX, FEATURE_NAMES, compute_features, prepare
from .ids import IdIndex, encode
from .io import iter_source, read_source
from .labels import label_candidates
from .metrics import entity_f05_counts
from .pair_data import load_pair_parts
from .predict import _prepare_targets
from .retrieve import (
    SOURCE_BASE, TokenFreq, build_index, build_token_freq, compute_keys, generate_candidates,
    pair_counts,
)
from .select import prepare_pool, select_candidates
from .splits import check_split, load_split, load_truth_csr, make_split, save_split
from .train import Matcher, train_matcher

SPLIT_SEED = 20260925
VAL_FOLD = 0
VAL_SAMPLE_SEED = 11      # the 20k validation businesses used in every experiment
TRAIN_SAMPLE_SEED = 12


def _plain(o):
    """numpy scalars -> plain Python numbers for JSON reports."""
    return o.item() if hasattr(o, "item") else str(o)


def _logger(log: Callable[[str], None]):
    t0 = time.time()
    return lambda m: log(f"[{time.time() - t0:7.1f}s] {m}")


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def build_splits(data_dir, art_dir, n_folds: int = 5, log=print) -> Path:
    """Folds grouped by S1 business (each business and all its true matches
    share a fold), stratified by country and match count.  Fold 0 = validation."""
    say = _logger(log)
    d, art = Path(data_dir) / "train", Path(art_dir)
    art.mkdir(parents=True, exist_ok=True)
    truth = load_truth_csr(d / "train_ground_truth.tsv")
    s1 = read_source(d / "train_source1.tsv", "S1-", usecols=["entity_id", "country"])
    pos = IdIndex(s1["entity_id"].tolist(), 1).positions(truth.s1_values)
    if (pos < 0).any():
        raise RuntimeError(f"{int((pos < 0).sum())} ground-truth S1 ids missing from source 1")
    s1_country = s1["country"].to_numpy()[pos]
    tv, tc = {}, {}
    for s in (2, 3):
        df = read_source(d / f"train_source{s}.tsv", f"S{s}-", usecols=["entity_id", "country"])
        tv[s], tc[s] = encode(df["entity_id"].tolist(), s), df["country"].to_numpy()
    split = make_split(truth, s1_country, tv, tc, n_folds=n_folds, seed=SPLIT_SEED)
    problems = check_split(split, truth)
    if problems:
        raise RuntimeError(f"split leaks: {problems}")
    out = art / f"split_{n_folds}fold.npz"
    save_split(out, split)
    say(f"{n_folds}-fold split verified leak-free -> {out}")
    return out


# --------------------------------------------------------------------------
# word counts
# --------------------------------------------------------------------------

def train_token_freq(data_dir, art_dir, workers: int = 6, log=print) -> Path:
    """Word counts over training records OUTSIDE the validation fold, so
    validation records are 'unseen' exactly as test records are."""
    d, art = Path(data_dir) / "train", Path(art_dir)
    sp = load_split(art / "split_5fold.npz")
    s1_out = np.sort(sp.s1_values[sp.s1_fold != VAL_FOLD].astype(np.int64))
    t_out = {s: np.sort(sp.tgt_values[s][sp.tgt_fold[s] != VAL_FOLD].astype(np.int64)) for s in (2, 3)}
    keep = lambda s, v: np.isin(v, s1_out if s == 1 else t_out[s])
    freq = build_token_freq({s: str(d / f"train_source{s}.tsv") for s in (1, 2, 3)},
                            keep=keep, workers=workers, log=log)
    out = art / "token_freq_excl_fold0.npz"
    freq.save(out)
    log(f"saved {len(freq.hashes):,} tokens -> {out}")
    return out


# --------------------------------------------------------------------------
# labelled pairs
# --------------------------------------------------------------------------

def build_pairs(
    data_dir, art_dir, out_name: str = "pairs", train_anchors: int = 100_000,
    val_anchors: int = 20_000, easy_rate: float = 1.0, cap: int = 3000, k: int = 25,
    batch: int = 100, workers: int = 6, anchors_per_chunk: int = 20_000,
    val_world: bool = False, k_wide: int = 0, k_formula: int = 0, log=print,
) -> Path:
    """Blocking + labels + features for a sample of training and validation
    businesses, one country at a time, written as parquet parts.

    Training businesses come from folds 1-4 and search an index WITHOUT any
    fold-0 record (no validation record ever enters training, even as a
    negative).  Validation businesses (fold 0) search the full pool.

    easy_rate < 1 keeps every positive and every hard negative but only that
    share of the easy negatives, with weight 1/easy_rate (probabilities stay
    calibrated).  easy_rate = 1 keeps everything.

    val_world=True builds a COMPLETE validation world instead: every fold-0
    business (val_anchors ignored) searching ONLY fold-0 records.  Every
    record's possible owners are then present, as in the test set -- needed
    for features that compare competing claims on the same record.
    train_anchors=0 skips the training side.
    """
    if not 0 < easy_rate <= 1:
        raise ValueError("easy_rate must be in (0, 1]")
    say = _logger(log)
    d, art = Path(data_dir) / "train", Path(art_dir)
    out = art / out_name
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("*.parquet")):
        raise RuntimeError(f"{out} already holds pair tables; choose another --out-name")
    src = {2: d / "train_source2.tsv", 3: d / "train_source3.tsv"}
    sp = load_split(art / "split_5fold.npz")
    truth = load_truth_csr(d / "train_ground_truth.tsv")
    freq = TokenFreq.load(art / "token_freq_excl_fold0.npz")

    if val_world:
        val_idx = np.flatnonzero(sp.s1_fold == VAL_FOLD)
    else:
        random.seed(VAL_SAMPLE_SEED)
        val_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold == VAL_FOLD).tolist(), val_anchors)))
    random.seed(TRAIN_SAMPLE_SEED)
    train_idx = np.sort(np.array(random.sample(np.flatnonzero(sp.s1_fold != VAL_FOLD).tolist(), train_anchors)))
    want = {"train": train_idx, "val": val_idx} if train_anchors else {"val": val_idx}

    # S1 text for exactly these businesses, streamed
    want_vals = {n: np.sort(truth.s1_values[i].astype(np.int64)) for n, i in want.items()}
    parts = {n: [] for n in want}
    for df in iter_source(d / "train_source1.tsv"):
        v = df["entity_id"].str.slice(3).astype(np.int64).to_numpy()
        for n in want:
            m = np.isin(v, want_vals[n])
            if m.any():
                sub = df.loc[m, ["business_name", "business_address", "country"]].copy()
                sub["s1_value"] = v[m]
                parts[n].append(sub)
    order = np.argsort(truth.s1_values)
    anchors = {}
    for n in want:
        a = pd.concat(parts[n], ignore_index=True)
        a["truth_row"] = order[np.searchsorted(truth.s1_values[order], a["s1_value"].to_numpy())]
        a["n_true"] = np.diff(truth.indptr)[a["truth_row"].to_numpy()]
        anchors[n] = a
        a[["s1_value", "country", "n_true"]].rename_axis("anchor").reset_index() \
            .to_parquet(out / f"anchors_{n}.parquet", index=False)
    del parts
    say("businesses: " + ", ".join(f"{len(a):,} {n}" for n, a in anchors.items()))

    fold0 = {s: np.sort(sp.tgt_values[s][sp.tgt_fold[s] == VAL_FOLD].astype(np.int64)) for s in (2, 3)}
    not_fold0 = lambda s, v: ~np.isin(v, fold0[s])
    only_fold0 = lambda s, v: np.isin(v, fold0[s])
    val_keep = only_fold0 if val_world else None
    rng = np.random.default_rng(20260926)
    stats = {"train_all": 0, "train_kept": 0, "train_pos": 0, "val": 0, "val_pos": 0}

    for ctry in sorted(set().union(*(set(a["country"]) for a in anchors.values()))):
        cands = {}
        for n in anchors:
            a = anchors[n]
            sel = np.flatnonzero((a["country"] == ctry).to_numpy())
            if not len(sel):
                continue
            index = build_index(src, keep=not_fold0 if n == "train" else val_keep, country=ctry,
                                freq=freq, workers=workers, keep_text=bool(k_wide > k and k_formula > 0))
            say(f"[{ctry}/{n}] index built: {index.n_targets:,} targets")
            if n == "train":   # leak guard, checked rather than assumed
                for s in (2, 3):
                    m = index.codes // SOURCE_BASE == s
                    if np.isin(index.codes[m] % SOURCE_BASE, fold0[s]).any():
                        raise RuntimeError("validation record in the training index")
            sub = a.iloc[sel]
            ak = compute_keys(sub["business_name"].tolist(), sub["business_address"].tolist(),
                              sub["country"].tolist(), workers=workers, freq=freq)
            if k_wide > k and k_formula > 0:
                sa = prepare(sub["business_name"].tolist(), sub["business_address"].tolist(),
                             workers=workers, countries=sub["country"].tolist())
                pool = prepare_pool(src, index, country=ctry,
                                    keep=not_fold0 if n == "train" else val_keep, workers=workers)
                c, cnt = select_candidates(index, ak, len(sel), sa["name"], sa["addr"],
                                           pool["name"], pool["addr"], cap=cap, k_keep=k,
                                           k_wide=k_wide, k_formula=k_formula, batch=batch,
                                           workers=workers, log=say)
                del sa, pool
            else:
                c = generate_candidates(index, ak, len(sel), cap=cap, k=k, batch=batch, workers=workers)
                cnt = pair_counts(index, ak, c.anchor, c.target, len(sel))
            cands[n] = dict(sel=sel, anchor=c.anchor.astype(np.int64), code=index.codes[c.target],
                            score=c.score, rank=c.rank, channels=c.channels, counts=cnt)
            say(f"[{ctry}/{n}] {len(sel):,} businesses -> {len(c):,} pairs")
            del index, ak, c

        need = np.unique(np.concatenate([c["code"] for c in cands.values()]))
        tcodes, T = _prepare_targets(src, ctry, need, workers)
        t_order = np.argsort(tcodes)

        for n, c in cands.items():
            a, sel = anchors[n], c["sel"]
            sub = a.iloc[sel]
            A = prepare(sub["business_name"].tolist(), sub["business_address"].tolist(),
                        workers=workers, countries=sub["country"].tolist())
            anchor, code = c["anchor"], c["code"]
            t_pos = t_order[np.searchsorted(tcodes[t_order], code)]
            label = label_candidates(truth, a["truth_row"].to_numpy()[sel], anchor, code)
            bounds = np.searchsorted(anchor, np.arange(0, len(sel) + anchors_per_chunk, anchors_per_chunk))
            for ci, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
                if lo == hi:
                    continue
                sl = slice(lo, hi)
                meta = {"anchor": anchor[sl], "score": c["score"][sl], "rank": c["rank"][sl],
                        "channels": c["channels"][sl], "source": (code[sl] // SOURCE_BASE).astype(np.int8),
                        **{k: v[sl] for k, v in c["counts"].items()}}
                X = compute_features(A, T, anchor[sl], t_pos[sl], [ctry] * (hi - lo), meta,
                                     freq=freq, workers=workers)
                y = label[sl]
                w = np.ones(hi - lo, np.float32)
                keep = np.ones(hi - lo, bool)
                if n == "train" and easy_rate < 1:
                    hard = ((X[:, F_INDEX["combo_rank"]] < 5) | (X[:, F_INDEX["ret_rank"]] < 3)
                            | (np.nan_to_num(X[:, F_INDEX["name_tset"]]) >= 0.8)
                            | (np.nan_to_num(X[:, F_INDEX["addr_tset"]]) >= 0.8))
                    easy_neg = (y == 0) & ~hard
                    keep = ~easy_neg | (rng.random(hi - lo) < easy_rate)
                    w[easy_neg] = 1.0 / easy_rate
                if n == "train":
                    stats["train_all"] += hi - lo
                    stats["train_kept"] += int(keep.sum())
                    stats["train_pos"] += int(y.sum())
                else:
                    stats["val"] += hi - lo
                    stats["val_pos"] += int(y.sum())
                df = pd.DataFrame(X[keep], columns=list(FEATURE_NAMES))
                df.insert(0, "weight", w[keep])
                df.insert(0, "label", y[keep])
                df.insert(0, "target_code", code[sl][keep])
                df.insert(0, "s1_value", a["s1_value"].to_numpy()[sel][anchor[sl][keep]])
                df.insert(0, "anchor", sel[anchor[sl][keep]].astype(np.int64))
                df.to_parquet(out / f"{n}_{ctry}_{ci:03d}.parquet", index=False)
                del X, df
            say(f"[{ctry}/{n}] features written")
            del A
        del T, tcodes, cands

    n_true_val = anchors["val"]["n_true"].sum()
    stats["val_blocking_recall"] = stats["val_pos"] / max(n_true_val, 1)
    (out / "build_report.json").write_text(json.dumps({
        "train_anchors": train_anchors, "val_anchors": len(val_idx), "easy_rate": easy_rate,
        "val_world": val_world, "k_wide": k_wide, "k_formula": k_formula,
        "cap": cap, "k": k, **stats}, indent=2, default=_plain))
    say(f"train pairs {stats['train_all']:,} -> kept {stats['train_kept']:,}; "
        f"val pairs {stats['val']:,}, blocking recall {stats['val_blocking_recall']:.4f}")
    return out


# --------------------------------------------------------------------------
# train + validate
# --------------------------------------------------------------------------

def train_and_validate(art_dir, pairs_name: str = "pairs", tag: str = "final",
                       t: float = 0.80, max_train_anchors: int = 0, log=print) -> Dict:
    """Train on <pairs>/train_*, score <pairs>/val_* with the competition
    metric (every validation business, blocking misses included) under the
    decision rule used for the submission: p >= t, then one owner."""
    say = _logger(log)
    art = Path(art_dir)
    pdir = art / pairs_name
    mmap = pdir / "_train_X.npy"          # disk-backed: see load_pair_parts
    X, y, anchor, w = load_pair_parts(sorted(pdir.glob("train_*.parquet")), mmap_path=mmap)
    if max_train_anchors:
        # learning curve: a fixed random subset of the training BUSINESSES
        ids = np.unique(anchor)
        pick = np.random.default_rng(3).choice(ids, size=min(max_train_anchors, len(ids)), replace=False)
        m = np.isin(anchor, pick)
        X, y, anchor, w = X[m], y[m], anchor[m], (None if w is None else w[m])
    say(f"train pairs {X.shape} from {len(np.unique(anchor)):,} businesses, "
        f"positives {y.mean():.3%}, weighted={w is not None}")
    m = train_matcher(X, y, anchor, weight=w)
    del X, y, anchor, w
    mmap.unlink(missing_ok=True)
    (art / "models").mkdir(exist_ok=True)
    model_path = art / "models" / f"matcher_{tag}.lgb"
    m.save(model_path)
    say(f"model: best iteration {m.meta['best_iteration']} -> {model_path}")

    vfiles = sorted(pdir.glob("val_*.parquet"))
    X, y, anchor, _ = load_pair_parts(vfiles, sort_anchors=True)
    p = m.predict(X)
    del X
    frames = [pd.read_parquet(f, columns=["anchor", "target_code"]) for f in vfiles]
    tv = pd.concat(frames, ignore_index=True)
    tv = tv.iloc[np.argsort(tv["anchor"].to_numpy(), kind="stable")]
    target = tv["target_code"].to_numpy(np.int64)
    va = pd.read_parquet(pdir / "anchors_val.parquet")
    n_true = va["n_true"].to_numpy()
    n = len(n_true)
    np.save(art / f"val_scores_{tag}.npy", p.astype(np.float32))

    found = np.bincount(anchor, weights=y, minlength=n)
    keep = one_owner(threshold(p, t), p, target, anchor)
    s = score_keep(keep, y, anchor, n_true)
    per_country = {c: score_keep(keep, y, anchor, n_true, (va["country"] == c).to_numpy())["macro_f05"]
                   for c in sorted(va["country"].unique())}
    # the threshold is re-checked on this model's own validation scores (a
    # different feature set can move the best cut-off); reported, not applied
    sweep = {round(float(tt), 2): score_keep(one_owner(threshold(p, tt), p, target, anchor),
                                             y, anchor, n_true)["macro_f05"]
             for tt in np.arange(0.5, 0.96, 0.05)}
    report = {
        "tag": tag, "threshold": t, "one_owner": True, **s, "threshold_sweep": sweep,
        "train_businesses": int(max_train_anchors) or None,
        "per_country": per_country,
        "blocking_ceiling": float(entity_f05_counts(found, np.zeros(n), n_true).mean()),
        "empty_baseline": float((n_true == 0).mean()),
        "model": m.meta, "top_features": dict(list(m.importance().items())[:15]),
    }
    (art / f"report_{tag}.json").write_text(json.dumps(report, indent=2, default=_plain))
    say(f"validation macro F0.5 {s['macro_f05']:.4f} (ceiling {report['blocking_ceiling']:.4f}); "
        f"per country {({k: round(v, 4) for k, v in per_country.items()})}")
    return report
