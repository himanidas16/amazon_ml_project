"""Probe: can NEW blocking channels recover the true matches the current
selection still misses (3.5% of true pairs; 80% of them have a name or
address >= 80% similar to S1)?

Channels tested (standalone, the production code is untouched):
  addr_pair    pairs among the 3 rarest address tokens (numbers ignored)
  addr_nonum   the whole cleaned address with number tokens removed
  num_trunc    first 2 digits of each number (len >= 3) + rarest address token
               -- catches "403" vs "40", "695" vs "69"

For the 20k validation businesses vs the full train pool, each new channel's
candidates (IDF-ranked, top --k per source per channel) are added to the
current selected candidates (artifacts/pairs_sel/val_*); the gain in recall,
candidates per business and oracle ceiling is reported.
"""
import argparse
import glob
import math
import random
import resource
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.io import iter_source  # noqa: E402
from business_er.metrics import entity_f05_counts  # noqa: E402
from business_er.retrieve import (IDF_REF, SOURCE_BASE, TokenFreq, _by_rarity,  # noqa: E402
                                  _die_with_parent, _freq_strings, parse_record, stable_hash, target_code)
from business_er.splits import load_split, load_truth_csr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--cap", type=int, default=3000)
ap.add_argument("--workers", type=int, default=12)
args = ap.parse_args()
D = ROOT / "amazon_ml_dataset/student_resource/dataset/train"
ART = ROOT / "artifacts"
t0 = time.time()
mark = lambda m: print(f"[{time.time() - t0:7.1f}s {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:4.1f}GB] {m}", flush=True)
NEW = ("addr_pair", "addr_nonum", "num_trunc")
freq = TokenFreq.load(ART / "token_freq_excl_fold0.npz")


def extra_keys_task(args_):
    names, addrs, countries = args_
    parsed = [parse_record(n, a) for n, a in zip(names, addrs)]
    strs, bounds = [], []
    for (nf, ntok, atok, _, _), c in zip(parsed, countries):
        bounds.append(len(strs)); strs += _freq_strings(c, ntok, atok, freq.countries)
    cnt = freq.lookup(strs)
    out = {ch: ([], []) for ch in NEW}
    for row, ((nf, ntok, atok, nums, af), c, s0) in enumerate(zip(parsed, countries, bounds)):
        k = len(ntok)
        atok_r = _by_rarity(atok, cnt[s0 + k:s0 + k + len(atok)])
        r3 = sorted(atok_r[:3])
        keys = {"addr_pair": [f"{c}|ap|{r3[i]}|{r3[j]}" for i in range(len(r3)) for j in range(i + 1, len(r3))],
                "addr_nonum": [], "num_trunc": []}
        nonum = " ".join(sorted({t for t in af.split() if not any(ch.isdigit() for ch in t)}))
        if len(nonum) >= 10:
            keys["addr_nonum"].append(f"{c}|an|{nonum}")
        if atok_r:
            keys["num_trunc"] = [f"{c}|nt|{n[:2]}|{atok_r[0]}" for n in sorted({n for n in nums if len(n) >= 3})[:3]]
        for ch, ks in keys.items():
            out[ch][0].extend(ks); out[ch][1].extend([row] * len(ks))
    return {ch: (stable_hash(ks), np.asarray(rs, np.int32)) for ch, (ks, rs) in out.items()}


def extra_keys(names, addrs, countries):
    tasks = [(names[s:s + 20000], addrs[s:s + 20000], countries[s:s + 20000]) for s in range(0, len(names), 20000)]
    with get_context("fork").Pool(args.workers, initializer=_die_with_parent) as pool:
        parts = pool.map(extra_keys_task, tasks)
    off = np.cumsum([0] + [len(t[0]) for t in tasks])[:-1]
    return {ch: (np.concatenate([p[ch][0] for p in parts]),
                 np.concatenate([p[ch][1] + np.int32(o) for p, o in zip(parts, off)])) for ch in NEW}


sp = load_split(ART / "split_5fold.npz")
truth = load_truth_csr(D / "train_ground_truth.tsv")
anchors = pd.read_parquet(ART / "pairs_sel/anchors_val.parquet")
want = np.sort(anchors["s1_value"].to_numpy())
parts = []
for df in iter_source(D / "train_source1.tsv"):
    v = df["entity_id"].str.slice(3).astype(np.int64).to_numpy()
    m = np.isin(v, want)
    if m.any():
        sub = df.loc[m, ["business_name", "business_address", "country"]].copy(); sub["s1_value"] = v[m]
        parts.append(sub)
a = pd.concat(parts).set_index("s1_value").loc[anchors["s1_value"]].reset_index()   # anchors_val order
mark(f"{len(a):,} validation businesses")

new_pairs = []   # (anchor row, target code)
for ctry in sorted(a["country"].unique()):
    sel = np.flatnonzero((a["country"] == ctry).to_numpy())
    kp, rp, codes = {ch: [] for ch in NEW}, {ch: [] for ch in NEW}, []
    off = 0
    for s in (2, 3):
        for df in iter_source(D / f"train_source{s}.tsv"):
            df = df[df["country"] == ctry]
            if not len(df):
                continue
            ek = extra_keys(df["business_name"].tolist(), df["business_address"].tolist(), df["country"].tolist())
            for ch in NEW:
                kp[ch].append(ek[ch][0]); rp[ch].append(ek[ch][1] + np.int32(off))
            codes.append(target_code(s, df["entity_id"].str.slice(3).astype(np.int64).to_numpy()))
            off += len(df)
    codes = np.concatenate(codes)
    src = codes // SOURCE_BASE
    sub = a.iloc[sel]
    ak = extra_keys(sub["business_name"].tolist(), sub["business_address"].tolist(), sub["country"].tolist())
    for ch in NEW:
        k = np.concatenate(kp[ch]); r = np.concatenate(rp[ch]); o = np.argsort(k, kind="stable"); k, r = k[o], r[o]
        ah, ar = ak[ch]
        lo = np.searchsorted(k, ah, "left"); sz = np.searchsorted(k, ah, "right") - lo
        ok = (sz > 0) & (sz <= args.cap); lo, sz, ar = lo[ok], sz[ok], ar[ok]
        tot = int(sz.sum())
        offs = np.arange(tot) - np.repeat(np.cumsum(sz) - sz, sz)
        trow = r[np.repeat(lo, sz) + offs]
        anc = np.repeat(ar, sz).astype(np.int64)
        w = np.repeat(math.log(IDF_REF) - np.log(sz), sz)
        # top-k per (anchor, source) by weight, ties by row
        grp = anc * 8 + src[trow]
        oo = np.lexsort((trow, -w, grp))
        st = np.flatnonzero(np.r_[True, grp[oo][1:] != grp[oo][:-1]]) if len(oo) else np.empty(0, int)
        rank = np.empty(len(oo), np.int64); rank[oo] = np.arange(len(oo)) - np.repeat(st, np.diff(np.r_[st, len(oo)]))
        keep = rank < args.k
        new_pairs.append(pd.DataFrame({"anchor": sel[anc[keep]], "code": codes[trow[keep]], "ch": ch}))
        mark(f"[{ctry}] {ch}: {int(keep.sum()):,} pairs (index {len(k):,} keys)")
    del kp, rp

newp = pd.concat(new_pairs).drop_duplicates(["anchor", "code", "ch"])
cur = pd.concat([pd.read_parquet(f, columns=["anchor", "target_code", "label"]) for f in glob.glob(str(ART / "pairs_sel/val_*.parquet"))])
cur_set = set(zip(cur["anchor"].tolist(), cur["target_code"].tolist()))
# true pairs per validation business
order = np.argsort(truth.s1_values)
rows = order[np.searchsorted(truth.s1_values[order], a["s1_value"].to_numpy())]
tset = set()
for i, r in enumerate(rows):
    v, s = truth.targets_of(int(r))
    tset.update((i, int(c)) for c in target_code(s.astype(np.int64), v.astype(np.int64)))
n_true = anchors["n_true"].to_numpy(); n = len(n_true)


def report(name, pairs):
    found = np.zeros(n)
    for i, c in pairs:
        if (i, c) in tset:
            found[i] += 1
    f = entity_f05_counts(found, np.zeros(n), n_true)
    print(f"  {name:<36}{found.sum() / n_true.sum():>9.4f}{len(pairs) / n:>9.1f}{f.mean():>10.4f}"
          f"{f[(a['country'] == 'India').to_numpy()].mean():>9.4f}{f[(a['country'] == 'US').to_numpy()].mean():>9.4f}")


print(f"\n  {'candidate set':<36}{'recall':>9}{'cand/S1':>9}{'ceiling':>10}{'India':>9}{'US':>9}")
report("current selection", cur_set)
allnew = set()
for ch in NEW:
    add = set(zip(newp.loc[newp.ch == ch, "anchor"].tolist(), newp.loc[newp.ch == ch, "code"].tolist()))
    report(f"+ {ch}", cur_set | add)
    allnew |= add
report("+ all three", cur_set | allnew)
mark("done")
