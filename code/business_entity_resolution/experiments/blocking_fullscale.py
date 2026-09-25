"""Step 5 check at realistic scale, using business_er.retrieve.

Anchors: a sample of S1 records from fold 0 of the 5-fold split.
Pool:    --pool all  -> every train S2/S3 record (10.3M, like the 10M test pool)
         --pool fold -> only fold 0's records (2.06M)

Blocking is unsupervised (no labels are used to build keys), so searching the
full pool leaks nothing; labels are used only to measure recall afterwards.

The index is built ONE COUNTRY AT A TIME (keys are country-scoped, so this is
exact) and freed before the next, which keeps peak memory near the largest
country's share instead of the whole pool.

Run from repo root, through the memory-capped launcher:
    code/business_entity_resolution/experiments/run_capped.sh \
        code/business_entity_resolution/experiments/blocking_fullscale.py --pool all --diagnose
"""
import argparse
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "code/business_entity_resolution/src"))

from business_er.io import iter_source, read_source  # noqa: E402
from business_er.retrieve import (  # noqa: E402
    CHANNELS, SOURCE_BASE, TokenFreq, build_index, build_token_freq, compute_keys,
    generate_candidates, target_code,
)
from business_er.splits import load_split, load_truth_csr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", choices=["all", "fold"], default="all")
ap.add_argument("--anchors", type=int, default=20_000)
ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--caps", default="300,1000,3000")
ap.add_argument("--ks", default="5,10,25,50")
ap.add_argument("--freq", action="store_true",
                help="order tokens by rarity (counts from train records outside fold 0)")
ap.add_argument("--batch", type=int, default=500,
                help="anchors per query batch; smaller = less memory at high caps")
ap.add_argument("--drop-channels", default="",
                help="comma-separated channels to leave out, e.g. name_glued")
ap.add_argument("--diagnose", action="store_true",
                help="split misses into never-found vs ranked-too-low, print examples")
args = ap.parse_args()

D = ROOT / "amazon_ml_dataset/student_resource/dataset/train"
t0 = time.time()


def mark(m):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"[{time.time() - t0:6.1f}s {rss:4.1f}GB] {m}", flush=True)


sp = load_split(ROOT / "artifacts/split_5fold.npz")
keep = None
if args.pool == "fold":
    fold_vals = {s: np.sort(sp.tgt_values[s][sp.tgt_fold[s] == 0]) for s in (2, 3)}
    keep = lambda s, v: np.isin(v, fold_vals[s])

# ---- token rarity table: training records OUTSIDE fold 0 only -------------
# Validation records must be "unseen" by the table, exactly as test records
# are unseen by a table counted on the training files.
freq = None
if args.freq:
    fpath = ROOT / "artifacts/token_freq_excl_fold0.npz"
    if fpath.exists():
        freq = TokenFreq.load(fpath)
    else:
        s1_out = np.sort(sp.s1_values[sp.s1_fold != 0].astype(np.int64))
        t_out = {s: np.sort(sp.tgt_values[s][sp.tgt_fold[s] != 0].astype(np.int64)) for s in (2, 3)}
        keep_out = lambda s, v: np.isin(v, s1_out if s == 1 else t_out[s])
        freq = build_token_freq({1: D / "train_source1.tsv", 2: D / "train_source2.tsv",
                                 3: D / "train_source3.tsv"}, keep=keep_out,
                                workers=args.workers, log=mark)
        freq.save(fpath)
    mark(f"token table: {len(freq.hashes):,} tokens (count >= 2), countries {sorted(freq.countries)}")

# ---- anchors first (small parent process before any workers fork) ---------
truth = load_truth_csr(D / "train_ground_truth.tsv")
f0 = np.flatnonzero(sp.s1_fold == 0)
random.seed(11)
pick = np.sort(np.array(random.sample(f0.tolist(), args.anchors)))
s1 = read_source(D / "train_source1.tsv", "S1-")
s1_vals = s1["entity_id"].str.slice(3).astype(np.int64).to_numpy()
order = np.argsort(s1_vals)
rows = order[np.searchsorted(s1_vals[order], truth.s1_values[pick])]
a = s1.iloc[rows].reset_index(drop=True)
del s1, s1_vals, order
n_true = np.diff(truth.indptr)[pick].astype(np.int64)
print(f"  {len(pick):,} anchors; singleton rate {(n_true == 0).mean():.3f}, "
      f"mean matches {n_true[n_true > 0].mean():.2f}")
mark("anchors loaded")

caps = [int(x) for x in args.caps.split(",")]
use_ch = [c for c in CHANNELS if c not in set(filter(None, args.drop_channels.split(",")))]
print(f"  channels: {', '.join(use_ch)}")
ks = [int(x) for x in args.ks.split(",")]
hits = {(c, k): np.zeros(len(pick)) for c in caps for k in ks}
ncand = {(c, k): np.zeros(len(pick)) for c in caps for k in ks}
qtime = {c: 0.0 for c in caps}
chan_tp = np.zeros(len(CHANNELS)); chan_n = 0
DIAG_CAP, DIAG_K = 3000, 25
diag_rank = []          # per true pair: rank within (anchor, source), -1 = never found
diag_pair = []          # (global anchor idx, target code) for examples

for ctry in sorted(a["country"].unique()):
    sel = np.flatnonzero((a["country"] == ctry).to_numpy())
    index = build_index({2: D / "train_source2.tsv", 3: D / "train_source3.tsv"},
                        keep=keep, country=ctry, freq=freq, workers=args.workers,
                        log=lambda m: None)
    mark(f"[{ctry}] index: {index.n_targets:,} targets, {index.n_postings():,} postings")
    sub = a.iloc[sel]
    akeys = compute_keys(sub["business_name"].tolist(), sub["business_address"].tolist(),
                         sub["country"].tolist(), workers=args.workers, freq=freq)

    # true pairs of these anchors as (local anchor, target row) codes
    code_order = np.argsort(index.codes); sorted_codes = index.codes[code_order]
    tr, tr_anchor, tr_code = [], [], []
    for li, gi in enumerate(sel):
        v, s_ = truth.targets_of(int(pick[gi]))
        cds = target_code(s_.astype(np.int64), v.astype(np.int64))
        p = np.searchsorted(sorted_codes, cds)
        assert (sorted_codes[np.minimum(p, len(sorted_codes) - 1)] == cds).all(), \
            "true match missing from its country's pool"
        tr.extend((li * index.n_targets + code_order[p]).tolist())
        tr_anchor.extend([gi] * len(cds)); tr_code.extend(cds.tolist())
    tr = np.asarray(tr, np.int64)
    trs = np.sort(tr)

    for cap in caps:
        tq = time.time()
        c = generate_candidates(index, akeys, len(sel), cap=cap, k=max(ks), workers=args.workers,
                                batch=args.batch, channels=use_ch)
        qtime[cap] += time.time() - tq
        pair = c.anchor.astype(np.int64) * index.n_targets + c.target
        is_true = np.isin(pair, trs)
        for k in ks:
            m = c.rank < k
            hits[cap, k][sel] = np.bincount(c.anchor[m], weights=is_true[m], minlength=len(sel))
            ncand[cap, k][sel] = np.bincount(c.anchor[m], minlength=len(sel))
        if cap == 1000:
            tp_mask = c.channels[is_true]
            chan_tp += [((tp_mask >> i) & 1).sum() for i in range(len(CHANNELS))]
            chan_n += len(tp_mask)
        del c, pair, is_true

    if args.diagnose:
        # whole union (k huge) at the loosest cap: rank of every true pair.
        # Small batches and few workers: this union is ~3000 candidates/anchor.
        c = generate_candidates(index, akeys, len(sel), cap=DIAG_CAP, k=100_000,
                                batch=100, workers=4, channels=use_ch)
        pair = c.anchor.astype(np.int64) * index.n_targets + c.target
        o = np.argsort(pair); sp_ = pair[o]
        p = np.searchsorted(sp_, tr)
        ok = (p < len(sp_)) & (sp_[np.minimum(p, len(sp_) - 1)] == tr)
        r = np.full(len(tr), -1); r[ok] = c.rank[o[p[ok]]]
        diag_rank.extend(r.tolist()); diag_pair.extend(zip(tr_anchor, tr_code))
        del c, pair, o, sp_
    del index, akeys
    mark(f"[{ctry}] done, index freed")

print(f"\n  {'cap':>6}{'k/src':>7}{'recall':>9}{'cand/S1':>10}{'p95':>7}{'oracle F0.5':>13}")
for cap in caps:
    for k in ks:
        h, nc = hits[cap, k], ncand[cap, k]
        f = np.where(n_true == 0, 1.0, 1.25 * h / np.maximum(1.25 * h + 0.25 * (n_true - h), 1e-12))
        print(f"  {cap:>6}{k:>7}{h.sum() / n_true.sum():>9.4f}{nc.mean():>10.1f}"
              f"{np.percentile(nc, 95):>7.0f}{f.mean():>13.4f}")
    print(f"         query {qtime[cap]:.1f}s -> {qtime[cap] * 1.73e6 / len(pick) / 60:.1f} min "
          f"for 1.73M test anchors")
if chan_n:
    print("  channel share of found true pairs (cap 1000): " + ", ".join(
        f"{ch}={chan_tp[i] / chan_n:.2f}" for i, ch in enumerate(CHANNELS)))

if args.diagnose:
    r = np.asarray(diag_rank); n = len(r); ok = r >= 0
    print(f"\n=== diagnosis at cap {DIAG_CAP}, k/src {DIAG_K}: {n:,} true pairs ===")
    print(f"  kept (rank < {DIAG_K})           {(ok & (r < DIAG_K)).sum() / n:.4f}")
    print(f"  found, ranked too low       {(ok & (r >= DIAG_K)).sum() / n:.4f}")
    for lo_, hi_ in ((25, 50), (50, 100), (100, 300), (300, 10**9)):
        print(f"      rank {lo_:>4}-{hi_ if hi_ < 10**9 else 'inf':<4}           "
              f"{(ok & (r >= lo_) & (r < hi_)).sum() / n:.4f}")
    print(f"  never found (no usable key) {(~ok).sum() / n:.4f}")

    rng = np.random.default_rng(3)
    miss_low = np.flatnonzero(ok & (r >= DIAG_K)); miss_none = np.flatnonzero(~ok)
    ex = {"RANKED TOO LOW": rng.choice(miss_low, min(8, len(miss_low)), replace=False),
          "NEVER FOUND": rng.choice(miss_none, min(8, len(miss_none)), replace=False)}
    want = {diag_pair[i][1] for v in ex.values() for i in v}
    text = {}
    for s_ in (2, 3):
        for df in iter_source(D / f"train_source{s_}.tsv"):
            codes_ = target_code(s_, df["entity_id"].str.slice(3).astype(np.int64).to_numpy())
            m = np.isin(codes_, list(want))
            for cd, nm, ad in zip(codes_[m], df["business_name"][m], df["business_address"][m]):
                text[int(cd)] = (nm, ad)
    for label, idxs in ex.items():
        print(f"\n  --- {label} ---")
        for i in idxs:
            gi, cd = diag_pair[i]
            print(f"  S1  {a['business_name'].iat[gi]!r:<45} {a['business_address'].iat[gi]!r}")
            print(f"  S{cd // SOURCE_BASE}  {text[cd][0]!r:<45} {text[cd][1]!r}   rank={r[i]}")
mark("done")
