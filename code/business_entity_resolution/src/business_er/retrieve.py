"""Candidate generation (blocking): shrink ~10^13 possible pairs to a shortlist.

How it works, in three moves
----------------------------
1. KEYS.  Every record gets a handful of short "keys" from several channels --
   a rare name word, an address number plus an address word, a pair of name
   words, and so on.  Two records become candidates when they share a key.
   Every key starts with the country, because country agrees in 100% of the
   7.6M training true pairs, and it stays an open set of labels (France works
   without being listed anywhere).

2. INDEX.  Keys are hashed to 8-byte integers and stored per channel as two
   flat sorted arrays (key hash, target row).  Finding who shares a key is then
   a binary search, not a Python dict lookup, which is what lets 10M target
   records fit in a few GB.  A key shared by more than `cap` records is
   ignored: `enterprises` or `road` would otherwise match everything.

3. RANK.  A candidate's score is the sum, over the keys it shares with the
   anchor, of log(N / block size) -- the IDF idea: a key shared by 2 records is
   strong evidence, one shared by 300 is weak.  We keep the top-K per anchor
   and source, so a large source cannot crowd the other out.

Measured on the 25-fold dev slice (experiments/blocking_rank.py): the plain
union keeps 416 candidates per anchor for 0.991 recall; top-50 keeps 50 for
0.975 recall and an oracle F0.5 ceiling of 0.991.

What leaves this module is the scored-pair universe the model runs on, so it
is also exactly what candidate_pairs.tsv must contain.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .io import iter_source
from .normalize import address_views, expand_address_tokens, name_views

RETRIEVE_VERSION = "1.5.0"   # 1.5: optional extra address channels   # 1.1 leading zeros; 1.2 rarest tokens; 1.3 name_glued channel

CHANNELS: Tuple[str, ...] = (
    "name_exact", "name_sorted", "name_token", "name_pair",
    "addr_token", "num_token", "num_set", "name_num", "name_glued", "addr_exact",
)
# Optional channels (build with extra=True), measured on 20k validation
# businesses as additions to the selected candidates: ceiling 0.9880 -> 0.9905.
# They target true matches whose street numbers were truncated or altered
# ("403" vs "40", "D-199" vs "D-99") and generic names outranked by look-alikes.
#   addr_pair   pairs among the 3 rarest address tokens (numbers ignored)
#   addr_nonum  the whole cleaned address with number tokens removed
#   num_trunc   first 2 digits of each 3+-digit number + rarest address token
EXTRA_CHANNELS = ("addr_pair", "addr_nonum", "num_trunc")
ALL_CHANNELS = CHANNELS + EXTRA_CHANNELS
# Channels whose group size is also a feature: how many records in the pool
# carry exactly this cleaned name / address.  A name or address used by one
# record is strong evidence; one shared by an office building or a chain is not.
COUNT_CHANNELS = ("name_exact", "addr_exact")
MIN_EXACT_ADDR_LEN = 8   # shorter addresses ("delhi") are too generic to be a key

# Dropped before gluing a name into one word (name_glued channel).  Only
# unambiguous legal forms -- NOT "enterprises"/"services", which are often the
# distinctive half of a glued web name like "#adityaenterprises" -- plus the
# pieces of a web address.  Applied to both sides, so it is symmetric.
_GLUE_DROP = frozenset("""
llc llp ltd limited inc incorporated corp corporation co company pvt private plc
sarl sas sasu eurl sa gmbh
www http https com net org info biz in fr us
""".split())
MIN_GLUED_LEN = 6


def glued_name(folded_name: str) -> str:
    """Name with legal/web tokens dropped and the rest glued into one word:
    "VDR Cornerstone Pegasus LLC" and "vdrcornerstonepegasus.com" both give
    "vdrcornerstonepegasus".  "" when the result is too short to be specific.
    Shared by blocking (name_glued channel) and pair features."""
    g = "".join(t for t in folded_name.split() if t not in _GLUE_DROP)
    return g if len(g) >= MIN_GLUED_LEN else ""
CHANNEL_BIT = {c: 1 << i for i, c in enumerate(CHANNELS + ("addr_pair", "addr_nonum", "num_trunc"))}

# S2-100 and S3-100 both exist, so a target is identified by (source, value).
SOURCE_BASE = 2_000_000_000

# Scores use log(IDF_REF / block size) with a FIXED reference instead of the
# index size, so a candidate's score does not depend on how the pool is sharded
# (per-country indexes rank exactly like one global index).  ~10M is the size
# of the real target pools; any constant above the largest cap works.
IDF_REF = 10_000_000


def target_code(source: int | np.ndarray, value: int | np.ndarray):
    return np.asarray(source, dtype=np.int64) * SOURCE_BASE + np.asarray(value, dtype=np.int64)


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------

def parse_record(name: str, address: str) -> Tuple[str, List[str], List[str], List[str], str]:
    """(folded name, name tokens, address tokens, numbers) -- the raw material
    of every key.  Tokens are de-duplicated; order is decided later."""
    nv, av = name_views(name), address_views(address)
    nf, af = nv["folded"], av["folded"]
    ntok = sorted({t for t in nf.split() if len(t) >= 3})
    atok = sorted({t for t in af.split() if len(t) >= 4 and not t.isdigit()})
    # "09585" and "9585" are the same street number; sources disagree on padding
    nums = sorted({n.lstrip("0") or "0" for n in av["numbers"].split()})
    return nf, ntok, atok, nums, af


def make_keys(country: str, nf: str, ntok: List[str], atok: List[str],
              nums: List[str], af: str = "", extra: bool = False) -> Dict[str, List[str]]:
    """The blocking keys of one record, grouped by channel.

    ntok / atok must already be in PRIORITY order (rarest first when a token
    frequency table is used): the limits below keep only the first few, so a
    long record cannot explode the index -- and what we keep should be the
    distinctive words, not "apartment" and "floor".
    """
    c = country
    k: Dict[str, List[str]] = {ch: [] for ch in CHANNELS}
    if nf:
        k["name_exact"].append(f"{c}|{nf}")
        if ntok:
            k["name_sorted"].append(f"{c}|{' '.join(sorted(ntok))}")
    k["name_token"] = [f"{c}|{t}" for t in ntok[:6]]
    st = sorted(ntok[:4])                       # pair keys must not depend on order
    k["name_pair"] = [f"{c}|{st[i]}|{st[j]}" for i in range(len(st)) for j in range(i + 1, len(st))]
    k["addr_token"] = [f"{c}|a|{t}" for t in atok[:6]]
    k["num_token"] = [f"{c}|{n}|{t}" for n in nums[:2] for t in atok[:3]]
    if nums:
        k["num_set"].append(f"{c}|#|{'-'.join(nums)}")
    k["name_num"] = [f"{c}|{t}|{n}" for t in ntok[:3] for n in nums[:2]]
    # "VDR Cornerstone Pegasus LLC" and "vdrcornerstonepegasus.com" share no
    # word, but glue to the same string.  Uses every name token (even 1-2
    # letter ones like the "s r k" of "S.R.K Traders").
    glued = glued_name(nf)
    if glued:
        k["name_glued"].append(f"{c}|g|{glued}")
    if len(af) >= MIN_EXACT_ADDR_LEN:
        k["addr_exact"].append(f"{c}|x|{af}")
    if extra:
        r3 = sorted(atok[:3])
        k["addr_pair"] = [f"{c}|ap|{r3[i]}|{r3[j]}" for i in range(len(r3)) for j in range(i + 1, len(r3))]
        nonum = " ".join(sorted({t for t in af.split() if not any(ch.isdigit() for ch in t)}))
        k["addr_nonum"] = [f"{c}|an|{nonum}"] if len(nonum) >= 10 else []
        k["num_trunc"] = ([f"{c}|nt|{n[:2]}|{atok[0]}" for n in sorted({n for n in nums if len(n) >= 3})[:3]]
                          if atok else [])
    return k


def stable_hash(keys: Sequence[str]) -> np.ndarray:
    """Hash strings to uint64 identically in every process and every run.

    Python's built-in hash() is salted per process, so it cannot be used across
    worker processes or saved indexes.  pandas' hash_array uses a fixed key.
    """
    if not len(keys):
        return np.empty(0, dtype=np.uint64)
    return pd.util.hash_array(np.asarray(keys, dtype=object), categorize=False)


# --------------------------------------------------------------------------
# token rarity
# --------------------------------------------------------------------------

@dataclass
class TokenFreq:
    """How many records contain each token, counted on TRAINING records only.

    Stored as sorted uint64 hashes + uint32 counts rather than a Python dict:
    forked workers share numpy arrays for free, whereas a dict of millions of
    strings gets copied page by page into every worker as refcounts change.

    Lookup strings are "n|<country>|tok" (name) and "a|<country>|tok"
    (address), plus "*" in place of the country for the all-countries count,
    which is what a country absent from training (France) falls back to.
    A token missing from the table counts as 1: unseen means rare.
    """

    hashes: np.ndarray                # sorted uint64
    counts: np.ndarray                # uint32, aligned
    countries: frozenset              # countries the table was counted on

    def lookup(self, strings: Sequence[str]) -> np.ndarray:
        h = stable_hash(strings)
        if not len(h) or not len(self.hashes):
            return np.ones(len(h), dtype=np.int64)
        p = np.minimum(np.searchsorted(self.hashes, h), len(self.hashes) - 1)
        return np.where(self.hashes[p] == h, self.counts[p], 1).astype(np.int64)

    def save(self, path) -> None:
        np.savez(path, hashes=self.hashes, counts=self.counts,
                 countries=np.asarray(sorted(self.countries), dtype="U32"))

    @classmethod
    def load(cls, path) -> "TokenFreq":
        z = np.load(path)
        return cls(z["hashes"], z["counts"], frozenset(z["countries"].tolist()))


def _freq_strings(country: str, ntok: List[str], atok: List[str], known: frozenset) -> List[str]:
    scope = country if country in known else "*"
    return [f"n|{scope}|{t}" for t in ntok] + [f"a|{scope}|{t}" for t in atok]


def _count_task(args) -> Tuple[np.ndarray, np.ndarray]:
    names, addrs, countries = args
    strs: List[str] = []
    for n, a, c in zip(names, addrs, countries):
        _, ntok, atok, _, _ = parse_record(n, a)
        strs += [f"n|{c}|{t}" for t in ntok] + [f"a|{c}|{t}" for t in atok]
        strs += [f"n|*|{t}" for t in ntok] + [f"a|*|{t}" for t in atok]
    h, cnt = np.unique(stable_hash(strs), return_counts=True)
    return h, cnt.astype(np.uint32)


def _merge_counts(parts: List[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    h = np.concatenate([p[0] for p in parts])
    c = np.concatenate([p[1] for p in parts])
    u, inv = np.unique(h, return_inverse=True)
    return u, np.bincount(inv, weights=c).astype(np.uint32)


def build_token_freq(
    source_paths: Dict[int, str],
    keep: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    workers: int = 0,
    min_count: int = 2,
    read_chunk: int = 400_000,
    log: Callable[[str], None] = lambda m: None,
) -> TokenFreq:
    """Count, per token, how many records contain it (document frequency).

    Pass TRAINING files only.  keep(source, values) lets validation leave its
    own fold out, so validation tokens are "unseen" exactly as test tokens
    will be.  Tokens seen fewer than min_count times are dropped: missing
    already means count 1, and dropping them keeps the table small.
    """
    acc: Optional[Tuple[np.ndarray, np.ndarray]] = None
    countries = set()
    n_rec = 0
    for source in sorted(source_paths):
        for df in iter_source(source_paths[source], chunksize=read_chunk):
            if keep is not None:
                values = df["entity_id"].str.slice(3).astype(np.int64).to_numpy()
                df = df[keep(source, values)]
            if not len(df):
                continue
            countries.update(df["country"].unique().tolist())
            nm, ad, ct = (df[c].tolist() for c in ("business_name", "business_address", "country"))
            tasks = [(nm[s:s + 20_000], ad[s:s + 20_000], ct[s:s + 20_000])
                     for s in range(0, len(nm), 20_000)]
            if workers and workers > 1 and len(tasks) > 1:
                with get_context("fork").Pool(workers, initializer=_die_with_parent) as pool:
                    parts = pool.map(_count_task, tasks)
            else:
                parts = [_count_task(t) for t in tasks]
            if acc is not None:
                parts.append(acc)
            acc = _merge_counts(parts)
            n_rec += len(df)
            log(f"token counts: S{source}, {n_rec:,} records, {len(acc[0]):,} distinct")
    if acc is None:
        return TokenFreq(np.empty(0, np.uint64), np.empty(0, np.uint32), frozenset())
    m = acc[1] >= min_count
    return TokenFreq(acc[0][m], acc[1][m], frozenset(countries))


def _by_rarity(tokens: List[str], counts: np.ndarray) -> List[str]:
    """Rarest first; ties -> longer first (longer words are usually more
    specific), then alphabetical so the order is deterministic."""
    return [t for _, _, t in sorted(zip(counts.tolist(), (-len(t) for t in tokens), tokens))]


# --------------------------------------------------------------------------
# keys for many records
# --------------------------------------------------------------------------

def record_keys(name: str, address: str, country: str,
                freq: Optional[TokenFreq] = None) -> Dict[str, List[str]]:
    """Keys of a single record (convenient for tests and inspection)."""
    return keys_for_rows([name], [address], [country], freq)["_raw"][0]


KeyChunk = Dict[str, Tuple[np.ndarray, np.ndarray]]   # channel -> (hash, local row)


def keys_for_rows(names: Sequence[str], addrs: Sequence[str], countries: Sequence[str],
                  freq: Optional[TokenFreq] = None, extra: bool = False) -> dict:
    """Keys for a block of records, as flat (hash, row) arrays per channel.

    With a frequency table, all tokens of the block are looked up in ONE
    vectorised call, then each record's tokens are put in rarity order.
    Without one, tokens stay alphabetical (the original behaviour).
    The extra "_raw" entry holds the string keys for record_keys().
    """
    parsed = [parse_record(n, a) for n, a in zip(names, addrs)]
    if freq is not None:
        strs, bounds = [], []
        for (nf, ntok, atok, _, _), c in zip(parsed, countries):
            s0 = len(strs)
            strs += _freq_strings(c, ntok, atok, freq.countries)
            bounds.append(s0)
        cnt = freq.lookup(strs)
        ordered = []
        for (nf, ntok, atok, nums, af), s0 in zip(parsed, bounds):
            k = len(ntok)
            ordered.append((nf, _by_rarity(ntok, cnt[s0:s0 + k]),
                            _by_rarity(atok, cnt[s0 + k:s0 + k + len(atok)]), nums, af))
        parsed = ordered

    active = ALL_CHANNELS if extra else CHANNELS
    buf: Dict[str, Tuple[List[str], List[int]]] = {ch: ([], []) for ch in active}
    raw = []
    for row, ((nf, ntok, atok, nums, af), c) in enumerate(zip(parsed, countries)):
        rk = make_keys(c, nf, ntok, atok, nums, af, extra=extra)
        raw.append(rk)
        for ch, ks in rk.items():
            if ks:
                buf[ch][0].extend(ks)
                buf[ch][1].extend([row] * len(ks))
    out: dict = {ch: (stable_hash(ks), np.asarray(rs, dtype=np.int32)) for ch, (ks, rs) in buf.items()}
    out["_raw"] = raw
    # the cleaned name and expanded address, identical to features.prepare's
    # "name"/"addr" columns -- kept so the selection formula needs no second pass
    out["_text"] = ([p[0] for p in parsed],
                    [expand_address_tokens(p[4], c) for p, c in zip(parsed, countries)])
    return out


def _die_with_parent() -> None:
    """Pool initializer: ask Linux to SIGKILL this worker if its parent dies.

    Without it, killing the parent (e.g. by a memory cap) leaves orphaned
    workers holding gigabytes -- which is how the laptop ran out of memory.
    """
    try:
        import ctypes
        import signal
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGKILL))  # PR_SET_PDEATHSIG
    except Exception:
        pass


_FREQ: dict = {}   # frequency table shared with forked workers (read-only)


def _keys_task(args):
    start, names, addrs, countries = args
    extra = bool(_FREQ.get("extra"))
    out = keys_for_rows(names, addrs, countries, _FREQ.get("freq"), extra=extra)
    res = {ch: (out[ch][0], out[ch][1] + np.int32(start)) for ch in (ALL_CHANNELS if extra else CHANNELS)}
    if _FREQ.get("keep_text"):
        res["_text"] = out["_text"]
    return start, res


def compute_keys(
    names: Sequence[str], addrs: Sequence[str], countries: Sequence[str],
    workers: int = 0, chunk: int = 20_000, freq: Optional[TokenFreq] = None,
    keep_text: bool = False, extra: bool = False,
) -> KeyChunk:
    """keys_for_rows over many records, in parallel.  Rows keep their order.

    The SAME freq table must be used for targets and anchors, or the two sides
    keep different tokens and stop sharing keys.
    """
    n = len(names)
    tasks = [(s, names[s:s + chunk], addrs[s:s + chunk], countries[s:s + chunk])
             for s in range(0, n, chunk)]
    _FREQ["freq"] = freq
    _FREQ["keep_text"] = keep_text
    _FREQ["extra"] = extra
    try:
        if workers and workers > 1 and len(tasks) > 1:
            with get_context("fork").Pool(workers, initializer=_die_with_parent) as pool:
                parts = pool.map(_keys_task, tasks)
        else:
            parts = [_keys_task(t) for t in tasks]
    finally:
        _FREQ.clear()
    parts.sort(key=lambda p: p[0])
    out = {
        ch: (np.concatenate([p[1][ch][0] for p in parts]) if parts else np.empty(0, np.uint64),
             np.concatenate([p[1][ch][1] for p in parts]) if parts else np.empty(0, np.int32))
        for ch in (ALL_CHANNELS if extra else CHANNELS)
    }
    if keep_text:
        out["_text"] = pd.DataFrame({
            "name": pd.Series([x for p in parts for x in p[1]["_text"][0]], dtype="str"),
            "addr": pd.Series([x for p in parts for x in p[1]["_text"][1]], dtype="str")})
    return out


# --------------------------------------------------------------------------
# the target index
# --------------------------------------------------------------------------

@dataclass
class KeyIndex:
    """Sorted postings per channel, plus the identity of every target row."""

    keys: Dict[str, np.ndarray]    # channel -> sorted uint64 key hashes
    rows: Dict[str, np.ndarray]    # channel -> int32 target row, aligned
    codes: np.ndarray              # int64 target_code per row
    country: Optional[str] = None  # shard label; None = all countries
    # COUNT_CHANNELS -> per target row, how many pool records share its key
    # (0 when the row has no such key, e.g. an empty address)
    row_counts: Optional[Dict[str, np.ndarray]] = None
    # cleaned name / expanded address per row (build_index(keep_text=True))
    text: Optional[pd.DataFrame] = None

    @property
    def n_targets(self) -> int:
        return len(self.codes)

    @property
    def sources(self) -> np.ndarray:
        return (self.codes // SOURCE_BASE).astype(np.int8)

    def n_postings(self) -> int:
        return sum(len(v) for v in self.keys.values())


def build_index(
    source_paths: Dict[int, str],
    keep: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    country: Optional[str] = None,
    freq: Optional[TokenFreq] = None,
    workers: int = 0,
    read_chunk: int = 400_000,
    keep_text: bool = False,
    extra: bool = False,
    log: Callable[[str], None] = lambda m: None,
) -> KeyIndex:
    """Index the S2/S3 files, streaming them in chunks.

    keep(source, values) -> bool mask lets validation restrict the pool to one
    fold's records.  Text is discarded as soon as its keys are made.

    country: index only records with this exact country label.  Every key
    starts with the country, so records of different countries can never share
    a key -- sharding by country gives IDENTICAL candidates at a fraction of the
    peak memory.  Any label works (open set); None indexes everything.
    """
    active = ALL_CHANNELS if extra else CHANNELS
    key_parts: Dict[str, List[np.ndarray]] = {ch: [] for ch in active}
    row_parts: Dict[str, List[np.ndarray]] = {ch: [] for ch in active}
    code_parts: List[np.ndarray] = []
    text_parts: List[pd.DataFrame] = []
    offset = 0
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
            kc = compute_keys(df["business_name"].tolist(), df["business_address"].tolist(),
                              df["country"].tolist(), workers=workers, freq=freq, keep_text=keep_text,
                              extra=extra)
            if keep_text:
                text_parts.append(kc["_text"])
            for ch in active:
                key_parts[ch].append(kc[ch][0])
                row_parts[ch].append(kc[ch][1] + np.int32(offset))
            code_parts.append(target_code(source, values))
            offset += len(df)
            log(f"indexed S{source}{'' if country is None else ' ' + country}: {offset:,} targets")

    keys, rows = {}, {}
    for ch in active:
        k = np.concatenate(key_parts[ch]) if key_parts[ch] else np.empty(0, np.uint64)
        r = np.concatenate(row_parts[ch]) if row_parts[ch] else np.empty(0, np.int32)
        key_parts[ch] = row_parts[ch] = None
        o = np.argsort(k, kind="stable")
        keys[ch], rows[ch] = k[o], r[o]
        del k, r, o
    n_rows = offset
    row_counts = {}
    for ch in COUNT_CHANNELS:
        k, r = keys[ch], rows[ch]
        cnt = np.zeros(n_rows, np.int32)
        if len(k):
            starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
            sizes = np.diff(np.r_[starts, len(k)])
            cnt[r] = np.repeat(sizes, sizes).astype(np.int32)
        row_counts[ch] = cnt
    return KeyIndex(
        keys=keys, rows=rows,
        codes=np.concatenate(code_parts) if code_parts else np.empty(0, np.int64),
        country=country, row_counts=row_counts,
        text=(pd.concat(text_parts, ignore_index=True) if text_parts else None) if keep_text else None,
    )


# --------------------------------------------------------------------------
# querying
# --------------------------------------------------------------------------

@dataclass
class Candidates:
    """The scored-pair universe: one row per (anchor, candidate target).

    Sorted by anchor, then source, then score descending.
    """

    anchor: np.ndarray    # int32 anchor position (row in the anchor list)
    target: np.ndarray    # int32 row in KeyIndex
    score: np.ndarray     # float32 summed IDF weight of shared keys
    channels: np.ndarray  # uint16 bitmask of CHANNEL_BIT that produced the pair
    rank: np.ndarray      # int16 0-based rank within (anchor, source)

    def __len__(self) -> int:
        return len(self.anchor)


_Q: dict = {}   # shared with forked workers: index + anchor keys + settings


def _rank_batch(bounds: Tuple[int, int]):
    lo_a, hi_a = bounds
    idx: KeyIndex = _Q["index"]
    akeys: KeyChunk = _Q["akeys"]
    cap, k, use = _Q["cap"], _Q["k"], _Q["channels"]
    n_t = idx.n_targets
    log_n = math.log(IDF_REF)

    codes, wts, bits = [], [], []
    for ch in use:
        ah, ar = akeys[ch]
        a0, a1 = np.searchsorted(ar, [lo_a, hi_a])   # anchor rows are sorted
        ah, ar = ah[a0:a1], ar[a0:a1]
        if not len(ah):
            continue
        sk, si = idx.keys[ch], idx.rows[ch]
        lo = np.searchsorted(sk, ah, "left")
        sz = np.searchsorted(sk, ah, "right") - lo
        ok = (sz > 0) & (sz <= cap)
        lo, sz, ar = lo[ok], sz[ok], ar[ok]
        if not len(lo):
            continue
        tot = int(sz.sum())
        offs = np.arange(tot) - np.repeat(np.cumsum(sz) - sz, sz)
        trow = si[np.repeat(lo, sz) + offs]
        codes.append(np.repeat(ar - lo_a, sz).astype(np.int64) * n_t + trow)
        wts.append(np.repeat((log_n - np.log(sz)).astype(np.float32), sz))
        bits.append(np.full(tot, CHANNEL_BIT[ch], dtype=np.uint16))

    if not codes:
        e = np.empty(0, np.int32)
        return e, e, np.empty(0, np.float32), np.empty(0, np.uint16), np.empty(0, np.int16)

    code = np.concatenate(codes)
    uniq, inv = np.unique(code, return_inverse=True)
    score = np.bincount(inv, weights=np.concatenate(wts)).astype(np.float32)
    chmask = np.zeros(len(uniq), dtype=np.uint16)
    np.bitwise_or.at(chmask, inv, np.concatenate(bits))

    anc = (uniq // n_t).astype(np.int32)
    trow = (uniq % n_t).astype(np.int32)
    src = idx.codes[trow] // SOURCE_BASE
    # anchor, then source, then best score, then target row as a stable tiebreak
    order = np.lexsort((trow, -score, src, anc))
    anc, trow, score, chmask, src = anc[order], trow[order], score[order], chmask[order], src[order]
    grp = anc.astype(np.int64) * 8 + src
    rank = np.arange(len(grp)) - np.searchsorted(grp, grp, "left")
    keep = rank < k
    return (anc[keep] + np.int32(lo_a), trow[keep], score[keep], chmask[keep],
            rank[keep].astype(np.int16))


def generate_candidates(
    index: KeyIndex,
    anchor_keys: KeyChunk,
    n_anchors: int,
    cap: int = 300,
    k: int = 25,
    batch: int = 500,
    workers: int = 0,
    channels: Optional[Sequence[str]] = None,
) -> Candidates:
    """Top-k candidates per anchor PER SOURCE (so up to 2k per anchor).

    anchor_keys comes from compute_keys on the anchor (S1) records.  Anchors
    with no shared key simply get no rows: the writer still emits them with an
    empty list.

    channels restricts which channels are searched (default: all) -- used to
    measure each channel's contribution.
    """
    use = tuple(CHANNELS if channels is None else channels)
    unknown = set(use) - set(ALL_CHANNELS)
    if unknown:
        raise ValueError(f"unknown channels {sorted(unknown)}")
    _Q.update(index=index, akeys=anchor_keys, cap=cap, k=k, channels=use)
    bounds = [(s, min(s + batch, n_anchors)) for s in range(0, n_anchors, batch)]
    try:
        if workers and workers > 1 and len(bounds) > 1:
            with get_context("fork").Pool(workers, initializer=_die_with_parent) as pool:
                parts = pool.map(_rank_batch, bounds, chunksize=4)
        else:
            parts = [_rank_batch(b) for b in bounds]
    finally:
        _Q.clear()
    cat = lambda i, dt: (np.concatenate([p[i] for p in parts]) if parts else np.empty(0, dt))
    return Candidates(
        anchor=cat(0, np.int32), target=cat(1, np.int32), score=cat(2, np.float32),
        channels=cat(3, np.uint16), rank=cat(4, np.int16),
    )


def pair_counts(index: KeyIndex, anchor_keys: KeyChunk, anchor: np.ndarray,
                target: np.ndarray, n_anchors: int) -> Dict[str, np.ndarray]:
    """Per pair, how many POOL records share the exact cleaned name / address
    of the candidate (t_*) and of the S1 record (a_*).  0 = no such key.

    These are unsupervised counts over the pool being searched -- the
    training pool in training, the test pool at test time."""
    out = {}
    for ch, tag in (("name_exact", "name"), ("addr_exact", "addr")):
        out[f"t_{tag}_cnt"] = index.row_counts[ch][target].astype(np.float32)
        ah, ar = anchor_keys[ch]
        per_anchor = np.zeros(n_anchors, np.float32)
        if len(ah):
            sk = index.keys[ch]
            size = np.searchsorted(sk, ah, "right") - np.searchsorted(sk, ah, "left")
            per_anchor[ar] = size
        out[f"a_{tag}_cnt"] = per_anchor[anchor]
    return out


def candidate_id_lists(cands: Candidates, index: KeyIndex, n_anchors: int) -> List[List[str]]:
    """Per anchor, candidate ids as "S2-47" strings -- for writing and scoring."""
    codes = index.codes[cands.target]
    src = codes // SOURCE_BASE
    val = codes % SOURCE_BASE
    out: List[List[str]] = [[] for _ in range(n_anchors)]
    for a, s, v in zip(cands.anchor.tolist(), src.tolist(), val.tolist()):
        out[a].append(f"S{s}-{v}")
    return out
