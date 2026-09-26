"""Pair features: describe each (S1 record, candidate) pair with numbers.

Blocking (retrieve.py) hands us ~50 candidates per S1 record.  The model in
Step 7 cannot read text, so for every pair we answer the same fixed list of
questions -- "how similar are the names?", "do the street numbers agree?",
"is the address missing?" -- and the answers become one row of a table.

Principles
----------
* Several similarity measures per field, because each catches different
  noise: edit ratio (typos), Jaro-Winkler (typos near the start), token-sort
  (word order), token-set (extra words such as "Consulting Consulting").
* Rarity matters.  Sharing "nematech" is strong evidence, sharing "solutions"
  is weak, so word overlap is weighted by log(IDF_REF / count) from the same
  TRAINING-only token table blocking uses.
* Blanks are never evidence.  If either side of a field is empty, its
  similarities are NaN ("unknown"), not 1.0.  LightGBM handles NaN natively
  and learns what missingness means.  Two empty addresses do not "match".
* A number conflict is evidence, not a veto: street numbers drift (2046 vs
  2048) in real true pairs, so we give the gap as a number and let the model
  decide.
* No country feature and no ids: the model must not memorise US vs India
  (France never appears in training) or learn from id numbering.
* Context: a candidate is judged partly against its rivals -- "is this the
  best name match this S1 record has?" -- which separates a true match from a
  sibling branch with a similar name.

All features for one S1 record are computed in the same block, so context
features see every rival.  Blocks run in parallel worker processes; only
plain lists and numpy arrays cross the process boundary.
"""

from __future__ import annotations

import math
import re
from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .normalize import address_views, has_indic, name_views
from .retrieve import CHANNELS, IDF_REF, TokenFreq, _die_with_parent, glued_name

FEATURES_VERSION = "1.2.0"   # 1.1: country abbreviations; 1.2: joined/prefix numbers, raw-width postal, pool counts

_DIGITS = re.compile(r"\d+")
POSTAL_LENGTHS = (5, 6)   # US ZIP / France code postal = 5 digits, India PIN = 6


# --------------------------------------------------------------------------
# per-record preparation (done once per record, not once per pair)
# --------------------------------------------------------------------------

PREP_COLUMNS = ("name", "name_ns", "glued", "addr", "nums", "nums_j", "postal", "indic")
# Only hyphens: noise writes 76 as "7-6"; spaces separate genuinely different
# numbers ("Door No 851 406") and must not be joined.
_IN_NUMBER_HYPHEN = re.compile(r"(?<=\d)-(?=\d)")


def canonical_numbers(text: str) -> List[str]:
    """Digit groups with leading zeros stripped, first-occurrence order, no
    repeats.  "09585 Duffney, #09585" -> ["9585"].  Order is kept because the
    first number of an address is usually the street number."""
    seen, out = set(), []
    for n in _DIGITS.findall(text):
        n = n.lstrip("0") or "0"
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def prepare_one(name: str, address: str, country: Optional[str] = None
                ) -> Tuple[str, str, str, str, str, str, str, int]:
    """The text views a pair comparison needs, computed once per record.
    country selects a country-specific abbreviation table, if one exists."""
    nv, av = name_views(name), address_views(address, country)
    return (
        nv["folded"],                                   # cleaned, transliterated name
        nv["nosuffix"],                                 # ... without Ltd/Pvt/LLC
        glued_name(nv["folded"]),                       # "vdrcornerstonepegasus"
        av["expanded"],                                 # cleaned address, Rd -> road
        " ".join(canonical_numbers(av["folded"])),      # folded view: Indic digits are ASCII
        # numbers with in-number hyphens removed: noise writes 76 as "7-6"
        # and 201 as "2-01"; comparing both versions catches it
        # ("translit" keeps punctuation and has Indic digits already in ASCII)
        " ".join(canonical_numbers(_IN_NUMBER_HYPHEN.sub("", av["translit"]))),
        # postal codes judged by their RAW width, before leading zeros are
        # stripped ("01234" is a 5-digit code, not the number 1234)
        " ".join(sorted({g for g in _DIGITS.findall(av["folded"]) if len(g) in POSTAL_LENGTHS})),
        int(has_indic(name) or has_indic(address)),     # was transliteration needed?
    )


def _prepare_frame(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(PREP_COLUMNS))
    for c in PREP_COLUMNS[:-1]:
        df[c] = df[c].astype("str")          # Arrow-backed: a few flat buffers
    df["indic"] = df["indic"].astype(np.int8)
    return df


def prepare_min_one(name: str, address: str, country: Optional[str] = None) -> Tuple[str, str]:
    """Only the two views the selection formula needs (same values as the
    "name" and "addr" columns of prepare_one), at a fraction of the cost."""
    return name_views(name)["folded"], address_views(address, country)["expanded"]


def _prepare_min_task(args):
    names, addrs, countries = args
    rows = [prepare_min_one(n, a, c) for n, a, c in zip(names, addrs, countries)]
    df = pd.DataFrame(rows, columns=["name", "addr"])
    return df.astype("str")


def prepare_min(names: Sequence[str], addrs: Sequence[str], countries: Sequence[str],
                workers: int = 0, chunk: int = 20_000) -> pd.DataFrame:
    tasks = [(list(names[s:s + chunk]), list(addrs[s:s + chunk]), list(countries[s:s + chunk]))
             for s in range(0, len(names), chunk)]
    if workers and workers > 1 and len(tasks) > 1:
        with get_context("fork").Pool(workers, initializer=_die_with_parent) as pool:
            parts = pool.map(_prepare_min_task, tasks)
    else:
        parts = [_prepare_min_task(t) for t in tasks]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        {"name": pd.Series([], dtype="str"), "addr": pd.Series([], dtype="str")})


def _prepare_task(args):
    names, addrs, countries = args
    # Built inside the worker, so the parent never holds millions of tuples.
    return _prepare_frame([prepare_one(n, a, c) for n, a, c in zip(names, addrs, countries)])


def prepare(names: Sequence[str], addrs: Sequence[str], workers: int = 0,
            chunk: int = 20_000, countries: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """prepare_one over many records, in parallel; rows keep their order.

    The result uses pandas' Arrow-backed string columns: millions of strings
    in a few flat buffers instead of millions of Python objects.
    """
    if countries is None:
        countries = [None] * len(names)
    if len(countries) != len(names):
        raise ValueError("countries must align with names")
    tasks = [(list(names[s:s + chunk]), list(addrs[s:s + chunk]), list(countries[s:s + chunk]))
             for s in range(0, len(names), chunk)]
    if workers and workers > 1 and len(tasks) > 1:
        with get_context("fork").Pool(workers, initializer=_die_with_parent) as pool:
            parts = pool.map(_prepare_task, tasks)
    else:
        parts = [_prepare_task(t) for t in tasks]
    if not parts:
        return _prepare_frame([])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# the feature list -- order is the model's input contract
# --------------------------------------------------------------------------

FEATURE_NAMES: Tuple[str, ...] = (
    # names
    "name_ratio", "name_jw", "name_tsort", "name_tset", "ns_ratio", "ns_tset",
    "name_equal", "glued_equal", "name_len_a", "name_len_t", "name_len_ratio",
    "name_jacc", "name_wov", "name_wshared_max", "name_wmiss_a", "name_wmiss_t",
    # addresses
    "addr_ratio", "addr_tsort", "addr_tset", "addr_jacc", "addr_wov",
    "addr_wshared_max", "addr_empty_a", "addr_empty_t", "addr_len_ratio",
    # numbers
    "num_n_a", "num_n_t", "num_shared", "num_jacc", "num_first_equal",
    "num_conflict", "num_min_reldiff", "postal_equal", "postal_conflict",
    "numj_jacc", "numj_first_equal", "num_prefix",
    # script
    "indic_a", "indic_t",
    # retrieval
    "ret_score", "ret_rank", "is_s3", "n_channels",
    *(f"ch_{c}" for c in CHANNELS),
    # how many pool records share this exact cleaned name / address (log1p);
    # NaN when not supplied
    "t_name_cnt", "t_addr_cnt", "a_name_cnt", "a_addr_cnt",
    # context: compared with the same S1 record's other candidates
    "n_cands", "ret_score_gap", "name_tset_gap", "addr_tset_gap",
    "name_tset_lead", "addr_tset_lead", "combo_rank",
    # missing-aware context (v1.2): combo_rank counts a missing field as 0, so a
    # true match with an empty address or a garbled name is ranked last among
    # its rivals -- more than half of the true pairs v1 rejected ranked 10th or
    # worse.  These rank on the fields that are PRESENT and skip missing ones.
    "avail_mean", "avail_rank", "avail_gap", "name_rank_avail", "addr_rank_avail",
    "n_strong_rivals",
)
F_INDEX = {f: i for i, f in enumerate(FEATURE_NAMES)}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _masked_sim(a: List[str], t: List[str], scorer, scale: float) -> np.ndarray:
    """Element-wise similarity in [0,1]; NaN where either side is empty.
    rapidfuzz scores "" vs "" as a perfect 100 -- exactly what must not reach
    the model."""
    s = process.cpdist(a, t, scorer=scorer, workers=1).astype(np.float32) / scale
    empty = np.fromiter((not x or not y for x, y in zip(a, t)), bool, len(a))
    s[empty] = np.nan
    return s


def _eligible(tokens: List[str], field: str) -> List[str]:
    """Same token filter as blocking keys, so weights come from the same table."""
    if field == "n":
        return sorted({t for t in tokens if len(t) >= 3})
    return sorted({t for t in tokens if len(t) >= 4 and not t.isdigit()})


def _token_block(a_txt: List[str], t_txt: List[str], countries: List[str], field: str,
                 freq: Optional[TokenFreq]) -> np.ndarray:
    """Jaccard, rarity-weighted overlap, max shared weight, and the share of
    each side's weight left unmatched.  Returns (n, 5) float32; NaN if a side
    has no eligible tokens."""
    n = len(a_txt)
    out = np.full((n, 5), np.nan, dtype=np.float32)
    a_tok = [_eligible(x.split(), field) for x in a_txt]
    t_tok = [_eligible(x.split(), field) for x in t_txt]

    # one vectorised lookup for every token in the block
    known = freq.countries if freq is not None else frozenset()
    strs: List[str] = []
    for c, at, tt in zip(countries, a_tok, t_tok):
        scope = c if c in known else "*"
        strs.extend(f"{field}|{scope}|{x}" for x in at)
        strs.extend(f"{field}|{scope}|{x}" for x in tt)
    if freq is not None:
        w_all = np.log(IDF_REF / freq.lookup(strs).astype(np.float64))
    else:
        w_all = np.full(len(strs), math.log(IDF_REF))
    w_all = w_all.tolist()

    pos = 0
    for i, (at, tt) in enumerate(zip(a_tok, t_tok)):
        wa = dict(zip(at, w_all[pos:pos + len(at)])); pos += len(at)
        wt = dict(zip(tt, w_all[pos:pos + len(tt)])); pos += len(tt)
        if not wa or not wt:
            continue
        shared = wa.keys() & wt.keys()
        union = wa.keys() | wt.keys()
        w_shared = sum(max(wa[x], wt[x]) for x in shared)
        w_union = sum(max(wa.get(x, 0.0), wt.get(x, 0.0)) for x in union)
        sa, st = sum(wa.values()), sum(wt.values())
        out[i, 0] = len(shared) / len(union)
        out[i, 1] = w_shared / w_union if w_union > 0 else 0.0
        out[i, 2] = max((max(wa[x], wt[x]) for x in shared), default=0.0) / math.log(IDF_REF)
        out[i, 3] = sum(w for x, w in wa.items() if x not in wt) / sa if sa > 0 else 0.0
        out[i, 4] = sum(w for x, w in wt.items() if x not in wa) / st if st > 0 else 0.0
    return out


def _number_block(a_nums: List[str], t_nums: List[str],
                  a_postal: List[str], t_postal: List[str]) -> np.ndarray:
    """(n, 10): counts, shared, Jaccard, first-number equal, conflict, closest
    relative gap, postal equal, postal conflict, prefix.  NaN = not applicable.

    prefix = some number of one side starts with a number of the other and
    is one digit longer ("170" vs "1703"): noise drops trailing digits."""
    out = np.full((len(a_nums), 10), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(a_nums, t_nums)):
        a, t = x.split(), y.split()
        out[i, 0], out[i, 1] = len(a), len(t)
        if not a or not t:
            continue
        sa, st = set(a), set(t)
        sh = sa & st
        out[i, 2] = len(sh)
        out[i, 3] = len(sh) / len(sa | st)
        out[i, 4] = float(a[0] == t[0])
        out[i, 5] = float(not sh)
        # closest pair of numbers, relative gap: 2046 vs 2048 -> ~0.001
        best = 1.0
        for p in a[:4]:
            ip = int(p)
            for q in t[:4]:
                iq = int(q)
                m = max(ip, iq)
                best = min(best, abs(ip - iq) / m if m else 0.0)
        out[i, 6] = best
        out[i, 9] = float(any(len(p) >= 2 and len(q) >= 2 and abs(len(p) - len(q)) == 1
                              and (p.startswith(q) or q.startswith(p))
                              for p in a[:4] for q in t[:4]))
    for i, (x, y) in enumerate(zip(a_postal, t_postal)):
        pa, pt = set(x.split()), set(y.split())
        if pa and pt:
            out[i, 7] = float(bool(pa & pt))
            out[i, 8] = float(not (pa & pt))
    return out


def _joined_number_block(a_nums: List[str], t_nums: List[str]) -> np.ndarray:
    """(n, 2): Jaccard and first-number equality on hyphen-joined numbers."""
    out = np.full((len(a_nums), 2), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(zip(a_nums, t_nums)):
        a, t = x.split(), y.split()
        if a and t:
            sa, st = set(a), set(t)
            out[i, 0] = len(sa & st) / len(sa | st)
            out[i, 1] = float(a[0] == t[0])
    return out


def _group_context(group: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each row: (group max - v, v - best OTHER value in the group).

    gap  = 0 for the best candidate, larger for weaker ones.
    lead > 0 only for a candidate that beats every rival, and by how much;
    a lone candidate gets lead = v.  NaN values count as 0 here."""
    v0 = np.nan_to_num(v, nan=0.0).astype(np.float64)
    n = len(v0)
    order = np.lexsort((-v0, group))
    g_sorted = group[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    sizes = np.diff(np.r_[starts, n])
    best = np.repeat(v0[order[starts]], sizes)
    second_pos = starts + 1
    has_second = sizes > 1
    second = np.zeros(len(starts))
    second[has_second] = v0[order[second_pos[has_second]]]
    second = np.repeat(second, sizes)
    is_first = np.zeros(n, bool); is_first[starts] = True
    other = np.where(is_first, second, best)          # best rival of each row
    gap = np.empty(n); lead = np.empty(n)
    gap[order] = best - v0[order]
    lead[order] = v0[order] - other
    return gap.astype(np.float32), lead.astype(np.float32)


def _group_rank_skipnan(group: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Competition rank (0 = best, ties share the best rank) within each group,
    counting only rivals whose value is present.  NaN where v is NaN."""
    out = np.full(len(v), np.nan, np.float32)
    ok = ~np.isnan(v)
    if ok.any():
        r = pd.Series(v[ok]).groupby(group[ok]).rank(method="min", ascending=False).to_numpy()
        out[ok] = r - 1
    return out


def _group_best_skipnan(group: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Best present value in each row's group (NaN if the group has none)."""
    s = pd.Series(np.where(np.isnan(v), -np.inf, v))
    best = s.groupby(group).transform("max").to_numpy()
    return np.where(np.isinf(best), np.nan, best).astype(np.float32)


# --------------------------------------------------------------------------
# one block of pairs
# --------------------------------------------------------------------------

_SHARED: dict = {}   # frequency table shared with forked workers (read-only numpy)


def pair_block(a: Dict[str, list], t: Dict[str, list], countries: List[str],
               meta: Dict[str, np.ndarray], freq: Optional[TokenFreq] = None) -> np.ndarray:
    """Features for a block of pairs.  a / t map PREP_COLUMNS to per-pair
    lists (anchor side, candidate side); meta holds the retrieval arrays
    'anchor', 'score', 'rank', 'channels', 'source' aligned with the pairs.
    Every candidate of an anchor must be in the same block.
    Returns float32 (n_pairs, len(FEATURE_NAMES))."""
    n = len(countries)
    X = np.full((n, len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    if n == 0:
        return X
    col = lambda f: F_INDEX[f]

    # names
    X[:, col("name_ratio")] = _masked_sim(a["name"], t["name"], fuzz.ratio, 100)
    X[:, col("name_jw")] = _masked_sim(a["name"], t["name"], JaroWinkler.normalized_similarity, 1)
    X[:, col("name_tsort")] = _masked_sim(a["name"], t["name"], fuzz.token_sort_ratio, 100)
    X[:, col("name_tset")] = _masked_sim(a["name"], t["name"], fuzz.token_set_ratio, 100)
    X[:, col("ns_ratio")] = _masked_sim(a["name_ns"], t["name_ns"], fuzz.ratio, 100)
    X[:, col("ns_tset")] = _masked_sim(a["name_ns"], t["name_ns"], fuzz.token_set_ratio, 100)
    la = np.fromiter((len(x) for x in a["name"]), np.float32, n)
    lt = np.fromiter((len(x) for x in t["name"]), np.float32, n)
    X[:, col("name_len_a")], X[:, col("name_len_t")] = la, lt
    with np.errstate(invalid="ignore", divide="ignore"):
        X[:, col("name_len_ratio")] = np.where(np.maximum(la, lt) > 0,
                                               np.minimum(la, lt) / np.maximum(la, lt), np.nan)
    X[:, col("name_equal")] = np.fromiter((float(x == y) if x and y else np.nan
                                           for x, y in zip(a["name"], t["name"])), np.float32, n)
    X[:, col("glued_equal")] = np.fromiter((float(x == y) if x and y else np.nan
                                            for x, y in zip(a["glued"], t["glued"])), np.float32, n)
    nt = _token_block(a["name"], t["name"], countries, "n", freq)
    for j, f in enumerate(("name_jacc", "name_wov", "name_wshared_max", "name_wmiss_a", "name_wmiss_t")):
        X[:, col(f)] = nt[:, j]

    # addresses
    X[:, col("addr_ratio")] = _masked_sim(a["addr"], t["addr"], fuzz.ratio, 100)
    X[:, col("addr_tsort")] = _masked_sim(a["addr"], t["addr"], fuzz.token_sort_ratio, 100)
    X[:, col("addr_tset")] = _masked_sim(a["addr"], t["addr"], fuzz.token_set_ratio, 100)
    at = _token_block(a["addr"], t["addr"], countries, "a", freq)
    for j, f in enumerate(("addr_jacc", "addr_wov", "addr_wshared_max")):
        X[:, col(f)] = at[:, j]
    aa = np.fromiter((len(x) for x in a["addr"]), np.float32, n)
    ta = np.fromiter((len(x) for x in t["addr"]), np.float32, n)
    X[:, col("addr_empty_a")], X[:, col("addr_empty_t")] = aa == 0, ta == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        X[:, col("addr_len_ratio")] = np.where(np.minimum(aa, ta) > 0,
                                               np.minimum(aa, ta) / np.maximum(aa, ta), np.nan)

    # numbers
    nb = _number_block(a["nums"], t["nums"], a["postal"], t["postal"])
    for j, f in enumerate(("num_n_a", "num_n_t", "num_shared", "num_jacc", "num_first_equal",
                           "num_conflict", "num_min_reldiff", "postal_equal", "postal_conflict",
                           "num_prefix")):
        X[:, col(f)] = nb[:, j]
    jb = _joined_number_block(a["nums_j"], t["nums_j"])
    X[:, col("numj_jacc")], X[:, col("numj_first_equal")] = jb[:, 0], jb[:, 1]

    # script
    X[:, col("indic_a")] = np.asarray(a["indic"], np.float32)
    X[:, col("indic_t")] = np.asarray(t["indic"], np.float32)

    # retrieval
    ch = meta["channels"].astype(np.int64)
    X[:, col("ret_score")] = meta["score"]
    X[:, col("ret_rank")] = meta["rank"]
    X[:, col("is_s3")] = meta["source"] == 3
    bits = np.stack([(ch >> i) & 1 for i in range(len(CHANNELS))], axis=1)
    X[:, col("n_channels")] = bits.sum(axis=1)
    for i, c in enumerate(CHANNELS):
        X[:, col(f"ch_{c}")] = bits[:, i]
    for f in ("t_name_cnt", "t_addr_cnt", "a_name_cnt", "a_addr_cnt"):
        if f in meta:
            X[:, col(f)] = np.log1p(np.asarray(meta[f], np.float32))

    # context: rivals of the same S1 record (both sources together)
    g = meta["anchor"].astype(np.int64)
    inv = np.unique(g, return_inverse=True)[1]
    X[:, col("n_cands")] = np.bincount(inv)[inv]
    X[:, col("ret_score_gap")], _ = _group_context(g, meta["score"])
    X[:, col("name_tset_gap")], X[:, col("name_tset_lead")] = _group_context(g, X[:, col("name_tset")])
    X[:, col("addr_tset_gap")], X[:, col("addr_tset_lead")] = _group_context(g, X[:, col("addr_tset")])
    combo = np.nan_to_num(X[:, col("name_tset")]) + np.nan_to_num(X[:, col("addr_tset")])
    order = np.lexsort((-combo, g))
    starts = np.r_[0, np.flatnonzero(g[order][1:] != g[order][:-1]) + 1]
    rank_sorted = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    combo_rank = np.empty(n, np.float32); combo_rank[order] = rank_sorted
    X[:, col("combo_rank")] = combo_rank

    nm, ad = X[:, col("name_tset")], X[:, col("addr_tset")]
    with np.errstate(invalid="ignore"):
        avail = np.nanmean(np.stack([nm, ad]), axis=0)          # NaN only if both missing
    X[:, col("avail_mean")] = avail
    X[:, col("avail_rank")] = _group_rank_skipnan(g, avail)
    X[:, col("avail_gap")] = _group_best_skipnan(g, avail) - avail
    X[:, col("name_rank_avail")] = _group_rank_skipnan(g, nm)
    X[:, col("addr_rank_avail")] = _group_rank_skipnan(g, ad)
    strong = np.nan_to_num(avail) >= 0.9
    n_strong = np.bincount(inv, weights=strong)[inv] - strong     # rivals only, not itself
    X[:, col("n_strong_rivals")] = np.log1p(n_strong)
    return X


def _pair_task(args):
    a, t, countries, meta = args
    return pair_block(a, t, countries, meta, _SHARED.get("freq"))


def compute_features(
    A: pd.DataFrame, T: pd.DataFrame, a_pos: np.ndarray, t_pos: np.ndarray,
    countries: Sequence[str], meta: Dict[str, np.ndarray], freq: Optional[TokenFreq] = None,
    workers: int = 0, block_pairs: int = 50_000, wave: int = 0,
) -> np.ndarray:
    """Features for many pairs, in anchor-aligned blocks, in parallel.

    A / T: prepared tables (from prepare) for anchors and candidate targets.
    a_pos / t_pos: row of each pair in A and T.  countries: anchor country per
    pair.  meta: retrieval arrays per pair (see pair_block); pairs must be
    grouped by meta['anchor'].

    Blocks are sent to the pool in waves of `wave` blocks (default 2x workers)
    so only a bounded amount of text is in flight at once.
    """
    n = len(a_pos)
    anchor = meta["anchor"]
    if n and np.any(anchor[1:] < anchor[:-1]):
        raise ValueError("pairs must be grouped (sorted) by anchor")
    # block boundaries on anchor changes
    bounds, s = [], 0
    while s < n:
        e = min(s + block_pairs, n)
        while e < n and anchor[e] == anchor[e - 1]:
            e += 1
        bounds.append((s, e)); s = e

    def task(b):
        s, e = b
        a = {c: A[c].iloc[a_pos[s:e]].tolist() for c in PREP_COLUMNS}
        t = {c: T[c].iloc[t_pos[s:e]].tolist() for c in PREP_COLUMNS}
        return a, t, list(countries[s:e]), {k: v[s:e] for k, v in meta.items()}

    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    _SHARED["freq"] = freq
    try:
        if workers and workers > 1 and len(bounds) > 1:
            wave = wave or 2 * workers
            with get_context("fork").Pool(workers, initializer=_die_with_parent) as pool:
                for w0 in range(0, len(bounds), wave):
                    bs = bounds[w0:w0 + wave]
                    for (s, e), part in zip(bs, pool.map(_pair_task, [task(b) for b in bs])):
                        X[s:e] = part
        else:
            for b in bounds:
                X[b[0]:b[1]] = _pair_task(task(b))
    finally:
        _SHARED.clear()
    return X


# --------------------------------------------------------------------------
# cheap features: a fast first-stage ranking over WIDE candidate lists
# --------------------------------------------------------------------------
#
# Blocking keeps the top 25 candidates per source by key-rarity score, and
# 4.4% of true matches are found but ranked lower and thrown away.  Widening
# to ~150 per source recovers most of them, but the full feature set is too
# slow for 6x the pairs.  These features use only rapidfuzz's bulk C routines
# and numpy, so a small ranking model can re-select the best candidates from
# the wide list before the full features run.

CHEAP_NAMES: Tuple[str, ...] = (
    "ret_score", "ret_rank", "is_s3", "n_channels", *(f"ch_{c}" for c in CHANNELS),
    "c_name_ratio", "c_name_jw", "c_name_tset", "c_ns_tset",
    "c_addr_ratio", "c_addr_tset", "c_glued_eq",
    "c_name_best", "c_addr_best", "c_name_gap", "c_addr_gap", "c_combo_gap",
    "t_name_cnt", "t_addr_cnt", "a_name_cnt", "a_addr_cnt",
)
C_INDEX = {f: i for i, f in enumerate(CHEAP_NAMES)}


def _bulk_sim(a: List[str], t: List[str], scorer, scale: float, threads: int) -> np.ndarray:
    s = process.cpdist(a, t, scorer=scorer, workers=threads).astype(np.float32) / scale
    empty = np.fromiter((not x or not y for x, y in zip(a, t)), bool, len(a))
    s[empty] = np.nan
    return s


def cheap_features(A: pd.DataFrame, T: pd.DataFrame, a_pos: np.ndarray, t_pos: np.ndarray,
                   meta: Dict[str, np.ndarray], chunk: int = 2_000_000,
                   threads: int = 6) -> np.ndarray:
    """(n_pairs, len(CHEAP_NAMES)) float32.  Pairs must be grouped by
    meta['anchor'] and include ALL of each anchor's wide candidates (the
    best/gap features compare every candidate with its rivals)."""
    n = len(a_pos)
    X = np.full((n, len(CHEAP_NAMES)), np.nan, dtype=np.float32)
    if n == 0:
        return X
    anchor = np.asarray(meta["anchor"])
    if np.any(anchor[1:] < anchor[:-1]):
        raise ValueError("pairs must be grouped (sorted) by anchor")
    col = lambda f: C_INDEX[f]
    X[:, col("ret_score")] = meta["score"]
    X[:, col("ret_rank")] = meta["rank"]
    X[:, col("is_s3")] = np.asarray(meta["source"]) == 3
    ch = np.asarray(meta["channels"]).astype(np.int64)
    nchan = np.zeros(n, np.float32)
    for i, c in enumerate(CHANNELS):
        b = (ch >> i) & 1
        X[:, col(f"ch_{c}")] = b
        nchan += b
    X[:, col("n_channels")] = nchan
    for f in ("t_name_cnt", "t_addr_cnt", "a_name_cnt", "a_addr_cnt"):
        if f in meta:
            X[:, col(f)] = np.log1p(np.asarray(meta[f], np.float32))

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        ap, tp = a_pos[s:e], t_pos[s:e]
        an, tn = A["name"].iloc[ap].tolist(), T["name"].iloc[tp].tolist()
        X[s:e, col("c_name_ratio")] = _bulk_sim(an, tn, fuzz.ratio, 100, threads)
        X[s:e, col("c_name_jw")] = _bulk_sim(an, tn, JaroWinkler.normalized_similarity, 1, threads)
        X[s:e, col("c_name_tset")] = _bulk_sim(an, tn, fuzz.token_set_ratio, 100, threads)
        del an, tn
        X[s:e, col("c_ns_tset")] = _bulk_sim(A["name_ns"].iloc[ap].tolist(), T["name_ns"].iloc[tp].tolist(),
                                             fuzz.token_set_ratio, 100, threads)
        aa, ta = A["addr"].iloc[ap].tolist(), T["addr"].iloc[tp].tolist()
        X[s:e, col("c_addr_ratio")] = _bulk_sim(aa, ta, fuzz.ratio, 100, threads)
        X[s:e, col("c_addr_tset")] = _bulk_sim(aa, ta, fuzz.token_set_ratio, 100, threads)
        del aa, ta
        ag = A["glued"].iloc[ap].to_numpy(dtype=object)
        tg = T["glued"].iloc[tp].to_numpy(dtype=object)
        eq = (ag == tg).astype(np.float32)
        eq[(ag == "") | (tg == "")] = np.nan
        X[s:e, col("c_glued_eq")] = eq

    # rivals: best similarity among this anchor's wide candidates, and the gap to it
    starts = np.flatnonzero(np.r_[True, anchor[1:] != anchor[:-1]])
    sizes = np.diff(np.r_[starts, n])
    for f, best, gap in (("c_name_tset", "c_name_best", "c_name_gap"),
                         ("c_addr_tset", "c_addr_best", "c_addr_gap")):
        v = np.nan_to_num(X[:, col(f)], nan=0.0)
        b = np.repeat(np.maximum.reduceat(v, starts), sizes)
        X[:, col(best)], X[:, col(gap)] = b, b - v
    combo = np.nan_to_num(X[:, col("c_name_tset")]) + np.nan_to_num(X[:, col("c_addr_tset")])
    X[:, col("c_combo_gap")] = np.repeat(np.maximum.reduceat(combo, starts), sizes) - combo
    return X
