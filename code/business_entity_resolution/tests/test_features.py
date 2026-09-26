"""Tests for pair features.

Fixtures are small invented records for software testing only -- not
competition data, never used for training.
"""

import numpy as np
import pytest

from business_er.features import (
    FEATURE_NAMES,
    F_INDEX,
    canonical_numbers,
    compute_features,
    prepare,
)

# (anchor name, anchor address, candidate name, candidate address)
PAIRS = [
    ("Nematech Solutions LLC", "2046 Autrey Road, Austin TX 73301",
     "Nematek Solutions", "2048 Autrey Rd, Austin 73301"),                     # true-ish, typos
    ("Nematech Solutions LLC", "2046 Autrey Road, Austin TX 73301",
     "Nematech Consulting", "15 Oak Street, Dallas 75001"),                     # rival
    ("Blue Heron Bakery", "", "Blue Heron Bakery", ""),                          # both addresses blank
    ("Shree Best Business Private Limited", "406 Manas Nagar Colony, Lucknow",
     "श्री बेस्ट बिजनेस प्राइवेट लिमिटेड", "406, MANAS NAGAR COLONY, LUCKNOW"),       # Devanagari
]
COUNTRIES = ["US", "US", "US", "India"]
ANCHOR = np.array([0, 0, 1, 2])


def _features(pairs=PAIRS, anchor=ANCHOR, countries=COUNTRIES, workers=0, block=50_000):
    A = prepare([p[0] for p in pairs], [p[1] for p in pairs])
    T = prepare([p[2] for p in pairs], [p[3] for p in pairs])
    n = len(pairs)
    meta = {"anchor": anchor, "score": np.linspace(10, 1, n).astype(np.float32),
            "rank": np.zeros(n, np.int16), "channels": np.full(n, 3, np.uint16),
            "source": np.array([2, 3] * n)[:n]}
    idx = np.arange(n)
    return compute_features(A, T, idx, idx, countries, meta, workers=workers, block_pairs=block)


def f(X, row, name):
    return X[row, F_INDEX[name]]


def test_feature_names_unique_and_no_country_or_id():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    assert not any("country" in x or "id" == x or x.endswith("_id") for x in FEATURE_NAMES)


def test_shape_and_dtype():
    X = _features()
    assert X.shape == (len(PAIRS), len(FEATURE_NAMES)) and X.dtype == np.float32


def test_typo_pair_scores_higher_than_rival():
    X = _features()
    assert f(X, 0, "name_tset") > f(X, 1, "name_tset")
    assert f(X, 0, "addr_tset") > f(X, 1, "addr_tset")
    assert f(X, 0, "name_jw") > 0.85


def test_blank_addresses_are_unknown_not_identical():
    X = _features()
    for name in ("addr_ratio", "addr_tset", "addr_tsort", "addr_jacc", "addr_wov", "addr_len_ratio"):
        assert np.isnan(f(X, 2, name)), name
    assert f(X, 2, "addr_empty_a") == 1 and f(X, 2, "addr_empty_t") == 1
    assert f(X, 2, "name_equal") == 1                       # the name, though, is real evidence


def test_transliterated_name_is_comparable():
    X = _features()
    assert f(X, 3, "indic_t") == 1 and f(X, 3, "indic_a") == 0
    assert f(X, 3, "name_tset") > 0.6                     # would be ~0.1 without transliteration
    assert f(X, 3, "num_first_equal") == 1


def test_number_features():
    X = _features()
    assert f(X, 0, "num_first_equal") == 0                # 2046 vs 2048
    assert f(X, 0, "num_min_reldiff") < 0.01              # ...but very close
    assert f(X, 0, "postal_equal") == 1 and f(X, 0, "postal_conflict") == 0
    assert f(X, 1, "postal_conflict") == 1                # 73301 vs 75001
    assert np.isnan(f(X, 2, "num_conflict"))              # no numbers: not applicable


def test_canonical_numbers():
    assert canonical_numbers("09585 Duffney, #09585, 000") == ["9585", "0"]
    assert canonical_numbers("") == []


def test_context_features():
    X = _features()
    # anchor 0 has two candidates: row 0 is the better name match
    assert f(X, 0, "name_tset_gap") == 0 and f(X, 0, "name_tset_lead") > 0
    assert f(X, 1, "name_tset_gap") > 0 and f(X, 1, "name_tset_lead") < 0
    assert f(X, 0, "combo_rank") == 0 and f(X, 1, "combo_rank") == 1
    assert f(X, 0, "n_cands") == 2 and f(X, 2, "n_cands") == 1
    assert f(X, 2, "name_tset_lead") == pytest.approx(f(X, 2, "name_tset"))   # lone candidate


def test_rarity_weighting(tmp_path):
    from business_er.retrieve import build_token_freq
    rows = "".join(f"S2-{i}\tAlpha Solutions {i}\t{i} Main\tUS\n" for i in range(1, 40))
    p = tmp_path / "f.tsv"
    p.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + rows, encoding="utf-8")
    freq = build_token_freq({2: str(p)})
    pairs = [("Zyqor Solutions", "", "Zyqor Consulting", ""),       # share the RARE word
             ("Zyqor Solutions", "", "Plumbo Solutions", "")]       # share the COMMON word
    A = prepare([x[0] for x in pairs], [x[1] for x in pairs])
    T = prepare([x[2] for x in pairs], [x[3] for x in pairs])
    meta = {"anchor": np.array([0, 1]), "score": np.ones(2, np.float32), "rank": np.zeros(2, np.int16),
            "channels": np.ones(2, np.uint16), "source": np.array([2, 2])}
    X = compute_features(A, T, np.arange(2), np.arange(2), ["US", "US"], meta, freq=freq)
    assert f(X, 0, "name_jacc") == f(X, 1, "name_jacc")        # same plain overlap...
    assert f(X, 0, "name_wov") > f(X, 1, "name_wov")          # ...but rarity tells them apart


def test_parallel_equals_serial():
    pairs, anchor = PAIRS * 25, np.repeat(np.arange(50), 2)
    countries = COUNTRIES * 25
    a = _features(pairs, anchor, countries, workers=1, block=7)
    b = _features(pairs, anchor, countries, workers=3, block=7)
    assert np.array_equal(a, b, equal_nan=True)


def test_blocks_never_split_an_anchor():
    # block_pairs=1 must still keep both candidates of anchor 0 together,
    # or the context features would silently change
    whole = _features(block=50_000)
    tiny = _features(block=1)
    assert np.array_equal(whole, tiny, equal_nan=True)


def test_unsorted_anchors_rejected():
    with pytest.raises(ValueError):
        _features(anchor=np.array([1, 0, 0, 2]))


def test_cheap_features_agree_with_full_ones_and_blank_is_unknown():
    from business_er.features import C_INDEX, CHEAP_NAMES, cheap_features
    A = prepare([p[0] for p in PAIRS], [p[1] for p in PAIRS])
    T = prepare([p[2] for p in PAIRS], [p[3] for p in PAIRS])
    n = len(PAIRS)
    meta = {"anchor": ANCHOR, "score": np.ones(n, np.float32), "rank": np.arange(n, dtype=np.int16),
            "channels": np.full(n, 5, np.uint16), "source": np.array([2, 3, 2, 3])}
    idx = np.arange(n)
    C = cheap_features(A, T, idx, idx, meta, chunk=3, threads=1)
    Xf = _features()
    assert C.shape == (n, len(CHEAP_NAMES))
    # same measure, same text -> same value as the full feature set
    assert np.allclose(C[:, C_INDEX["c_name_tset"]], Xf[:, F_INDEX["name_tset"]], equal_nan=True)
    assert np.allclose(C[:, C_INDEX["c_addr_tset"]], Xf[:, F_INDEX["addr_tset"]], equal_nan=True)
    assert np.isnan(C[2, C_INDEX["c_addr_tset"]])                 # blank addresses stay unknown
    assert C[0, C_INDEX["c_name_gap"]] == 0 and C[1, C_INDEX["c_name_gap"]] > 0   # rival context
    assert C[0, C_INDEX["n_channels"]] == 2                        # bits 0 and 2 of 5


# ---- v1.2 number and count features ---------------------------------------

def _one_pair(a_name, a_addr, t_name, t_addr, extra_meta=None):
    A = prepare([a_name], [a_addr]); T = prepare([t_name], [t_addr])
    meta = {"anchor": np.array([0]), "score": np.ones(1, np.float32), "rank": np.zeros(1, np.int16),
            "channels": np.ones(1, np.uint16), "source": np.array([2])}
    meta.update(extra_meta or {})
    return compute_features(A, T, np.arange(1), np.arange(1), ["India"], meta)


def test_hyphen_split_numbers_are_rejoined():
    X = _one_pair("Klassic & Co", "Plot No 76 Honga Industrial Estate",
                  "Klassic Co", "Plot No 7-6 Honga Industrial Estate")
    assert f(X, 0, "num_first_equal") == 0          # plain view: 76 vs 7
    assert f(X, 0, "numj_first_equal") == 1         # joined view: 76 vs 76
    X = _one_pair("A", "Door No 851 406, Lucknow", "A", "Door No 851406, Lucknow")
    assert f(X, 0, "numj_first_equal") == 0         # spaces are never joined


def test_truncated_number_prefix():
    X = _one_pair("Gold Om Systems", "1703, 17Th Floor, 73, Bansilal Bhavan",
                  "Gold Om Systems", "170, 17Th Floor, 73, Bansilal Bhavan")
    assert f(X, 0, "num_prefix") == 1
    X = _one_pair("A", "12 Main Road", "A", "45 Main Road")
    assert f(X, 0, "num_prefix") == 0


def test_postal_uses_raw_width():
    X = _one_pair("A", "5 Elm St, Hartford CT 01234", "A", "5 Elm Street, CT 01234")
    assert f(X, 0, "postal_equal") == 1             # 01234 is still a postal code


def test_pool_counts_are_log1p_or_unknown():
    X = _one_pair("A", "x", "A", "x")
    assert np.isnan(f(X, 0, "t_addr_cnt"))          # not supplied -> unknown
    X = _one_pair("A", "x", "A", "x", {"t_name_cnt": np.array([0.0]), "t_addr_cnt": np.array([3.0]),
                                       "a_name_cnt": np.array([1.0]), "a_addr_cnt": np.array([1.0])})
    assert f(X, 0, "t_addr_cnt") == pytest.approx(np.log1p(3)) and f(X, 0, "t_name_cnt") == 0


def test_missing_aware_context_does_not_punish_missing_fields():
    # anchor 0: candidate A has a perfect name but NO address (a true-match
    # pattern); candidate B has a similar name and a different address
    pairs = [("Ekana & Associates", "Block 6B, Sane Guruji Premises, Mumbai", "Ekana & Associates Co", ""),
             ("Ekana & Associates", "Block 6B, Sane Guruji Premises, Mumbai",
              "Ekana & Associates Pvt", "Block 9, Sane Guruji Premises, Pune")]
    A = prepare([p[0] for p in pairs], [p[1] for p in pairs])
    T = prepare([p[2] for p in pairs], [p[3] for p in pairs])
    meta = {"anchor": np.array([0, 0]), "score": np.ones(2, np.float32), "rank": np.zeros(2, np.int16),
            "channels": np.ones(2, np.uint16), "source": np.array([2, 3])}
    X = compute_features(A, T, np.arange(2), np.arange(2), ["India", "India"], meta)
    assert f(X, 0, "combo_rank") == 1                    # the old rank punishes the missing address
    assert f(X, 0, "avail_rank") == 0                    # the new one does not
    assert np.isnan(f(X, 0, "addr_rank_avail")) and f(X, 1, "addr_rank_avail") == 0
    assert f(X, 0, "name_rank_avail") == 0
    assert f(X, 0, "avail_gap") == 0


def test_rank_ties_share_the_best_rank():
    from business_er.features import _group_rank_skipnan
    r = _group_rank_skipnan(np.array([0, 0, 0, 1]), np.array([0.9, 0.9, 0.5, np.nan], np.float32))
    assert r[:3].tolist() == [0, 0, 2] and np.isnan(r[3])


def test_prepare_min_matches_full_prepare():
    from business_er.features import prepare_min
    names = [p[0] for p in PAIRS] + [p[2] for p in PAIRS]
    addrs = [p[1] for p in PAIRS] + [p[3] for p in PAIRS]
    ctry = ["US", "India"] * 4
    full = prepare(names, addrs, countries=ctry)
    slim = prepare_min(names, addrs, ctry)
    assert full["name"].tolist() == slim["name"].tolist()
    assert full["addr"].tolist() == slim["addr"].tolist()
