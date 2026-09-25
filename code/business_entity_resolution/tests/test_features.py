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
