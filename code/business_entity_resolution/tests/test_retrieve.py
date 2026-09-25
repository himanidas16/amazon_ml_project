"""Tests for candidate generation.

The fixtures are tiny invented records written for software testing only --
they are not competition data and are never used for training.
"""

import numpy as np
import pytest

from business_er.retrieve import (
    CHANNEL_BIT,
    build_index,
    candidate_id_lists,
    compute_keys,
    generate_candidates,
    record_keys,
    stable_hash,
)

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"

S2 = [
    ("S2-10", "Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US"),
    ("S2-11", "Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US"),   # identical text, distinct id
    ("S2-12", "Blue Heron Bakery", "77 Pine Street, Denver CO", "US"),
    ("S2-13", "Boulangerie Lemaire SARL", "5 bis Rue Victor Hugo, Lille", "France"),
    ("S2-14", "Nematech Solutions", "2046 Autrey Road, Austin", "India"),      # wrong country
    ("S2-15", "", "", "US"),                                                     # all empty
]
S3 = [
    ("S3-10", "Nematek Solutions", "2048 Autrey Rd, Austin", "US"),            # typo + drifted number
    ("S3-20", "Boulangerie Lemaire", "5 R. Victor Hugo, 59000 Lille", "France"),
]


def _write(path, rows):
    path.write_text(HEADER + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


@pytest.fixture
def index(tmp_path):
    return build_index({2: _write(tmp_path / "s2.tsv", S2), 3: _write(tmp_path / "s3.tsv", S3)})


def _query(index, anchors, **kw):
    names, addrs, ctry = zip(*anchors)
    ak = compute_keys(list(names), list(addrs), list(ctry))
    c = generate_candidates(index, ak, len(anchors), **kw)
    return c, candidate_id_lists(c, index, len(anchors))


def test_stable_hash_is_deterministic():
    a = stable_hash(["US|nematech", "India|x"])
    b = stable_hash(["US|nematech", "India|x"])
    assert a.dtype == np.uint64 and (a == b).all() and a[0] != a[1]


def test_typo_and_number_drift_still_retrieved(index):
    _, ids = _query(index, [("Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US")])
    assert {"S2-10", "S2-11", "S3-10"} <= set(ids[0])


def test_identical_text_distinct_ids_both_kept(index):
    _, ids = _query(index, [("Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US")])
    assert "S2-10" in ids[0] and "S2-11" in ids[0]


def test_country_is_respected_and_open_set(index):
    _, ids = _query(index, [
        ("Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US"),
        ("Boulangerie Lemaire", "5 Rue Victor Hugo Lille", "France"),   # unseen in training
    ])
    assert "S2-14" not in ids[0]                      # India record never offered to a US anchor
    assert {"S2-13", "S3-20"} <= set(ids[1])
    assert all(i not in ids[1] for i in ("S2-10", "S2-11", "S3-10"))


def test_empty_fields_do_not_crash_or_match(index):
    c, ids = _query(index, [("", "", "US"), ("Blue Heron Bakery", "77 Pine Street Denver", "US")])
    assert ids[0] == []                               # blank never "matches" blank
    assert "S2-15" not in ids[1]
    assert "S2-12" in ids[1]


def test_no_candidate_anchor_gets_empty_list(index):
    _, ids = _query(index, [("Zzyzx Quux", "999999 Nowhere", "US")])
    assert ids == [[]]


def test_top_k_is_per_source(index):
    c, _ = _query(index, [("Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US")], k=1)
    src = index.codes[c.target] // 2_000_000_000
    assert (np.bincount(src, minlength=4)[2:] <= 1).all()
    assert (c.rank == 0).all()


def test_channel_mask_and_scores(index):
    c, ids = _query(index, [("Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US")])
    pos = ids[0].index("S2-10")
    assert c.channels[pos] & CHANNEL_BIT["name_exact"]
    assert c.score[pos] > 0
    # exact twin must outscore the typo'd version
    assert c.score[pos] > c.score[ids[0].index("S3-10")]


def test_cap_drops_huge_blocks(index):
    c, _ = _query(index, [("Nematech Solutions LLC", "2046 Autrey Road, Austin TX", "US")], cap=1)
    # with cap=1 only keys unique to one target survive; the twins S2-10/S2-11 share all keys
    tg = set(index.codes[c.target].tolist())
    assert 2 * 2_000_000_000 + 10 not in tg


def test_parallel_equals_serial(index):
    anchors = [("Nematech Solutions LLC", "2046 Autrey Road", "US"),
               ("Boulangerie Lemaire", "5 Rue Victor Hugo", "France")] * 30
    names, addrs, ctry = zip(*anchors)
    ak1 = compute_keys(list(names), list(addrs), list(ctry), workers=1, chunk=7)
    ak2 = compute_keys(list(names), list(addrs), list(ctry), workers=3, chunk=7)
    for ch in ak1:
        assert (ak1[ch][0] == ak2[ch][0]).all() and (ak1[ch][1] == ak2[ch][1]).all()
    c1 = generate_candidates(index, ak1, len(anchors), batch=5, workers=1)
    c2 = generate_candidates(index, ak2, len(anchors), batch=5, workers=3)
    for f in ("anchor", "target", "score", "channels", "rank"):
        assert (getattr(c1, f) == getattr(c2, f)).all()


def test_record_keys_bounded():
    k = record_keys(" ".join(f"word{i}" for i in range(50)),
                    " ".join(f"{i} street{i}" for i in range(50)), "US")
    assert len(k["name_token"]) <= 6 and len(k["name_pair"]) <= 6
    assert len(k["num_token"]) <= 6 and len(k["name_num"]) <= 6


def test_country_sharding_gives_identical_candidates(tmp_path):
    paths = {2: _write(tmp_path / "s2.tsv", S2), 3: _write(tmp_path / "s3.tsv", S3)}
    anchors = [("Nematech Solutions LLC", "2046 Autrey Road", "US"),
               ("Boulangerie Lemaire", "5 Rue Victor Hugo", "France"),
               ("Nematech Solutions", "2046 Autrey Road", "India")]
    full = build_index(paths)
    for k in (1, 25):
        _, want = _query(full, anchors, k=k)
        for i, (_, _, ctry) in enumerate(anchors):
            shard = build_index(paths, country=ctry)
            c, got = _query(shard, [anchors[i]], k=k)
            assert sorted(got[0]) == sorted(want[i])


def test_leading_zeros_do_not_block_number_match():
    a = record_keys("EQ Rate Inc", "2003 Brookfield Road, Columbus", "US")
    b = record_keys("EQ Re Inc", "02003 Brookfield Road, Columbus, Ohio", "US")
    assert set(a["num_token"]) & set(b["num_token"])
    assert record_keys("X", "D-014 Floor", "IN")["num_set"] == record_keys("X", "D-14 Floor", "IN")["num_set"]
    assert record_keys("X", "000 Main", "US")["num_set"] == ["US|#|0"]


# ---- token rarity ---------------------------------------------------------

def _freq_fixture(tmp_path):
    from business_er.retrieve import build_token_freq
    common = [(f"S2-{100 + i}", f"Shop {i}", f"{i} Apartment Floor Centre City Colony", "India")
              for i in range(30)]
    return build_token_freq({2: _write(tmp_path / "f.tsv", common)})


def test_rarest_address_tokens_are_kept(tmp_path):
    freq = _freq_fixture(tmp_path)
    addr = "Ist Floor, Maa Bhagwati Apartment, Near Gokul Apartment, Patel Nagar, City, Centre, Colony"
    plain = record_keys("Gujarat Exports", addr, "India")["addr_token"]
    rare = record_keys("Gujarat Exports", addr, "India", freq)["addr_token"]
    # alphabetical order keeps the common words; rarity order keeps the distinctive ones
    common = {f"India|a|{w}" for w in ("apartment", "floor", "centre", "city", "colony")}
    distinctive = {f"India|a|{w}" for w in ("bhagwati", "gokul", "nagar", "patel", "near")}
    assert len(common & set(plain)) >= 4            # alphabetical: mostly common words
    assert set(rare[:5]) == distinctive              # rarity: every distinctive word first
    assert len(common & set(rare)) <= 1              # only the leftover 6th slot is common


def test_unseen_country_uses_global_counts(tmp_path):
    freq = _freq_fixture(tmp_path)
    assert "France" not in freq.countries
    keys = record_keys("X", "Centre Apartment Floor City Colony Lemaire Gambetta", "France", freq)
    # global counts still know "apartment"/"centre" are common, so French rare words win
    assert "France|a|lemaire" in keys["addr_token"] and "France|a|gambetta" in keys["addr_token"]


def test_freq_parallel_equals_serial(tmp_path):
    freq = _freq_fixture(tmp_path)
    names = ["Blue Heron Bakery", "Gujarat Exports"] * 20
    addrs = ["77 Pine Street Apartment Denver", "Ist Floor Bhagwati Apartment City"] * 20
    ctry = ["US", "India"] * 20
    a = compute_keys(names, addrs, ctry, workers=1, chunk=7, freq=freq)
    b = compute_keys(names, addrs, ctry, workers=3, chunk=7, freq=freq)
    for ch in a:
        assert (a[ch][0] == b[ch][0]).all() and (a[ch][1] == b[ch][1]).all()


def test_token_freq_round_trip(tmp_path):
    from business_er.retrieve import TokenFreq
    freq = _freq_fixture(tmp_path)
    freq.save(tmp_path / "f.npz")
    back = TokenFreq.load(tmp_path / "f.npz")
    assert (back.hashes == freq.hashes).all() and back.countries == freq.countries
    assert back.lookup(["a|India|apartment", "a|India|neverseen"]).tolist() == [30, 1]


# ---- glued-name channel ---------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("VDR Cornerstone Pegasus LLC", "vdrcornerstonepegasus.com"),
    ("Richmond Academy", "richmondacademy.com"),
    ("Aditya Enterprises Limited", "#adityaenterprises"),
    ("S.R.K Traders", "SRK Traders Pvt Ltd"),
    ("Blue Heron Bakery", "www.blueheronbakery.net"),
])
def test_glued_names_share_a_key(a, b):
    ka = record_keys(a, "", "US")["name_glued"]
    kb = record_keys(b, "", "US")["name_glued"]
    assert ka and ka == kb


def test_glued_key_skips_short_or_generic_names():
    assert record_keys("ABC Ltd", "", "US")["name_glued"] == []        # "abc" too short
    assert record_keys("Private Limited", "", "US")["name_glued"] == []
    assert record_keys("Blue Heron", "", "US")["name_glued"] != record_keys("Blue Herons", "", "US")["name_glued"]
