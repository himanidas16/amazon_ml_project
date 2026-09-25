"""Tests for reading the challenge TSVs and writing the submission files.

The synthetic rows below are software fixtures, deliberately tiny and invented.
They are NOT extra training data and are never used for modelling.
"""

import csv

import pytest

from business_er.io import (
    Dataset,
    check_truth_consistency,
    file_sha256,
    read_ground_truth,
    read_id_list_tsv,
    read_source,
    write_id_list_tsv,
)


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def test_reads_tabs_and_keeps_commas_inside_addresses(tmp_path):
    """Addresses contain commas, which is exactly why the files are tab-separated."""
    p = write(tmp_path / "s1.tsv", (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tABC Pvt Ltd\t12 MG Road, Bengaluru, KA\tIndia\n"
    ))
    df = read_source(p, "S1-")
    assert df.loc[0, "business_address"] == "12 MG Road, Bengaluru, KA"
    assert df.loc[0, "country"] == "India"


def test_does_not_invent_missing_values(tmp_path):
    """"NA" and "null" are plain text in an address; an empty cell must become
    "" and never the string "nan"."""
    p = write(tmp_path / "s2.tsv", (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tNA Foods\tnull Street\tUS\n"
        "S2-2\tNo Address Co\t\t\n"
    ))
    df = read_source(p, "S2-")
    assert df.loc[0, "business_name"] == "NA Foods"
    assert df.loc[0, "business_address"] == "null Street"
    assert df.loc[1, "business_address"] == ""
    assert df.loc[1, "country"] == ""
    assert "nan" not in df.to_string()


def test_ids_stay_exact_strings(tmp_path):
    """A numeric-looking id must not lose a leading zero to int conversion."""
    p = write(tmp_path / "s3.tsv", (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-007\tSeven\tRoad\tUS\n"
    ))
    df = read_source(p, "S3-")
    assert df.loc[0, "entity_id"] == "S3-007"


def test_unicode_and_accents_survive(tmp_path):
    """France is in the test set, so accented text must round-trip untouched."""
    p = write(tmp_path / "fr.tsv", (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-9\tBoulangerie Crème Brûlée\t3 Rue de l'Église, Paris\tFrance\n"
    ))
    df = read_source(p, "S2-")
    assert df.loc[0, "business_name"] == "Boulangerie Crème Brûlée"
    assert "Église" in df.loc[0, "business_address"]


def test_rejects_duplicate_and_misprefixed_ids(tmp_path):
    dup = write(tmp_path / "dup.tsv", (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-1\tA\tX\tUS\nS1-1\tB\tY\tUS\n"
    ))
    with pytest.raises(ValueError, match="duplicate"):
        read_source(dup, "S1-")

    wrong = write(tmp_path / "wrong.tsv", (
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-1\tA\tX\tUS\n"
    ))
    with pytest.raises(ValueError, match="do not start with"):
        read_source(wrong, "S1-")


def test_rejects_missing_column(tmp_path):
    p = write(tmp_path / "bad.tsv", "entity_id\tbusiness_name\n" "S1-1\tA\n")
    with pytest.raises(ValueError, match="missing columns"):
        read_source(p, "S1-")


def test_ground_truth_keeps_singletons_as_empty_lists(tmp_path):
    p = write(tmp_path / "gt.tsv", (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-1,S3-2\n"
        "S1-2\t\n"
    ))
    gt = read_ground_truth(p)
    assert gt["S1-1"] == ["S2-1", "S3-2"]
    assert gt["S1-2"] == []          # the singleton row is PRESENT and empty
    assert len(gt) == 2


def test_truth_consistency_reports_problems_without_crashing(tmp_path):
    def src(prefix, n):
        head = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        rows = "".join(f"{prefix}{i}\tName{i}\tAddr{i}\tUS\n" for i in range(1, n + 1))
        return read_source(write(tmp_path / f"{prefix}.tsv", head + rows), prefix)

    ds = Dataset(s1=src("S1-", 2), s2=src("S2-", 2), s3=src("S3-", 1))
    # S1-2 has no row; S2-99 does not exist; S2-1 is claimed by two S1 records.
    ds.truth = {"S1-1": ["S2-1", "S2-99"], "S1-3": ["S2-1"]}
    problems = check_truth_consistency(ds)
    text = " ".join(problems)
    assert "unknown S1 ids" in text
    assert "NO ground-truth row" in text
    assert "do not exist" in text
    assert "more than one S1" in text


# --------------------------------------------------------------------------
# writing -- the submission contract
# --------------------------------------------------------------------------

def test_empty_row_is_a_real_trailing_tab(tmp_path):
    """The single most common way to get a submission rejected: writing "[]",
    "None" or "nan" instead of leaving the field genuinely empty."""
    out = tmp_path / "matching_results.tsv"
    write_id_list_tsv(out, ["S1-1", "S1-2"], {"S1-1": ["S2-5"]}, "matched_entity_ids")
    raw = out.read_text(encoding="utf-8")
    assert raw == (
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-1\tS2-5\n"
        "S1-2\t\n"
    )
    assert "[]" not in raw and "None" not in raw and "nan" not in raw


def test_output_is_sorted_deduplicated_and_reproducible(tmp_path):
    a = tmp_path / "a.tsv"
    b = tmp_path / "b.tsv"
    ids = {"S1-1": ["S3-9", "S2-1", "S2-1", "S2-10"]}
    write_id_list_tsv(a, ["S1-1"], ids, "matched_entity_ids")
    write_id_list_tsv(b, ["S1-1"], {"S1-1": list(reversed(ids["S1-1"]))}, "matched_entity_ids")
    assert a.read_text() == "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S2-10,S3-9\n"
    # Same content in a different input order -> identical bytes.
    assert file_sha256(a) == file_sha256(b)


def test_rows_follow_input_order_and_cover_every_s1(tmp_path):
    out = tmp_path / "o.tsv"
    order = ["S1-3", "S1-1", "S1-2"]
    write_id_list_tsv(out, order, {}, "candidate_entity_ids")
    back = read_id_list_tsv(out)
    assert list(back) == order
    assert all(v == [] for v in back.values())


def test_writer_refuses_invalid_ids(tmp_path):
    out = tmp_path / "o.tsv"
    with pytest.raises(ValueError, match="not an S2-/S3- id"):
        write_id_list_tsv(out, ["S1-1"], {"S1-1": ["S1-2"]}, "matched_entity_ids")
    with pytest.raises(ValueError, match="does not exist"):
        write_id_list_tsv(out, ["S1-1"], {"S1-1": ["S2-404"]}, "matched_entity_ids",
                          allowed_ids={"S2-1"})
    with pytest.raises(ValueError, match="duplicate S1 ids"):
        write_id_list_tsv(out, ["S1-1", "S1-1"], {}, "matched_entity_ids")


def test_round_trip_preserves_everything(tmp_path):
    out = tmp_path / "o.tsv"
    data = {"S1-1": ["S2-1", "S3-2"], "S1-2": [], "S1-3": ["S3-7"]}
    order = ["S1-1", "S1-2", "S1-3"]
    write_id_list_tsv(out, order, data, "matched_entity_ids")
    assert read_id_list_tsv(out) == data


def test_multiple_matches_from_the_same_source_are_kept(tmp_path):
    """A real S1 record here averages 3.7 matches and may have several from one
    source -- the writer must not collapse them."""
    out = tmp_path / "o.tsv"
    ids = ["S2-1", "S2-2", "S2-3", "S3-1"]
    write_id_list_tsv(out, ["S1-1"], {"S1-1": ids}, "matched_entity_ids")
    assert read_id_list_tsv(out)["S1-1"] == ids


def test_written_file_parses_as_strict_tsv(tmp_path):
    """Two fields on every line, no quoting anywhere."""
    out = tmp_path / "o.tsv"
    write_id_list_tsv(out, ["S1-1", "S1-2"], {"S1-1": ["S2-1", "S2-2"]}, "matched_entity_ids")
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    assert all(len(r) == 2 for r in rows)
    assert '"' not in out.read_text(encoding="utf-8")


# ---- streamed writer ------------------------------------------------------

def test_grouped_writer_matches_dict_writer(tmp_path):
    import numpy as np
    from business_er.io import read_id_list_tsv, write_grouped_id_lists, write_id_list_tsv

    s1 = ["S1-5", "S1-1", "S1-9"]
    B = 2_000_000_000
    row = np.array([0, 0, 2, 0])
    codes = np.array([2 * B + 193, 3 * B + 812, 3 * B + 4, 2 * B + 47])
    write_grouped_id_lists(tmp_path / "g.tsv", s1, row, codes, "matched_entity_ids")
    write_id_list_tsv(tmp_path / "d.tsv", s1,
                      {"S1-5": ["S2-193", "S3-812", "S2-47"], "S1-9": ["S3-4"]},
                      "matched_entity_ids")
    assert (tmp_path / "g.tsv").read_bytes() == (tmp_path / "d.tsv").read_bytes()
    back = read_id_list_tsv(tmp_path / "g.tsv")
    assert list(back) == s1 and back["S1-1"] == []           # empty row kept, order kept


def test_grouped_writer_rejects_bad_input(tmp_path):
    import numpy as np
    from business_er.io import write_grouped_id_lists

    B = 2_000_000_000
    with pytest.raises(ValueError):          # duplicate inside a list
        write_grouped_id_lists(tmp_path / "x.tsv", ["S1-1"], np.array([0, 0]),
                               np.array([2 * B + 1, 2 * B + 1]), "c")
    with pytest.raises(ValueError):          # an S1 target
        write_grouped_id_lists(tmp_path / "x.tsv", ["S1-1"], np.array([0]), np.array([1 * B + 1]), "c")
    with pytest.raises(ValueError):          # duplicate S1 row
        write_grouped_id_lists(tmp_path / "x.tsv", ["S1-1", "S1-1"], np.array([], int),
                               np.array([], int), "c")


def test_group_writer_matches_sorting_writer(tmp_path):
    import numpy as np
    from business_er.io import write_grouped_id_lists, write_id_lists_by_group

    B = 2_000_000_000
    s1 = ["S1-5", "S1-1", "S1-9", "S1-7"]
    # group A owns S1 rows 0 and 3; group B owns row 2; row 1 has nothing
    ga = (np.array([0, 3]), np.array([0, 0, 1]), np.array([2 * B + 193, 3 * B + 812, 2 * B + 4]))
    gb = (np.array([2]), np.array([0]), np.array([3 * B + 47]))
    write_id_lists_by_group(tmp_path / "g.tsv", s1, [ga, gb], "c")
    rows = np.r_[ga[0][ga[1]], gb[0][gb[1]]]
    codes = np.r_[ga[2], gb[2]]
    write_grouped_id_lists(tmp_path / "s.tsv", s1, rows, codes, "c")
    assert (tmp_path / "g.tsv").read_bytes() == (tmp_path / "s.tsv").read_bytes()


def test_group_writer_rejects_bad_input(tmp_path):
    import numpy as np
    from business_er.io import write_id_lists_by_group

    B = 2_000_000_000
    with pytest.raises(ValueError):   # duplicate id inside a list
        write_id_lists_by_group(tmp_path / "x", ["S1-1"], [(np.array([0]), np.array([0, 0]),
                                np.array([2 * B + 1, 2 * B + 1]))], "c")
    with pytest.raises(ValueError):   # S1 record claimed by two groups
        g = (np.array([0]), np.array([0]), np.array([2 * B + 1]))
        write_id_lists_by_group(tmp_path / "x", ["S1-1"], [g, g], "c")
    with pytest.raises(ValueError):   # pairs not grouped by anchor
        write_id_lists_by_group(tmp_path / "x", ["S1-1", "S1-2"], [(np.array([0, 1]), np.array([1, 0]),
                                np.array([2 * B + 1, 2 * B + 2]))], "c")
