"""Tests for the compact integer id representation.

These matter more than they look.  The submission file must contain the
organizers' ids byte for byte, so any id that fails to round-trip is a silent
scoring bug.  We therefore prove the round trip and prove that the dangerous
cases raise instead of guessing.
"""

import numpy as np
import pytest

from business_er.ids import ID_DTYPE, IdIndex, decode, encode, source_of


def test_round_trip_is_exact():
    ids = ["S2-47", "S2-193", "S2-999999995", "S2-1"]
    assert decode(encode(ids, 2), 2) == ids


def test_uses_four_bytes_per_id():
    """The whole point: int32, not 8-byte ints and not Python strings."""
    arr = encode(["S3-1", "S3-2"], 3)
    assert arr.dtype == ID_DTYPE
    assert arr.itemsize == 4


def test_source_comes_from_the_prefix():
    assert source_of("S1-5") == 1
    assert source_of("S2-5") == 2
    assert source_of("S3-5") == 3
    with pytest.raises(ValueError):
        source_of("S9-5")


def test_same_number_in_two_sources_stays_distinct():
    """S2-100 and S3-100 both exist in this dataset.  They encode to the same
    number, so the source must always be carried alongside -- never inferred."""
    assert encode(["S2-100"], 2)[0] == encode(["S3-100"], 3)[0]
    assert decode([100], 2) == ["S2-100"]
    assert decode([100], 3) == ["S3-100"]


def test_rejects_a_leading_zero_instead_of_corrupting_it():
    """"S2-007" would come back as "S2-7".  The real files have no such ids
    (verified), so the right response is a loud error, not a quiet rename."""
    with pytest.raises(ValueError, match="leading zero"):
        encode(["S2-007"], 2)


def test_rejects_wrong_prefix_and_junk_tails():
    with pytest.raises(ValueError, match="expected source"):
        encode(["S3-1"], 2)
    with pytest.raises(ValueError, match="bad id prefix"):
        encode(["X1-1"])
    with pytest.raises(ValueError, match="non-numeric"):
        encode(["S2-12a"], 2)


def test_empty_input_is_allowed():
    """Plenty of S1 records will have no candidates at all."""
    arr = encode([], 2)
    assert len(arr) == 0 and arr.dtype == ID_DTYPE


# --------------------------------------------------------------------------
# IdIndex: id -> row position
# --------------------------------------------------------------------------

def test_index_maps_ids_to_their_original_row_order():
    ids = ["S2-500", "S2-3", "S2-77"]          # deliberately unsorted
    idx = IdIndex(ids, 2)
    assert len(idx) == 3
    pos = idx.positions(encode(ids, 2))
    assert pos.tolist() == [0, 1, 2]
    assert decode(idx.values, 2) == ids


def test_index_reports_unknown_ids_as_minus_one():
    """Ground truth may name an id outside the split we loaded; the caller
    decides whether that is an error, so we signal rather than raise."""
    idx = IdIndex(["S2-1", "S2-2"], 2)
    pos = idx.positions(encode(["S2-2", "S2-999"], 2))
    assert pos.tolist() == [1, -1]
    assert idx.contains(encode(["S2-1", "S2-999"], 2)).tolist() == [True, False]


def test_index_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="not unique"):
        IdIndex(["S2-1", "S2-1"], 2)


def test_index_round_trips_at_a_realistic_size():
    """Sparse, random, 9-digit ids -- the shape of the real files."""
    rng = np.random.default_rng(0)
    values = rng.choice(1_000_000_000, size=50_000, replace=False)
    ids = [f"S3-{v}" for v in values]
    idx = IdIndex(ids, 3)
    pos = idx.positions(encode(ids, 3))
    assert pos.tolist() == list(range(len(ids)))
    assert decode(idx.values, 3) == ids
