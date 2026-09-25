"""Compact integer representation of entity ids.

Why this file exists
--------------------
Every id in this dataset looks like "S2-" followed by up to 9 digits, with no
leading zeros (verified on all six files: 0 exceptions).  The largest value is
999,999,995, which fits in a signed 32-bit integer.

That lets us store an id as 4 bytes instead of a ~70-byte Python string.  On
the 10.3M test target records that is the difference between roughly 40 MB and
roughly 800 MB -- and we have to hold several such arrays at once on a 15 GB
machine.  This is what makes the pipeline fit in memory at all.

The rules we rely on, all of them checked in tests:
  * the numeric part is exact -- int(s) then str() gives back the same digits
  * the source lives in the prefix, so we keep it separately rather than
    squeezing it into the number
  * conversion back to a string must be byte-identical, because the submission
    file has to contain the organizers' exact ids
"""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

# The only three prefixes that appear in the data.
SOURCE_PREFIX = {1: "S1-", 2: "S2-", 3: "S3-"}
PREFIX_SOURCE = {v: k for k, v in SOURCE_PREFIX.items()}

ID_DTYPE = np.int32
# Signed 32-bit holds up to 2,147,483,647; the data maxes out below 10^9.
MAX_ID_VALUE = 2_000_000_000


def source_of(entity_id: str) -> int:
    """1, 2 or 3 from the "S1-"/"S2-"/"S3-" prefix."""
    try:
        return PREFIX_SOURCE[entity_id[:3]]
    except KeyError:
        raise ValueError(f"unrecognised id prefix: {entity_id!r}") from None


def encode(ids: Iterable[str], expected_source: int | None = None) -> np.ndarray:
    """["S2-47", "S2-193"] -> int32 array([47, 193]).

    We validate rather than trust: a silently mangled id would corrupt every
    downstream join, and the damage would only surface as a bad score.
    """
    s = pd.Series(list(ids), dtype="string")
    if len(s) == 0:
        return np.empty(0, dtype=ID_DTYPE)

    prefixes = s.str.slice(0, 3)
    tails = s.str.slice(3)

    bad_prefix = ~prefixes.isin(SOURCE_PREFIX.values())
    if bad_prefix.any():
        raise ValueError(f"bad id prefix, e.g. {s[bad_prefix].head(3).tolist()}")

    if expected_source is not None:
        wanted = SOURCE_PREFIX[expected_source]
        wrong = prefixes != wanted
        if wrong.any():
            raise ValueError(
                f"expected source {wanted!r}, e.g. got {s[wrong].head(3).tolist()}"
            )

    # Only plain digits, and no leading zero that str(int(...)) would drop.
    not_digits = ~tails.str.fullmatch(r"[0-9]+").fillna(False)
    if not_digits.any():
        raise ValueError(f"non-numeric id tail, e.g. {s[not_digits].head(3).tolist()}")
    leading_zero = tails.str.len().gt(1) & tails.str.startswith("0")
    if leading_zero.any():
        raise ValueError(
            f"id with a leading zero cannot round-trip, e.g. "
            f"{s[leading_zero].head(3).tolist()}"
        )

    values = pd.to_numeric(tails).to_numpy()
    if values.max() > MAX_ID_VALUE:
        raise ValueError(f"id value {values.max()} does not fit in int32")
    return values.astype(ID_DTYPE)


def decode(values: Sequence[int] | np.ndarray, source: int) -> List[str]:
    """int32 array([47, 193]) + source 2 -> ["S2-47", "S2-193"]."""
    prefix = SOURCE_PREFIX[source]
    return [prefix + str(int(v)) for v in np.asarray(values)]


class IdIndex:
    """Turns ids into dense row positions 0..n-1, and back.

    Most of the pipeline wants to say "row 5 of the S2 table" rather than carry
    the id around.  This class is the bridge.  It sorts the encoded values once
    so lookups are a binary search (vectorised), which is far cheaper in memory
    than a Python dict of 5M strings.
    """

    __slots__ = ("source", "_sorted", "_order")

    def __init__(self, ids: Iterable[str] | np.ndarray, source: int):
        self.source = source
        values = (
            np.asarray(ids, dtype=ID_DTYPE)
            if isinstance(ids, np.ndarray)
            else encode(ids, source)
        )
        if len(np.unique(values)) != len(values):
            raise ValueError("ids are not unique")
        self._order = np.argsort(values, kind="stable").astype(np.int64)
        self._sorted = values[self._order]

    def __len__(self) -> int:
        return len(self._sorted)

    @property
    def values(self) -> np.ndarray:
        """Encoded ids in their original row order."""
        out = np.empty(len(self._sorted), dtype=ID_DTYPE)
        out[self._order] = self._sorted
        return out

    def positions(self, values: np.ndarray, missing: int = -1) -> np.ndarray:
        """Encoded ids -> row positions.  Unknown ids become `missing` (-1).

        Returning -1 instead of raising is deliberate: ground truth can point at
        an id that is not in the split we loaded, and the caller should decide
        whether that is an error or an expected filter.
        """
        values = np.asarray(values, dtype=ID_DTYPE)
        idx = np.searchsorted(self._sorted, values)
        idx_clipped = np.clip(idx, 0, max(len(self._sorted) - 1, 0))
        found = (idx < len(self._sorted)) & (self._sorted[idx_clipped] == values)
        out = np.where(found, self._order[idx_clipped], missing)
        return out.astype(np.int64)

    def contains(self, values: np.ndarray) -> np.ndarray:
        return self.positions(values) >= 0
