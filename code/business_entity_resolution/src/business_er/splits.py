"""Building an honest train / validation split.

The problem this solves
-----------------------
We get no test labels, so the only way to know whether a change helps is to
hide part of the training data and score against it.  The classic mistake is to
split individual candidate PAIRS at random: the same business then appears on
both sides, the model has effectively seen the answer, and the local score
becomes meaningless.

So we split by BUSINESS.  An S1 record and every one of its true matches always
land on the same side of the fence.

Why that is simple here
-----------------------
In general you would build connected components of the "is a match" graph,
because a chain (S1-a -- S2-x -- S1-b) would force two S1 records into one
group.  We measured that this never happens: across all 7,638,365 true pairs, no
S2/S3 record is claimed by more than one S1 record.  Every group is therefore a
star -- one S1 plus its own matches -- and grouping reduces to "group by S1".

`load_truth_csr` asserts that property rather than trusting this comment, so if
a future data drop breaks it we find out immediately instead of leaking quietly.

Memory
------
A dict of 2.2M S1 ids -> 7.6M target id strings costs well over a gigabyte.  We
store the labels the way a sparse matrix is stored instead: one flat int32 array
of targets plus an offsets array saying where each S1's slice begins.  That is
about 100 MB for the whole training label set.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .ids import ID_DTYPE, decode

# Match-count buckets used for stratification.  We keep 0 and 1 on their own
# because they behave differently from the rest: singletons are the only rows
# where predicting nothing is correct, and single-match rows are where one false
# positive does the most damage.
COUNT_BUCKETS = [(0, 0), (1, 1), (2, 3), (4, 5), (6, 10**9)]


def bucket_of(n: int) -> int:
    for i, (lo, hi) in enumerate(COUNT_BUCKETS):
        if lo <= n <= hi:
            return i
    raise AssertionError(f"unbucketed match count {n}")


@dataclass
class TruthCSR:
    """Training labels in compressed sparse-row form.

    s1_values[i]                     -- encoded id of the i-th S1 record
    tgt_values[indptr[i]:indptr[i+1]] -- encoded ids of its matches
    tgt_source[...]                   -- 2 or 3, saying which file each came from

    The source must travel alongside the value because S2-100 and S3-100 both
    exist in this dataset and encode to the same number.
    """

    s1_values: np.ndarray      # int32, one per S1 record, file order
    indptr: np.ndarray         # int64, length len(s1_values) + 1
    tgt_values: np.ndarray     # int32, one per true match
    tgt_source: np.ndarray     # int8, 2 or 3

    def __len__(self) -> int:
        return len(self.s1_values)

    @property
    def match_counts(self) -> np.ndarray:
        return np.diff(self.indptr).astype(np.int32)

    def targets_of(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = self.indptr[i], self.indptr[i + 1]
        return self.tgt_values[lo:hi], self.tgt_source[lo:hi]

    def truth_ids_of(self, i: int) -> List[str]:
        """Decoded back to "S2-47" form -- for scoring and for eyeballing."""
        vals, srcs = self.targets_of(i)
        return [decode([v], int(s))[0] for v, s in zip(vals, srcs)]


def load_truth_csr(path: str | Path, check_exclusive: bool = True) -> TruthCSR:
    """Read train_ground_truth.tsv straight into flat arrays.

    We parse by hand rather than through pandas because we want the compact
    representation without ever materialising 7.6M Python strings.
    """
    path = Path(path)
    s1_vals: List[int] = []
    indptr: List[int] = [0]
    tgt_vals: List[int] = []
    tgt_srcs: List[int] = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header[:2] != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"{path.name}: unexpected header {header}")

        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\n")
            if not line:
                continue
            s1, tab, rest = line.partition("\t")
            if not tab:
                raise ValueError(f"{path.name}:{lineno}: no tab separator")
            if s1[:3] != "S1-":
                raise ValueError(f"{path.name}:{lineno}: bad S1 id {s1!r}")
            s1_vals.append(int(s1[3:]))

            if rest:
                for tok in rest.split(","):
                    if not tok:
                        continue
                    pre, num = tok[:3], tok[3:]
                    if pre == "S2-":
                        tgt_srcs.append(2)
                    elif pre == "S3-":
                        tgt_srcs.append(3)
                    else:
                        raise ValueError(f"{path.name}:{lineno}: bad match id {tok!r}")
                    tgt_vals.append(int(num))
            indptr.append(len(tgt_vals))

    truth = TruthCSR(
        s1_values=np.asarray(s1_vals, dtype=ID_DTYPE),
        indptr=np.asarray(indptr, dtype=np.int64),
        tgt_values=np.asarray(tgt_vals, dtype=ID_DTYPE),
        tgt_source=np.asarray(tgt_srcs, dtype=np.int8),
    )

    if len(np.unique(truth.s1_values)) != len(truth.s1_values):
        raise ValueError("duplicate source1_entity_id rows in ground truth")

    if check_exclusive:
        # The property the whole splitting strategy rests on.  Encode source and
        # value into one int64 key so S2-100 and S3-100 stay distinct.
        keys = truth.tgt_source.astype(np.int64) * 2_000_000_000 + truth.tgt_values
        if len(np.unique(keys)) != len(keys):
            raise ValueError(
                "a target record is claimed by more than one S1 record: groups are "
                "no longer stars, so splitting must use connected components"
            )
    return truth


# --------------------------------------------------------------------------
# the split itself
# --------------------------------------------------------------------------

@dataclass
class Split:
    """Which fold every record belongs to.

    Folds are numbered 0..n_folds-1 for S1 records.  Target records get the fold
    of their owning S1 record; unmatched ("orphan") targets are dealt out across
    folds so that every fold has a realistic supply of distractors.
    """

    n_folds: int
    seed: int
    s1_values: np.ndarray      # int32, file order
    s1_fold: np.ndarray        # int8, aligned with s1_values
    tgt_values: Dict[int, np.ndarray]   # source -> int32, file order
    tgt_fold: Dict[int, np.ndarray]     # source -> int8, aligned

    def s1_mask(self, fold: int) -> np.ndarray:
        return self.s1_fold == fold

    def target_mask(self, source: int, fold: int) -> np.ndarray:
        return self.tgt_fold[source] == fold

    def summary(self) -> str:
        lines = [f"split: {self.n_folds} folds, seed {self.seed}"]
        for f in range(self.n_folds):
            n1 = int(self.s1_mask(f).sum())
            n2 = int(self.target_mask(2, f).sum())
            n3 = int(self.target_mask(3, f).sum())
            ratio = (n2 + n3) / n1 if n1 else float("nan")
            lines.append(
                f"  fold {f}: S1={n1:>9,}  S2={n2:>9,}  S3={n3:>9,}  "
                f"targets per S1={ratio:5.2f}"
            )
        return "\n".join(lines)


def _stratified_fold_assignment(
    strata: np.ndarray, n_folds: int, seed: int
) -> np.ndarray:
    """Deal each stratum's members round-robin into folds, after shuffling.

    Doing it per stratum (rather than globally at random) keeps every fold's
    country mix and match-count mix close to the whole dataset's, so a fold is a
    fair miniature of the real problem rather than a lucky or unlucky sample.
    """
    rng = np.random.default_rng(seed)
    fold = np.empty(len(strata), dtype=np.int8)
    for value in np.unique(strata):
        idx = np.flatnonzero(strata == value)
        rng.shuffle(idx)
        # Round-robin gives near-exact equal sizes regardless of stratum size.
        fold[idx] = (np.arange(len(idx)) % n_folds).astype(np.int8)
    return fold


def make_split(
    truth: TruthCSR,
    s1_country: Sequence[str],
    target_values: Dict[int, np.ndarray],
    target_country: Dict[int, Sequence[str]],
    n_folds: int = 5,
    seed: int = 20260925,
) -> Split:
    """Assign every record to a fold.

    s1_country must be aligned with truth.s1_values (i.e. train_source1.tsv
    order).  target_values / target_country are keyed by source (2 and 3) and
    aligned with their own files.

    Three stages:
      1. Stratify S1 records by (country, match-count bucket) and deal them out.
      2. Give every matched target its owner's fold -- this is what keeps a
         business whole.
      3. Deal the leftover unmatched targets out, stratified by (country,
         source), so each fold gets its fair share of hard distractors.
    """
    if len(s1_country) != len(truth):
        raise ValueError("s1_country is not aligned with the ground truth")

    counts = truth.match_counts
    buckets = np.array([bucket_of(int(c)) for c in counts], dtype=np.int16)
    country_arr = np.asarray(s1_country, dtype=object)
    strata = np.array(
        [f"{c}|{b}" for c, b in zip(country_arr, buckets)], dtype=object
    )
    s1_fold = _stratified_fold_assignment(strata, n_folds, seed)

    # Stage 2: matched targets inherit their owner's fold.
    owner_fold = np.repeat(s1_fold, counts)          # one entry per true match
    tgt_fold: Dict[int, np.ndarray] = {}
    for source in (2, 3):
        vals = np.asarray(target_values[source], dtype=ID_DTYPE)
        ctry = np.asarray(target_country[source], dtype=object)
        if len(vals) != len(ctry):
            raise ValueError(f"source {source}: values and country disagree in length")

        fold = np.full(len(vals), -1, dtype=np.int8)

        # Map this source's labelled targets onto their rows.
        sel = truth.tgt_source == source
        labelled_vals = truth.tgt_values[sel]
        labelled_fold = owner_fold[sel]

        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        pos = np.searchsorted(sorted_vals, labelled_vals)
        pos_c = np.clip(pos, 0, max(len(sorted_vals) - 1, 0))
        found = (pos < len(sorted_vals)) & (sorted_vals[pos_c] == labelled_vals)
        if not found.all():
            raise ValueError(
                f"source {source}: {int((~found).sum())} labelled ids are absent "
                f"from the source file"
            )
        fold[order[pos_c]] = labelled_fold

        # Stage 3: orphans, stratified by country so folds stay comparable.
        orphan = np.flatnonzero(fold < 0)
        if len(orphan):
            orphan_strata = np.array(
                [f"{c}|{source}" for c in ctry[orphan]], dtype=object
            )
            fold[orphan] = _stratified_fold_assignment(
                orphan_strata, n_folds, seed + source
            )
        tgt_fold[source] = fold

    return Split(
        n_folds=n_folds,
        seed=seed,
        s1_values=truth.s1_values,
        s1_fold=s1_fold,
        tgt_values={s: np.asarray(target_values[s], dtype=ID_DTYPE) for s in (2, 3)},
        tgt_fold=tgt_fold,
    )


def check_split(split: Split, truth: TruthCSR) -> List[str]:
    """Prove the split does what it claims.  Returns a list of violations.

    This is not decoration.  A leak here inflates every later number, and the
    only symptom is a local score that the leaderboard refuses to reproduce.
    """
    problems: List[str] = []

    if (split.s1_fold < 0).any():
        problems.append("some S1 records were never assigned a fold")
    for source in (2, 3):
        if (split.tgt_fold[source] < 0).any():
            n = int((split.tgt_fold[source] < 0).sum())
            problems.append(f"source {source}: {n} targets were never assigned a fold")

    # The core guarantee: no true match crosses a fold boundary.
    owner_fold = np.repeat(split.s1_fold, truth.match_counts)
    for source in (2, 3):
        sel = truth.tgt_source == source
        vals = split.tgt_values[source]
        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        pos = np.searchsorted(sorted_vals, truth.tgt_values[sel])
        pos_c = np.clip(pos, 0, max(len(sorted_vals) - 1, 0))
        rows = order[pos_c]
        crossing = split.tgt_fold[source][rows] != owner_fold[sel]
        if crossing.any():
            problems.append(
                f"source {source}: {int(crossing.sum())} true matches sit in a "
                f"different fold from their S1 record"
            )
    return problems


def save_split(path: str | Path, split: Split) -> None:
    np.savez_compressed(
        path,
        n_folds=split.n_folds,
        seed=split.seed,
        s1_values=split.s1_values,
        s1_fold=split.s1_fold,
        tgt2_values=split.tgt_values[2],
        tgt2_fold=split.tgt_fold[2],
        tgt3_values=split.tgt_values[3],
        tgt3_fold=split.tgt_fold[3],
    )


def load_split(path: str | Path) -> Split:
    z = np.load(path)
    return Split(
        n_folds=int(z["n_folds"]),
        seed=int(z["seed"]),
        s1_values=z["s1_values"],
        s1_fold=z["s1_fold"],
        tgt_values={2: z["tgt2_values"], 3: z["tgt3_values"]},
        tgt_fold={2: z["tgt2_fold"], 3: z["tgt3_fold"]},
    )
