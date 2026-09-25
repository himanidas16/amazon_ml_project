"""Tests for the train / validation split.

The point of these tests is to prove the split cannot leak.  A leak does not
crash anything -- it just makes every later measurement optimistic, which is the
worst kind of bug because it feels like progress.
"""

import numpy as np
import pytest

from business_er.splits import (
    TruthCSR,
    bucket_of,
    check_split,
    load_split,
    load_truth_csr,
    make_split,
    save_split,
)


def write_gt(tmp_path, rows):
    p = tmp_path / "train_ground_truth.tsv"
    lines = ["source1_entity_id\tmatched_entity_ids"]
    lines += [f"{s1}\t{','.join(ids)}" for s1, ids in rows]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --------------------------------------------------------------------------
# reading labels into the compact form
# --------------------------------------------------------------------------

def test_reads_labels_into_flat_arrays(tmp_path):
    p = write_gt(tmp_path, [
        ("S1-10", ["S2-1", "S3-2"]),
        ("S1-11", []),                 # a singleton
        ("S1-12", ["S2-5"]),
    ])
    t = load_truth_csr(p)
    assert len(t) == 3
    assert t.s1_values.tolist() == [10, 11, 12]
    assert t.match_counts.tolist() == [2, 0, 1]
    assert t.truth_ids_of(0) == ["S2-1", "S3-2"]
    assert t.truth_ids_of(1) == []          # the singleton survives as empty
    assert t.truth_ids_of(2) == ["S2-5"]


def test_same_number_in_two_sources_stays_distinct(tmp_path):
    """S2-100 and S3-100 both exist in the real data and encode to 100, so the
    source has to travel with the value."""
    p = write_gt(tmp_path, [("S1-1", ["S2-100", "S3-100"])])
    t = load_truth_csr(p)
    vals, srcs = t.targets_of(0)
    assert vals.tolist() == [100, 100]
    assert srcs.tolist() == [2, 3]
    assert t.truth_ids_of(0) == ["S2-100", "S3-100"]


def test_detects_a_shared_target_instead_of_leaking(tmp_path):
    """The whole star-shaped-groups assumption rests on this.  If a future data
    drop shared a target between two S1 records, splitting by S1 would put the
    same record on both sides -- so we refuse to load rather than leak."""
    p = write_gt(tmp_path, [("S1-1", ["S2-7"]), ("S1-2", ["S2-7"])])
    with pytest.raises(ValueError, match="claimed by more than one S1"):
        load_truth_csr(p)


def test_rejects_duplicate_s1_rows_and_bad_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_truth_csr(write_gt(tmp_path, [("S1-1", ["S2-1"]), ("S1-1", ["S2-2"])]))
    with pytest.raises(ValueError, match="bad match id"):
        load_truth_csr(write_gt(tmp_path, [("S1-1", ["S1-2"])]))


def test_bucketing_keeps_zero_and_one_separate():
    """Singletons and single-match records behave differently from the rest, so
    they get their own strata."""
    assert bucket_of(0) == 0
    assert bucket_of(1) == 1
    assert bucket_of(2) == bucket_of(3) == 2
    assert bucket_of(11) == bucket_of(50) == 4


# --------------------------------------------------------------------------
# the split
# --------------------------------------------------------------------------

def build(n_s1=200, n_orphan=300, seed=1, n_folds=5):
    """A synthetic dataset shaped like the real one: two countries, a few
    singletons, multi-match groups, plus unmatched distractor records."""
    rng = np.random.default_rng(seed)
    s1_vals, s1_ctry, rows = [], [], []
    t2_vals, t2_ctry, t3_vals, t3_ctry = [], [], [], []
    nxt = 1000

    for i in range(n_s1):
        ctry = "US" if i % 3 else "India"
        s1_vals.append(i + 1)
        s1_ctry.append(ctry)
        k = int(rng.integers(0, 5))          # 0 means a singleton
        ids = []
        for j in range(k):
            nxt += 1
            if j % 2:
                t3_vals.append(nxt); t3_ctry.append(ctry); ids.append((3, nxt))
            else:
                t2_vals.append(nxt); t2_ctry.append(ctry); ids.append((2, nxt))
        rows.append(ids)

    for _ in range(n_orphan):                # distractors nobody matches
        nxt += 1
        ctry = "US" if nxt % 2 else "India"
        if nxt % 3:
            t2_vals.append(nxt); t2_ctry.append(ctry)
        else:
            t3_vals.append(nxt); t3_ctry.append(ctry)

    counts = [len(r) for r in rows]
    truth = TruthCSR(
        s1_values=np.array(s1_vals, dtype=np.int32),
        indptr=np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
        tgt_values=np.array([v for r in rows for _, v in r], dtype=np.int32),
        tgt_source=np.array([s for r in rows for s, _ in r], dtype=np.int8),
    )
    split = make_split(
        truth, s1_ctry,
        {2: np.array(t2_vals, dtype=np.int32), 3: np.array(t3_vals, dtype=np.int32)},
        {2: t2_ctry, 3: t3_ctry},
        n_folds=n_folds, seed=7,
    )
    return truth, split


def test_no_true_match_crosses_a_fold_boundary():
    """THE test.  A business and all its matches must stay together."""
    truth, split = build()
    assert check_split(split, truth) == []


def test_every_record_gets_exactly_one_fold():
    truth, split = build()
    assert (split.s1_fold >= 0).all()
    for source in (2, 3):
        assert (split.tgt_fold[source] >= 0).all()
        assert (split.tgt_fold[source] < split.n_folds).all()


def test_a_validation_record_never_appears_in_training():
    """Record-level disjointness, checked directly rather than assumed."""
    truth, split = build()
    for source in (2, 3):
        vals = split.tgt_values[source]
        folds = split.tgt_fold[source]
        val = set(vals[folds == 0].tolist())
        train = set(vals[folds != 0].tolist())
        assert not (val & train)
    s1_val = set(split.s1_values[split.s1_fold == 0].tolist())
    s1_train = set(split.s1_values[split.s1_fold != 0].tolist())
    assert not (s1_val & s1_train)


def test_folds_are_balanced_and_stratified():
    """Each fold should be a fair miniature: similar size, similar country mix,
    similar singleton rate.  Otherwise fold-to-fold comparisons are noise."""
    truth, split = build(n_s1=1000)
    sizes = [int((split.s1_fold == f).sum()) for f in range(split.n_folds)]
    # Dealing happens per stratum, so each of the ~10 strata can leave one
    # record of remainder.  What matters is that the imbalance stays tiny
    # relative to fold size, not that it is exactly zero.
    assert (max(sizes) - min(sizes)) / min(sizes) < 0.05

    counts = truth.match_counts
    rates = [
        float((counts[split.s1_fold == f] == 0).mean()) for f in range(split.n_folds)
    ]
    assert max(rates) - min(rates) < 0.06               # singleton rate is stable


def test_split_is_deterministic_for_a_given_seed():
    a_truth, a = build(seed=3)
    b_truth, b = build(seed=3)
    assert a.s1_fold.tolist() == b.s1_fold.tolist()
    assert a.tgt_fold[2].tolist() == b.tgt_fold[2].tolist()


def test_check_split_actually_catches_a_leak():
    """A guard that never fires is worthless, so break the split on purpose."""
    truth, split = build()
    assert check_split(split, truth) == []
    moved = 1 if split.s1_fold[0] != 1 else 0
    split.s1_fold[0] = moved                    # orphan this business's matches
    problems = check_split(split, truth)
    assert any("different fold" in p for p in problems)


def test_singletons_are_spread_across_folds():
    """Singletons are only 5.6% of the real data, so they must not pile into one
    fold -- that fold's score would be measuring something different."""
    truth, split = build(n_s1=1000)
    counts = truth.match_counts
    for f in range(split.n_folds):
        assert (counts[split.s1_fold == f] == 0).sum() > 0


def test_save_and_load_round_trip(tmp_path):
    truth, split = build()
    p = tmp_path / "split.npz"
    save_split(p, split)
    back = load_split(p)
    assert back.n_folds == split.n_folds and back.seed == split.seed
    assert back.s1_fold.tolist() == split.s1_fold.tolist()
    assert back.tgt_fold[3].tolist() == split.tgt_fold[3].tolist()
    assert check_split(back, truth) == []


def test_rejects_misaligned_country_list():
    truth, _ = build(n_s1=10)
    with pytest.raises(ValueError, match="not aligned"):
        make_split(truth, ["US"] * 3, {2: np.array([], np.int32), 3: np.array([], np.int32)},
                   {2: [], 3: []})


def test_rejects_labelled_id_missing_from_source_file():
    """If ground truth names a record the source file does not contain, that is
    a data problem we must surface, not silently drop."""
    truth, _ = build(n_s1=5)
    with pytest.raises(ValueError, match="absent from the source file"):
        make_split(
            truth, ["US"] * 5,
            {2: np.array([999999], np.int32), 3: np.array([999998], np.int32)},
            {2: ["US"], 3: ["US"]},
        )
