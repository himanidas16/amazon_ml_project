"""Tests for the competition metric.

The first test is the important one: it reproduces the worked example printed
in the official problem statement.  If that number is wrong, nothing else in
the project can be trusted.
"""

import math

import pytest

from business_er.metrics import (
    blocking_report,
    entity_f05,
    macro_entity_f05,
    score_report,
)


def test_official_worked_example():
    """From the problem statement: predict 3, of which 2 are right, truth is 2.
    Precision = 2/3, Recall = 1.0, F0.5 = 0.714."""
    score = entity_f05(["S2-00047", "S3-00812"], ["S2-00047", "S2-00193", "S3-00812"])
    assert score == pytest.approx(0.714, abs=5e-4)
    assert score == pytest.approx(5 / 7)


@pytest.mark.parametrize(
    "truth, pred, expected",
    [
        ([], [], 1.0),                    # singleton, correctly silent
        ([], ["A"], 0.0),                 # singleton, false merge -> total loss
        (["A"], [], 0.0),                 # had a match, said nothing
        (["A", "B"], ["A", "B"], 1.0),    # perfect
        (["A", "B"], ["A", "B", "C"], 5 / 7),   # one false positive
        (["A", "B"], ["A"], 5 / 6),             # one miss (cheaper than an FP)
        (["A"], ["B"], 0.0),              # wrong answer
    ],
)
def test_reference_table(truth, pred, expected):
    assert entity_f05(truth, pred) == pytest.approx(expected)


def test_false_positive_hurts_more_than_false_negative():
    """The whole point of beta=0.5: with two true matches, adding a wrong id
    costs more than dropping a right one.  This is why our threshold will end
    up well above 0.5."""
    truth = ["A", "B"]
    cost_of_extra = 1 - entity_f05(truth, ["A", "B", "C"])   # 2/7
    cost_of_missing = 1 - entity_f05(truth, ["A"])           # 1/6
    assert cost_of_extra > cost_of_missing


def test_input_order_and_duplicates_do_not_matter():
    """We compare sets, so a different order or a repeated id scores the same.
    The submission writer is what forbids duplicates on disk."""
    assert entity_f05(["A", "B"], ["B", "A"]) == 1.0
    assert entity_f05(["A", "B"], ["A", "A", "B"]) == 1.0


def test_no_division_by_zero_when_nothing_is_correct():
    assert entity_f05(["A", "B", "C"], ["X", "Y"]) == 0.0


def test_macro_averages_per_entity_not_per_pair():
    """This is the trap the metric is designed around.

    S1-1 has 10 true matches and we get them all.  S1-2 has 1 true match and we
    get it wrong.  Pooling pairs would say 10/11 = 0.91.  The real metric gives
    each ENTITY equal weight, so the answer is (1.0 + 0.0)/2 = 0.5.
    """
    ids = ["S1-1", "S1-2"]
    truth = {"S1-1": [f"S2-{i}" for i in range(10)], "S1-2": ["S2-99"]}
    pred = {"S1-1": [f"S2-{i}" for i in range(10)], "S1-2": ["S2-98"]}
    assert macro_entity_f05(ids, truth, pred) == pytest.approx(0.5)


def test_macro_refuses_incomplete_predictions():
    """A missing S1 row must be an error, not a silently higher score."""
    ids = ["S1-1", "S1-2"]
    truth = {"S1-1": ["S2-1"], "S1-2": []}
    with pytest.raises(ValueError):
        macro_entity_f05(ids, truth, {"S1-1": ["S2-1"]})       # S1-2 absent
    with pytest.raises(ValueError):
        macro_entity_f05(["S1-1", "S1-1"], truth, truth)        # duplicate id


def test_empty_baseline_equals_singleton_rate():
    """Predicting nothing at all scores exactly the fraction of singletons.
    On this dataset that is ~0.056, so it is a floor, not a strategy."""
    ids = [f"S1-{i}" for i in range(10)]
    truth = {i: ([] if int(i.split("-")[1]) < 3 else ["S2-1"]) for i in ids}
    pred = {i: [] for i in ids}
    rep = score_report(ids, truth, pred)
    assert rep["singleton_rate"] == pytest.approx(0.3)
    assert rep["macro_f05"] == pytest.approx(0.3)
    assert rep["empty_baseline"] == pytest.approx(0.3)
    assert rep["non_singleton_score"] == pytest.approx(0.0)


def test_score_report_separates_the_two_populations():
    ids = ["A", "B"]
    truth = {"A": [], "B": ["S2-1", "S2-2"]}
    pred = {"A": [], "B": ["S2-1"]}
    rep = score_report(ids, truth, pred)
    assert rep["singleton_score"] == pytest.approx(1.0)
    assert rep["non_singleton_score"] == pytest.approx(5 / 6)
    assert rep["macro_f05"] == pytest.approx((1.0 + 5 / 6) / 2)
    assert rep["pair_tp"] == 1 and rep["pair_fn"] == 1 and rep["pair_fp"] == 0


def test_blocking_oracle_is_an_upper_bound():
    """Blocking gives S1-1 both its true matches but misses one of S1-2's.
    A perfect model could then score 1.0 on the first and 5/6 on the second."""
    truth = {"S1-1": ["S2-1", "S2-2"], "S1-2": ["S2-3", "S2-4"], "S1-3": []}
    cands = {"S1-1": ["S2-1", "S2-2", "S2-9"], "S1-2": ["S2-3"], "S1-3": ["S2-7"]}
    rep = blocking_report(truth, cands, total_pairs=100)
    assert rep["pair_blocking_recall"] == pytest.approx(3 / 4)
    assert rep["full_coverage_rate"] == pytest.approx(0.5)
    assert rep["missed_positives"] == 1
    assert rep["oracle_macro_f05"] == pytest.approx((1.0 + 5 / 6 + 1.0) / 3)
    assert rep["reduction_ratio"] == pytest.approx(1 - 5 / 100)


def test_a_missed_candidate_lowers_the_ceiling():
    """Sanity check that blocking quality really does cap the final score."""
    truth = {"S1-1": ["S2-1", "S2-2"]}
    good = blocking_report(truth, {"S1-1": ["S2-1", "S2-2"]})
    bad = blocking_report(truth, {"S1-1": ["S2-1"]})
    assert good["oracle_macro_f05"] == 1.0
    assert bad["oracle_macro_f05"] < good["oracle_macro_f05"]


def test_singleton_with_candidates_still_reaches_one():
    """Blocking proposing candidates for a true singleton does not hurt the
    ceiling -- a perfect model would simply reject them all."""
    rep = blocking_report({"S1-1": []}, {"S1-1": ["S2-1", "S2-2"]})
    assert rep["oracle_macro_f05"] == 1.0
    assert math.isnan(rep["full_coverage_rate"])
