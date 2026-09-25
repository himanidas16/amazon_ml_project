"""Tests for the decision rules.  Tiny invented numbers only."""

import numpy as np
import pytest

from business_er.evaluate import expected_f05, one_owner, rescue, score_keep, threshold


def test_one_owner_keeps_best_claim_only():
    anchor = np.array([0, 0, 1, 2])
    target = np.array([7, 8, 7, 7])
    p = np.array([0.9, 0.8, 0.95, 0.6])
    keep = one_owner(threshold(p, 0.5), p, target, anchor)
    assert keep.tolist() == [False, True, True, False]     # target 7 goes to anchor 1


def test_one_owner_ignores_unkept_pairs():
    anchor = np.array([0, 1]); target = np.array([7, 7]); p = np.array([0.9, 0.3])
    assert one_owner(threshold(p, 0.5), p, target, anchor).tolist() == [True, False]


def test_rescue_only_empty_anchors():
    anchor = np.array([0, 0, 1, 1, 2])
    p = np.array([0.9, 0.6, 0.4, 0.3, 0.1])
    keep = rescue(threshold(p, 0.8), p, anchor, t2=0.35)
    # anchor 0 already has one; anchor 1 rescues its best (0.4); anchor 2 too weak
    assert keep.tolist() == [True, False, True, False, False]


def test_expected_f05_behaviour():
    # clear single match; two clear matches; nothing plausible
    anchor = np.array([0, 0, 1, 1, 1, 2, 2])
    p = np.array([0.95, 0.05, 0.9, 0.85, 0.02, 0.05, 0.03])
    keep = expected_f05(p, anchor)
    assert keep.tolist() == [True, False, True, True, False, False, False]


def test_expected_f05_is_order_independent_within_anchor():
    anchor = np.array([0, 0, 0]); p = np.array([0.2, 0.9, 0.85])
    assert expected_f05(p, anchor).tolist() == [False, True, True]


def test_score_keep_counts_singletons_and_misses():
    anchor = np.array([0, 1]); label = np.array([1, 0])
    n_true = np.array([2, 0, 1])               # row 2 has no candidates at all
    keep = np.array([True, False])
    s = score_keep(keep, label, anchor, n_true)
    assert s["macro_f05"] == pytest.approx((1.25 / 1.5 + 1 + 0) / 3)
    assert s["singleton"] == 1.0


def test_unsorted_anchor_rejected():
    with pytest.raises(ValueError):
        expected_f05(np.array([0.5, 0.5]), np.array([1, 0]))
