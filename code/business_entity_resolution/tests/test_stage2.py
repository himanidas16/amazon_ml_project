"""Tests for competing-claim features (tiny hand-made numbers)."""

import numpy as np

from business_er.stage2 import competition_features


def test_competition_features_by_hand():
    # record 7 is claimed by businesses 0 (p=.9) and 1 (p=.6); record 8 only by 0
    anchor = np.array([0, 0, 1])
    target = np.array([7, 8, 7])
    p = np.array([0.9, 0.3, 0.6])
    c = competition_features(anchor, target, p)
    assert c["t_nclaims"].tolist() == [2, 1, 2]
    assert np.allclose(c["t_best_other"], [0.6, 0.0, 0.9])
    assert np.allclose(c["t_margin"], [0.3, 0.3, -0.3])
    assert c["t_rank"].tolist() == [0, 0, 1]
    assert c["t_n50_other"].tolist() == [1, 0, 1]
    assert np.allclose(c["a_pmax"], [0.9, 0.9, 0.6]) and c["a_rank"].tolist() == [0, 1, 0]
    assert c["a_n50"].tolist() == [1, 1, 1]


def test_order_does_not_matter():
    rng = np.random.default_rng(0)
    anchor = rng.integers(0, 50, 400); target = rng.integers(0, 120, 400); p = rng.random(400)
    a = competition_features(anchor, target, p)
    o = rng.permutation(400)
    b = competition_features(anchor[o], target[o], p[o])
    for k in a:
        assert np.allclose(a[k][o], b[k]), k
