"""Tests for the matcher.  Synthetic numbers only -- not competition data."""

import numpy as np
import pytest

from business_er.features import FEATURE_NAMES
from business_er.train import Matcher, dev_split, train_matcher


def _data(n_anchor=400, per=10, seed=0):
    rng = np.random.default_rng(seed)
    n = n_anchor * per
    X = rng.random((n, len(FEATURE_NAMES))).astype(np.float32)
    X[rng.random(X.shape) < 0.05] = np.nan                     # missing values allowed
    y = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) > 1.2).astype(np.int8)
    return X, y, np.repeat(np.arange(n_anchor), per)


def test_dev_split_keeps_anchors_whole():
    anchor = np.repeat(np.arange(100), 7)
    dev = dev_split(anchor, 0.2)
    for a in range(100):
        assert len(set(dev[anchor == a])) == 1
    assert 0.15 < dev.mean() < 0.25


def test_train_save_load_round_trip(tmp_path):
    X, y, anchor = _data()
    m = train_matcher(X, y, anchor, params={"num_leaves": 7, "min_data_in_leaf": 5},
                      num_rounds=50, early_stopping=10, log_every=0)
    p = m.predict(X)
    assert p.shape == (len(X),) and 0 <= p.min() and p.max() <= 1
    assert ((p > 0.5) == y).mean() > 0.9                        # learnt the rule
    m.save(tmp_path / "m.lgb")
    back = Matcher.load(tmp_path / "m.lgb")
    assert np.array_equal(back.predict(X), p)
    assert back.meta["features_version"] == m.meta["features_version"]


def test_training_is_deterministic():
    X, y, anchor = _data()
    kw = dict(params={"num_leaves": 7, "min_data_in_leaf": 5}, num_rounds=30,
              early_stopping=10, log_every=0)
    assert np.array_equal(train_matcher(X, y, anchor, **kw).predict(X),
                          train_matcher(X, y, anchor, **kw).predict(X))


def test_wrong_feature_order_refused():
    X, y, anchor = _data(100, 5)
    m = train_matcher(X, y, anchor, params={"num_leaves": 3, "min_data_in_leaf": 5},
                      num_rounds=5, early_stopping=5, log_every=0)
    with pytest.raises(ValueError):
        m.predict(X, feature_names=list(reversed(FEATURE_NAMES)))


def test_weights_keep_probabilities_calibrated():
    # Subsample 90% of negatives away and re-weight the rest by 10: the mean
    # predicted probability should stay near the true positive rate.
    rng = np.random.default_rng(1)
    n = 20000
    X = rng.random((n, len(FEATURE_NAMES))).astype(np.float32)
    y = (rng.random(n) < 0.1 + 0.3 * X[:, 0]).astype(np.int8)
    anchor = np.arange(n) // 10
    keep = (y == 1) | (rng.random(n) < 0.1)
    w = np.where(y[keep] == 1, 1.0, 10.0)
    kw = dict(params={"num_leaves": 7, "min_data_in_leaf": 20}, num_rounds=60,
              early_stopping=20, log_every=0)
    m = train_matcher(X[keep], y[keep], anchor[keep], weight=w, **kw)
    unweighted = train_matcher(X[keep], y[keep], anchor[keep], **kw)
    true_rate = y.mean()
    assert abs(m.predict(X).mean() - true_rate) < 0.03              # weighted: calibrated
    assert unweighted.predict(X).mean() > true_rate + 0.1           # unweighted: inflated
