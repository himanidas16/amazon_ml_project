"""Step 7: the matcher -- a LightGBM binary classifier over pair features.

Question it answers, for each (S1 record, candidate) pair: "same business?"
It outputs a probability; turning that into yes/no is Step 8's job.

Choices
-------
* LightGBM (MIT licensed, ~millions of parameters at most -- far below the 8B
  limit, and no pretrained weights).  Gradient-boosted trees handle NaN
  natively, which our features rely on ("address missing" is NaN, not 0).
* Binary log-loss.  It is NOT the competition metric; the metric is applied
  when choosing the decision rule on held-out data.
* Early stopping on a DEV slice carved out of the training anchors (by
  anchor, never by pair), so the validation fold stays untouched until the
  final measurement.
* Deterministic: fixed seed, deterministic=True, force_col_wise, fixed thread
  count -- the same data gives the same model on this machine.
* The feature order is saved with the model and checked at predict time: a
  silently reordered column would not crash, it would just predict garbage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import lightgbm as lgb
import numpy as np

from .features import FEATURES_VERSION, FEATURE_NAMES

TRAIN_VERSION = "1.0.0"

DEFAULT_PARAMS: Dict = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "num_threads": 6,
    "seed": 20260925,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


@dataclass
class Matcher:
    booster: lgb.Booster
    feature_names: Sequence[str]
    meta: Dict = field(default_factory=dict)

    def predict(self, X: np.ndarray, feature_names: Sequence[str] = FEATURE_NAMES,
                num_threads: Optional[int] = None) -> np.ndarray:
        if list(feature_names) != list(self.feature_names):
            raise ValueError("feature order differs from the one the model was trained on")
        return self.booster.predict(X, num_threads=num_threads or DEFAULT_PARAMS["num_threads"])

    def save(self, path) -> None:
        path = Path(path)
        self.booster.save_model(str(path))
        path.with_suffix(".json").write_text(json.dumps(
            {"feature_names": list(self.feature_names), **self.meta}, indent=2))

    @classmethod
    def load(cls, path) -> "Matcher":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        names = meta.pop("feature_names")
        return cls(lgb.Booster(model_file=str(path)), names, meta)

    def importance(self, kind: str = "gain") -> Dict[str, float]:
        imp = self.booster.feature_importance(importance_type=kind)
        tot = imp.sum() or 1.0
        return dict(sorted(zip(self.feature_names, imp / tot), key=lambda x: -x[1]))


def dev_split(anchor: np.ndarray, dev_fraction: float = 0.1, seed: int = 7) -> np.ndarray:
    """Boolean mask over pairs: True = dev.  Split by ANCHOR so one S1
    record's candidates never sit on both sides."""
    ids = np.unique(anchor)
    rng = np.random.default_rng(seed)
    dev_ids = rng.choice(ids, size=max(1, int(len(ids) * dev_fraction)), replace=False)
    return np.isin(anchor, dev_ids)


def train_matcher(
    X: np.ndarray, y: np.ndarray, anchor: np.ndarray,
    params: Optional[Dict] = None, num_rounds: int = 3000, early_stopping: int = 100,
    dev_fraction: float = 0.1, log_every: int = 100,
) -> Matcher:
    p = {**DEFAULT_PARAMS, **(params or {})}
    dev = dev_split(anchor, dev_fraction)
    names = list(FEATURE_NAMES)
    dtrain = lgb.Dataset(X[~dev], label=y[~dev], feature_name=names, free_raw_data=True)
    ddev = lgb.Dataset(X[dev], label=y[dev], feature_name=names, reference=dtrain)
    evals: Dict = {}
    booster = lgb.train(
        p, dtrain, num_boost_round=num_rounds, valid_sets=[ddev], valid_names=["dev"],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                   lgb.log_evaluation(log_every), lgb.record_evaluation(evals)],
    )
    meta = {
        "train_version": TRAIN_VERSION, "features_version": FEATURES_VERSION,
        "lightgbm_version": lgb.__version__, "params": p,
        "best_iteration": booster.best_iteration,
        "dev_logloss": float(min(evals["dev"]["binary_logloss"])),
        "n_train_pairs": int((~dev).sum()), "n_dev_pairs": int(dev.sum()),
    }
    return Matcher(booster, names, meta)
