"""The official competition metric, implemented exactly.

Read this file carefully -- every later decision in the project is judged by it.

The score is computed PER SOURCE-1 ENTITY and then averaged.  It is NOT
computed over all pairs pooled together.  That difference is huge: an S1
record with 20 true matches counts exactly as much as one with zero.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence


def entity_f05(true_ids: Iterable[str], predicted_ids: Iterable[str]) -> float:
    """F0.5 for ONE Source-1 entity.

    truth = the set of S2/S3 ids that really match this S1 record
    pred  = the set of S2/S3 ids we output for it

    Singleton rule: if this S1 record truly has no matches, we score 1.0 only
    if we also predicted nothing, else 0.0.  There is no partial credit.
    """
    truth = set(true_ids)
    pred = set(predicted_ids)

    if not truth:
        # `not pred` is True when we correctly stayed silent -> float(True) == 1.0
        return float(not pred)

    tp = len(truth & pred)   # predicted and correct
    fp = len(pred - truth)   # predicted but wrong  (costly: coefficient 1)
    fn = len(truth - pred)   # missed              (cheaper: coefficient 0.25)

    # Count form of F0.5 = 1.25*P*R / (0.25*P + R).
    # Using counts avoids a divide-by-zero when tp == 0: the denominator is
    # fp + 0.25*fn, and fn > 0 whenever truth is non-empty, so it is safe.
    return 1.25 * tp / (1.25 * tp + fp + 0.25 * fn)


def macro_entity_f05(
    source1_ids: Sequence[str],
    truth_by_id: Mapping[str, Iterable[str]],
    pred_by_id: Mapping[str, Iterable[str]],
) -> float:
    """The leaderboard score: plain average of entity_f05 over every S1 record.

    We deliberately refuse to run on incomplete inputs.  A silently missing S1
    record would make the score look better than reality, which is exactly the
    kind of bug that costs a competition.
    """
    ids = list(source1_ids)
    if not ids:
        raise ValueError("source1_ids is empty")
    if len(ids) != len(set(ids)):
        raise ValueError("source1_ids contains duplicates")

    expected = set(ids)
    if set(truth_by_id) != expected:
        raise ValueError("truth_by_id must cover exactly the S1 ids")
    if set(pred_by_id) != expected:
        raise ValueError("pred_by_id must cover exactly the S1 ids")

    total = sum(entity_f05(truth_by_id[i], pred_by_id[i]) for i in ids)
    return total / len(ids)


def score_report(
    source1_ids: Sequence[str],
    truth_by_id: Mapping[str, Iterable[str]],
    pred_by_id: Mapping[str, Iterable[str]],
) -> Dict[str, float]:
    """Same score, broken into the pieces we actually need for decisions.

    Why bother: the overall number hides what is happening.  If 60% of S1
    records are singletons, predicting NOTHING at all already scores 0.60.
    Splitting the average into singleton and non-singleton halves tells us
    whether our matching is genuinely working or we are just riding that 0.60.
    """
    ids = list(source1_ids)
    macro = macro_entity_f05(ids, truth_by_id, pred_by_id)

    singleton_scores = []
    matched_scores = []
    tp = fp = fn = 0

    for i in ids:
        truth = set(truth_by_id[i])
        pred = set(pred_by_id[i])
        s = entity_f05(truth, pred)
        if truth:
            matched_scores.append(s)
        else:
            singleton_scores.append(s)
        tp += len(truth & pred)
        fp += len(pred - truth)
        fn += len(truth - pred)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    n_singleton = len(singleton_scores)
    return {
        "macro_f05": macro,
        "n_entities": float(len(ids)),
        "n_singletons": float(n_singleton),
        "n_non_singletons": float(len(matched_scores)),
        "singleton_rate": n_singleton / len(ids),
        # What an all-empty submission would score.  Our real score must beat it.
        "empty_baseline": n_singleton / len(ids),
        "singleton_score": mean(singleton_scores),
        "non_singleton_score": mean(matched_scores),
        # Pooled pair counts -- diagnostics only, NOT the competition metric.
        "pair_tp": float(tp),
        "pair_fp": float(fp),
        "pair_fn": float(fn),
        "pair_precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "pair_recall": tp / (tp + fn) if (tp + fn) else float("nan"),
    }


def blocking_report(
    truth_by_id: Mapping[str, Iterable[str]],
    candidates_by_id: Mapping[str, Iterable[str]],
    total_pairs: int | None = None,
) -> Dict[str, float]:
    """How good is our candidate generation, before any model runs?

    A true match that blocking never proposes is lost forever -- no model can
    recover it.  So this sets the ceiling on our final score.

    oracle_macro_f05 is the key number: it is the score a PERFECT model would
    get using these candidates.  If it is 0.80, we can never exceed 0.80.
    """
    ids = list(truth_by_id)
    if not ids:
        raise ValueError("truth_by_id is empty")

    found = missed = 0
    n_cand = 0
    per_entity_recall = []
    fully_covered = 0
    n_non_singleton = 0
    oracle_pred: Dict[str, set] = {}
    counts = []

    for i in ids:
        truth = set(truth_by_id[i])
        cand = set(candidates_by_id.get(i, ()))
        counts.append(len(cand))
        n_cand += len(cand)
        hit = truth & cand
        found += len(hit)
        missed += len(truth - cand)
        # A perfect model keeps exactly the true matches it was shown, and
        # stays silent for singletons.
        oracle_pred[i] = hit
        if truth:
            n_non_singleton += 1
            per_entity_recall.append(len(hit) / len(truth))
            if truth <= cand:
                fully_covered += 1

    counts.sort()

    def pct(p: float) -> float:
        k = min(len(counts) - 1, int(p * (len(counts) - 1)))
        return float(counts[k])

    out = {
        "pair_blocking_recall": found / (found + missed) if (found + missed) else float("nan"),
        "mean_per_entity_recall": (
            sum(per_entity_recall) / len(per_entity_recall) if per_entity_recall else float("nan")
        ),
        "full_coverage_rate": fully_covered / n_non_singleton if n_non_singleton else float("nan"),
        "oracle_macro_f05": macro_entity_f05(ids, truth_by_id, oracle_pred),
        "candidates_total": float(n_cand),
        "candidates_mean": n_cand / len(ids),
        "candidates_median": pct(0.50),
        "candidates_p95": pct(0.95),
        "candidates_max": float(counts[-1]),
        "missed_positives": float(missed),
    }
    if total_pairs:
        out["reduction_ratio"] = 1.0 - n_cand / total_pairs
    return out
