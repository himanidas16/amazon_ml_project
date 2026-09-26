"""Conservative addition of matches found by a complementary retriever."""
import numpy as np
from .evaluate import one_owner


def unclaimed_rescue(claimed_targets, anchor, target, probability, threshold=.95):
    """Select new matches without changing any existing target ownership.

    Inputs must already exclude pairs from the original candidate set. All
    claimed target IDs from the entire baseline population must be supplied,
    including businesses outside a scored validation subset. No labels used.
    """
    anchor=np.asarray(anchor)
    target=np.asarray(target)
    p=np.asarray(probability)
    if not (len(anchor)==len(target)==len(p)):
        raise ValueError('candidate columns differ in length')
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('invalid probabilities')
    keep=(p>=threshold)&~np.isin(target,claimed_targets)
    return one_owner(keep,p,target,anchor)
