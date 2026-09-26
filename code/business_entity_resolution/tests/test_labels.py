import numpy as np
import pytest
from business_er.labels import label_candidates
from business_er.retrieve import target_code
from business_er.splits import TruthCSR

def test_country_shard_uses_local_not_global_anchor_indices():
    truth = TruthCSR(np.array([10, 11, 12, 13]), np.array([0, 1, 2, 2, 4]),
                     np.array([41, 42, 43, 44]), np.array([2, 3, 2, 3]))
    rows = np.array([3, 0, 2])
    anchor = np.array([0, 0, 0, 1, 1, 2])
    code = target_code(np.array([2, 3, 3, 2, 3, 2]), np.array([43, 44, 42, 41, 41, 43]))
    assert label_candidates(truth, rows, anchor, code).tolist() == [1, 1, 0, 1, 0, 0]
    assert label_candidates(truth, rows, [], []).size == 0
    with pytest.raises(ValueError, match="outside"):
        label_candidates(truth, rows, [3], [code[0]])
