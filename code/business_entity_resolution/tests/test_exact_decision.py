from itertools import product
import numpy as np
import pytest
from business_er.exact_decision import cardinality_utilities, expected_f05_exact


@pytest.mark.parametrize('p',[[.9,.6,.2],[1.,.5,0.],[.01,.01],[1.,1.],[0.,0.]])
def test_matches_exhaustive_outcome_enumeration(p):
    p = np.array(p)
    exact = np.zeros(len(p)+1)
    for bits in product([0,1],repeat=len(p)):
        y = np.array(bits)
        prob = np.prod(np.where(y,p,1-p))
        exact[0] += prob*(y.sum()==0)
        for k in range(1,len(p)+1):
            exact[k] += prob*5*y[:k].sum()/(4*k+y.sum())
    np.testing.assert_allclose(cardinality_utilities(p),exact,rtol=1e-12,atol=1e-12)


def test_singletons_group_boundaries_and_permutation():
    p = np.array([0.,0.,.2,1.,1.])
    anchor = np.array([0,0,3,3,3])
    assert expected_f05_exact(p,anchor).tolist() == [False,False,False,True,True]
    assert expected_f05_exact([],[]).tolist() == []
    with pytest.raises(ValueError):
        expected_f05_exact([.1,.2],[1,0])
