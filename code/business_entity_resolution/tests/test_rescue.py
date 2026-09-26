import numpy as np
from business_er.rescue import unclaimed_rescue


def test_preserves_claims_even_with_higher_probability():
    keep=unclaimed_rescue([10],[0,1,2],[10,11,12],[.999,.96,.94])
    assert keep.tolist()==[False,True,False]


def test_competition_and_ties_have_one_deterministic_owner():
    a=np.array([3,2,1,0]); t=np.array([10,10,20,20]); p=np.array([.97,.99,.99,.99])
    assert unclaimed_rescue([],a,t,p).tolist()==[False,True,False,True]
    assert unclaimed_rescue([],[],[],[]).tolist()==[]
