import numpy as np
import pytest
from business_er.competition import selected_competition_features


def test_selected_output_keeps_all_competitors_and_arbitrary_row_order():
    rng=np.random.default_rng(834)
    a=rng.integers(0,25,500); t=rng.integers(0,70,500)
    p=rng.choice([0.,.005,.2,.5,.9,.99,1.],500).astype(np.float32)
    selected=np.array([499,1,20,77,3,1])
    actual=selected_competition_features(a,t,p,selected)
    for j,i in enumerate(selected):
        ai=np.flatnonzero(a==a[i]); ti=np.flatnonzero(t==t[i])
        ar=sorted(ai,key=lambda k:(-p[k],k)).index(i)
        tr=sorted(ti,key=lambda k:(-p[k],k)).index(i)
        other=[p[k] for k in ti if k!=i]
        expected={'a_pmax':p[ai].max(),'a_gap':p[ai].max()-p[i],'a_rank':ar,
                  'a_n50':sum(p[ai]>.5),'a_n90':sum(p[ai]>.9),'a_psum':p[ai].astype(float).sum(),
                  't_nclaims':len(ti),'t_rank':tr,'t_best_other':max(other,default=0),
                  't_margin':p[i]-max(other,default=0),'t_n50_other':sum(v>.5 for v in other)}
        for key,value in expected.items():
            np.testing.assert_allclose(actual[key][j],value,rtol=1e-6,atol=1e-7)
    full=selected_competition_features(a,t,p)
    for key in full:
        np.testing.assert_array_equal(actual[key],full[key][selected])


def test_empty_and_invalid_selection():
    assert all(len(v)==0 for v in selected_competition_features([],[],[]).values())
    assert all(len(v)==0 for v in selected_competition_features([0],[1],[.2],[]).values())
    with pytest.raises(ValueError):
        selected_competition_features([0],[1],[.2],[1])
