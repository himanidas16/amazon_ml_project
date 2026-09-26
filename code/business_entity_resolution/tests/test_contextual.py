import numpy as np
import pandas as pd
from business_er.features import FEATURE_NAMES
from business_er.contextual import extend_features, CONTEXT_FEATURE_NAMES

def test_ties_share_rank_and_context_is_per_anchor():
    df=pd.DataFrame(np.zeros((4,len(FEATURE_NAMES))),columns=FEATURE_NAMES)
    df["anchor"]=[0,0,0,1]
    df["name_tset"]=[1,1,.5,.3]; df["addr_tset"]=[1,1,.5,.2]
    x=extend_features(df); col=lambda k: x[:,CONTEXT_FEATURE_NAMES.index(k)]
    np.testing.assert_equal(col("combo_dense_rank"),[0,0,1,0])
    np.testing.assert_equal(col("combo_competition_rank"),[0,0,2,0])
    np.testing.assert_equal(col("combo_ties"),[2,2,1,1])
    np.testing.assert_allclose(col("combo_gap"),[0,0,1,0])
    np.testing.assert_allclose(x[:,:len(FEATURE_NAMES)],df[list(FEATURE_NAMES)].to_numpy(),rtol=1e-6)


def test_refresh_context_matches_recomputing_selected_pairs():
    from business_er.features import prepare, compute_features
    from business_er.contextual import refresh_context
    A=prepare(['Acme Ltd','North Shop'],['12 Main Road','8 Hill Lane'])
    T=prepare(['Acme','Acme Ltd','North','Different'],['12 Main Rd','80 Elsewhere','8 Hill Ln','1 West St'])
    anchors=np.array([0,0,1,1]); targets=np.arange(4)
    meta={'anchor':anchors,'score':np.array([8.,7.,6.,5.]),'rank':np.array([0,1,0,1]),'channels':np.ones(4,np.uint16),'source':np.full(4,2)}
    full=compute_features(A,T,anchors,targets,['US']*4,meta)
    keep=np.array([True,False,True,False])
    frame=pd.DataFrame(full[keep],columns=FEATURE_NAMES); frame['anchor']=anchors[keep]
    expected=compute_features(A,T,anchors[keep],targets[keep],['US']*2,{k:v[keep] for k,v in meta.items()})
    np.testing.assert_allclose(refresh_context(frame)[list(FEATURE_NAMES)].to_numpy(),expected,equal_nan=True)
