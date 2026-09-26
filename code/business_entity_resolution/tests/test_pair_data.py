import numpy as np
import pandas as pd
import pytest
from business_er.features import FEATURE_NAMES
from business_er.pair_data import load_pair_parts

def test_sharded_loader_keeps_features_labels_weights_aligned(tmp_path):
    paths=[]
    for name,anchors in [("a",[3,1]),("b",[2,0])]:
        frame=pd.DataFrame({f:np.asarray(anchors,dtype=np.float32) for f in FEATURE_NAMES})
        frame["anchor"]=anchors; frame["label"]=np.asarray(anchors)%2; frame["weight"]=np.asarray(anchors)+1
        path=tmp_path/f"{name}.parquet"; frame.to_parquet(path,index=False); paths.append(path)
    X,y,a,w=load_pair_parts(paths,sort_anchors=True)
    np.testing.assert_equal(a,[0,1,2,3]); np.testing.assert_equal(X[:,0],a)
    np.testing.assert_equal(y,a%2); np.testing.assert_equal(w,a+1)
    X,y,a,w=load_pair_parts(paths)
    np.testing.assert_equal(a,[3,1,2,0]); np.testing.assert_equal(X[:,-1],a)
    with pytest.raises(ValueError,match="no pair"):
        load_pair_parts([])
