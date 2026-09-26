"""Additional, tie-aware candidate context; independent of the v1 feature schema."""
import numpy as np
import pandas as pd
from .features import FEATURE_NAMES

EXTRA_NAMES = ("combo_dense_rank", "combo_competition_rank", "combo_ties", "combo_gap",
               "name_dense_rank", "addr_dense_rank", "joint_min", "joint_product",
               "joint_max", "name_addr_disagreement", "addr_supported_name",
               "number_supported_name")
CONTEXT_FEATURE_NAMES = tuple(FEATURE_NAMES) + EXTRA_NAMES
CONTEXT_VERSION = "1.0.0"

def extend_features(frame):
    """All candidates of each anchor must be present before training subsampling."""
    n = frame.name_tset.fillna(0).to_numpy(dtype=np.float32)
    a = frame.addr_tset.fillna(0).to_numpy(dtype=np.float32)
    combo = n + a
    group = frame.anchor.to_numpy()
    def rank(v, method="dense"):
        return pd.Series(v).groupby(group, sort=False).rank(method=method, ascending=False).to_numpy(dtype=np.float32)-1
    gs = pd.Series(combo).groupby(group, sort=False)
    ties = pd.DataFrame({"g":group,"v":combo}).groupby(["g","v"],sort=False).v.transform("size").to_numpy(dtype=np.float32)
    extras = np.column_stack((rank(combo), rank(combo,"min"), ties,
        gs.transform("max").to_numpy()-combo, rank(n), rank(a),
        np.minimum(n,a), n*a, np.maximum(n,a), np.abs(n-a),
        frame.name_ratio.fillna(0).to_numpy()*a,
        n*frame.num_jacc.fillna(0).to_numpy()))
    return np.column_stack((frame[list(FEATURE_NAMES)].to_numpy(dtype=np.float32),extras)).astype(np.float32)


def refresh_context(frame):
    """Recompute v1 context after selecting a smaller set of candidate rows.

    Pairwise values and retrieval scores/ranks stay unchanged. Returned frame
    matches computing v1 features directly on the selected candidate universe.
    """
    from .features import _group_context
    out=frame.copy(); g=out.anchor.to_numpy(np.int64)
    if not len(out): return out
    inv=np.unique(g,return_inverse=True)[1]
    out["n_cands"]=np.bincount(inv)[inv].astype(np.float32)
    out["ret_score_gap"],_=_group_context(g,out.ret_score.to_numpy())
    for name in ("name_tset","addr_tset"):
        out[name+"_gap"],out[name+"_lead"]=_group_context(g,out[name].to_numpy())
    combo=out.name_tset.fillna(0).to_numpy()+out.addr_tset.fillna(0).to_numpy()
    order=np.lexsort((-combo,g)); starts=np.r_[0,np.flatnonzero(g[order][1:]!=g[order][:-1])+1]
    ranks=np.empty(len(out),np.float32); ranks[order]=np.arange(len(out))-np.repeat(starts,np.diff(np.r_[starts,len(out)]))
    out["combo_rank"]=ranks
    return out
