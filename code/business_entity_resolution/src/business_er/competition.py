"""Global competition statistics, materialized only for requested rows.

Selection limits OUTPUT rows, never the competing population. Sorting uses
all candidates, preserving claims from low-probability and unselected rows.
"""
import numpy as np


def selected_competition_features(anchor, target, p, selected=None):
    p=np.asarray(p)
    anchor=np.asarray(anchor); target=np.asarray(target)
    n=len(p)
    if not (len(anchor)==len(target)==n):
        raise ValueError('candidate columns differ in length')
    idx=np.arange(n,dtype=np.int64) if selected is None else np.asarray(selected,dtype=np.int64)
    if idx.ndim != 1 or (len(idx) and (idx.min()<0 or idx.max()>=n)):
        raise ValueError('selected rows outside candidate array')
    names=('a_pmax','a_gap','a_rank','a_n50','a_n90','a_psum','t_nclaims','t_rank','t_best_other','t_margin','t_n50_other')
    if not len(idx):
        return {k:np.empty(0,np.float32) for k in names}
    out={}
    rank_type=np.int32 if n < np.iinfo(np.int32).max else np.int64
    for key,side in ((anchor,'a'),(target,'t')):
        order=np.lexsort((-p,key))
        sorted_key=key[order]
        starts=np.flatnonzero(np.r_[True,sorted_key[1:]!=sorted_key[:-1]])
        del sorted_key
        sizes=np.diff(np.r_[starts,n])
        inverse=np.empty(n,dtype=rank_type)
        inverse[order]=np.arange(n,dtype=rank_type)
        pos=inverse[idx].astype(np.int64)
        del inverse
        group=np.searchsorted(starts,pos,side='right')-1
        rank=pos-starts[group]
        out[f'{side}_rank']=rank.astype(np.float32)
        sorted_p=p[order]
        n50=np.add.reduceat((sorted_p>.5).astype(np.int32),starts)
        if side=='a':
            out['a_pmax']=sorted_p[starts[group]].astype(np.float32)
            out['a_gap']=(out['a_pmax']-p[idx]).astype(np.float32)
            out['a_n50']=n50[group].astype(np.float32)
            n90=np.add.reduceat((sorted_p>.9).astype(np.int32),starts)
            out['a_n90']=n90[group].astype(np.float32)
            sums=np.add.reduceat(sorted_p,starts,dtype=np.float64)
            out['a_psum']=sums[group].astype(np.float32)
            del n90,sums
        else:
            best=sorted_p[starts[group]].copy()
            top=rank==0
            multiple=sizes[group]>1
            replace=top&multiple
            best[replace]=sorted_p[starts[group[replace]]+1]
            best[top&~multiple]=0
            out['t_best_other']=best.astype(np.float32)
            out['t_margin']=(p[idx]-out['t_best_other']).astype(np.float32)
            out['t_nclaims']=sizes[group].astype(np.float32)
            out['t_n50_other']=(n50[group]-(p[idx]>.5)).astype(np.float32)
        del order,sorted_p,starts,sizes,pos,group,rank,n50
    return out
