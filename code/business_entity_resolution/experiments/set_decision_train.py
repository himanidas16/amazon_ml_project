"""Learn expected per-entity F0.5 from the original matcher's held-out dev anchors.

The v1 matcher never fitted these anchors (they were used for early stopping).
Train a small utility regressor for top-k sets, counting all unretrieved truth.
Evaluate modifications on the original external validation businesses.
"""
import sys,time,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
from business_er.train import Matcher,dev_split
from business_er.features import FEATURE_NAMES
from business_er.metrics import entity_f05_counts
from business_er.evaluate import one_owner
out=ROOT/'artifacts/set_decision_v1'; out.mkdir(parents=True,exist_ok=True); t0=time.time()
def log(s): print(f'[{time.time()-t0:.1f}s] {s}',flush=True)
fields=['anchor','target_code','label','name_ratio','name_tset','addr_tset','num_jacc','addr_empty_a','addr_empty_t','indic_a','indic_t']
A=pd.read_parquet(ROOT/'artifacts/anchors_train.parquet'); ids=pd.read_parquet(ROOT/'artifacts/pairs_train.parquet',columns=['anchor']).anchor.to_numpy(); dev_ids=np.unique(ids[dev_split(ids)]); del ids
if (out/'dev_pairs.parquet').exists():
    D=pd.read_parquet(out/'dev_pairs.parquet')
else:
    matcher=Matcher.load(ROOT/'artifacts/models/matcher_v1.lgb'); parts=[]
    for batch in pq.ParquetFile(ROOT/'artifacts/pairs_train.parquet').iter_batches(batch_size=100000):
        df=batch.to_pandas(); df=df[df.anchor.isin(dev_ids)]
        if len(df):
            p=matcher.predict(df[list(FEATURE_NAMES)].to_numpy(np.float32),num_threads=2)
            piece=df[fields].copy(); piece['p']=p; parts.append(piece)
    D=pd.concat(parts,ignore_index=True); D.to_parquet(out/'dev_pairs.parquet',index=False); del parts,matcher
log(f'{len(dev_ids)} original early-stopping dev anchors; {len(D)} pairs')
K=10
names=[f'p{i}' for i in range(K)]+['prob_sum','uncertainty','count_50','count_80','count_95','best_name','best_addr','best_num','best_missing','indic','k','sum_k','min_k','next_p','k_over_sum']
def table(df,anchors):
    order=np.lexsort((-df.p.to_numpy(),df.anchor.to_numpy()))
    d=df.iloc[order].reset_index(drop=True); vals=d.anchor.to_numpy(); starts=np.flatnonzero(np.r_[True,vals[1:]!=vals[:-1]]); ends=np.r_[starts[1:],len(d)]
    xx=[]; yy=[]; group=[]; positions=[]; base=[]
    for start,end in zip(starts,ends):
        row=d.iloc[start:end]; aid=int(row.anchor.iloc[0]); nt=int(anchors.n_true.iloc[aid]); p=row.p.to_numpy(); y=row.label.to_numpy(); top=np.pad(p[:K],(0,max(0,K-len(p))))
        common=np.r_[top,p.sum(),np.sum(p*(1-p)),(p>=.5).sum(),(p>=.8).sum(),(p>=.95).sum(),row.name_ratio.iloc[0],row.addr_tset.iloc[0],row.num_jacc.iloc[0],row.addr_empty_a.iloc[0]+row.addr_empty_t.iloc[0],max(row.indic_a.iloc[0],row.indic_t.iloc[0])]
        ks=np.arange(min(K,len(p))+1); tp=np.r_[0,np.cumsum(y)][:len(ks)]; utility=entity_f05_counts(tp,ks-tp,np.full(len(ks),nt))
        for k,u in zip(ks,utility):
            xx.append(np.r_[common,k,p[:k].sum(),p[k-1] if k else 0.,p[k] if k<len(p) else 0.,k/max(p.sum(),.01)]); yy.append(u); group.append(aid)
        positions.append((aid,start,end,len(ks))); base.append(min(int((p>=.8).sum()),K))
    return np.asarray(xx,np.float32),np.asarray(yy,np.float32),np.asarray(group),d,positions,np.asarray(base)
X,y,g,_,_,_=table(D,A); del D; gc.collect(); dev=dev_split(g,dev_fraction=.2,seed=318)
train=lgb.Dataset(X[~dev],label=y[~dev],feature_name=names); valid=lgb.Dataset(X[dev],label=y[dev],reference=train)
model=lgb.train({'objective':'regression','learning_rate':.03,'num_leaves':15,'min_data_in_leaf':150,'lambda_l2':10.,'verbosity':-1,'num_threads':2,'seed':918,'deterministic':True,'force_col_wise':True},train,num_boost_round=1500,valid_sets=[valid],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(100)])
model.save_model(str(out/'utility.lgb')); del X,y,g,train,valid; gc.collect()
V=pd.read_parquet(ROOT/'artifacts/pairs_val.parquet',columns=fields); V['p']=np.load(ROOT/'artifacts/val_scores_v1.npy'); VA=pd.read_parquet(ROOT/'artifacts/anchors_val.parquet'); n=len(VA)
X,_,_,d,positions,base=table(V,VA); pred=model.predict(X,num_threads=2); del X,V
p=d.p.to_numpy(); anchor=d.anchor.to_numpy(); code=d.target_code.to_numpy(); label=d.label.to_numpy(); nt=VA.n_true.to_numpy()
def score(keep):
    keep=one_owner(keep,p,code,anchor); tp=np.bincount(anchor[keep],weights=label[keep],minlength=n); fp=np.bincount(anchor[keep],weights=1-label[keep],minlength=n)
    return entity_f05_counts(tp,fp,nt)
baseline=score(p>=.8); tune=np.zeros(n,bool); tune[np.random.default_rng(918).permutation(n)[:n//2]]=True
report={'baseline':float(baseline.mean()),'n_train_dev_anchors':len(dev_ids),'results':[]}; best=None
for margin in [0.,.001,.003,.01,.02,.05,.1,2.]:
    keep=p>=.8; pos=0; changes=0
    for (aid,start,end,count),bk in zip(positions,base):
        q=pred[pos:pos+count]; pos+=count; k=int(q.argmax())
        if q[k]>q[bk]+margin:
            keep[start:end]=False; keep[start:start+k]=True; changes+=1
    s=score(keep); r={'margin':margin,'tune':float(s[tune].mean()),'holdout':float(s[~tune].mean()),'delta_holdout':float((s-baseline)[~tune].mean()),'all':float(s.mean()),'changed_anchors':changes}; report['results'].append(r); log(r)
    if best is None or r['tune']>best[0]: best=(r['tune'],margin,s)
_,margin,s=best; delta=(s-baseline)[~tune]; rng=np.random.default_rng(7); means=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)]
report['selected']={'margin':margin,'holdout_delta':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}; (out/'report.json').write_text(json.dumps(report,indent=2)); log(report['selected'])
