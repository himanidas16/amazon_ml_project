"""Train a complementary matcher on existing pair features, without changing v1.

Every positive and hard negative is kept; background negatives have inverse
sampling weights. Context is computed on complete anchor groups first.
Thresholds are tuned on one half of validation and evaluated on the other.
"""
import argparse, gc, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/"code/business_entity_resolution/src"))
from business_er.contextual import extend_features, CONTEXT_FEATURE_NAMES, CONTEXT_VERSION
from business_er.train import DEFAULT_PARAMS, dev_split
from business_er.metrics import entity_f05_counts
from business_er.evaluate import one_owner
ap=argparse.ArgumentParser(); ap.add_argument("--out",default="artifacts/contextual_v1"); ap.add_argument("--threads",type=int,default=3); ap.add_argument("--pair-only",action="store_true"); ap.add_argument("--rounds",type=int,default=2200); ap.add_argument("--extra-positives")
args=ap.parse_args(); feature_names=list(CONTEXT_FEATURE_NAMES[:37] if args.pair_only else CONTEXT_FEATURE_NAMES)

def make_features(df):
    return df[feature_names].to_numpy(dtype=np.float32) if args.pair_only else extend_features(df)

out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
t0=time.time()
def log(s): print(f"[{time.time()-t0:.1f}s] {s}",flush=True)
def groups(path):
    carry=None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100000):
        df=batch.to_pandas()
        if carry is not None: df=pd.concat([carry,df],ignore_index=True)
        last=df.anchor.iloc[-1]; carry=df[df.anchor==last]
        yield df[df.anchor!=last].reset_index(drop=True)
    if carry is not None: yield carry.reset_index(drop=True)
rng=np.random.default_rng(826); parts=[]; ys=[]; anchors=[]; weights=[]
for df in groups(ROOT/"artifacts/pairs_train.parquet"):
    X=make_features(df); y=df.label.to_numpy(np.int8)
    hard=(df.combo_rank.to_numpy()<5)|(df.ret_rank.to_numpy()<2)|((df.name_tset.to_numpy()>.8)&(df.addr_tset.to_numpy()>.7))
    easy=(y==0)&~hard; keep=~easy|(rng.random(len(y))<.1)
    parts.append(X[keep]); ys.append(y[keep]); anchors.append(df.anchor.to_numpy(np.int32)[keep]); weights.append(np.where(easy[keep],10.,1.).astype(np.float32))
    del X,df
if args.extra_positives:
    if not args.pair_only: raise ValueError("extra positives have no valid retrieval context; use --pair-only")
    extra=pd.read_parquet(ROOT/args.extra_positives)
    if not (extra.label==1).all(): raise ValueError("expected training positives only")
    known=pd.read_parquet(ROOT/"artifacts/anchors_train.parquet").anchor.to_numpy()
    if not extra.anchor.isin(known).all(): raise ValueError("extra positives must belong to training anchors")
    parts.append(make_features(extra)); ys.append(extra.label.to_numpy(np.int8)); anchors.append(extra.anchor.to_numpy(np.int32)); weights.append(np.ones(len(extra),np.float32)); del extra
X=np.concatenate(parts); del parts
y=np.concatenate(ys); anchor=np.concatenate(anchors); w=np.concatenate(weights); del ys,anchors,weights; gc.collect()
log(f"training on {X.shape}, positive rate {y.mean():.4f}")
dev=dev_split(anchor)
params={**DEFAULT_PARAMS,"num_threads":args.threads,"num_leaves":63,"min_data_in_leaf":100}
dtrain=lgb.Dataset(X[~dev],label=y[~dev],weight=w[~dev],feature_name=feature_names)
ddev=lgb.Dataset(X[dev],label=y[dev],weight=w[dev],reference=dtrain)
dtrain.construct(); ddev.construct(); del X,y,anchor,w,dev; gc.collect()
model=lgb.train(params,dtrain,num_boost_round=args.rounds,valid_sets=[ddev],callbacks=[lgb.early_stopping(120),lgb.log_evaluation(100)])
model.save_model(str(out/"matcher.lgb")); del dtrain,ddev; gc.collect()
log(f"saved iteration {model.best_iteration}")
p=[]
for df in groups(ROOT/"artifacts/pairs_val.parquet"):
    p.append(model.predict(make_features(df),num_threads=args.threads))
p=np.concatenate(p).astype(np.float32); np.save(out/"val_scores.npy",p)
D=pd.read_parquet(ROOT/"artifacts/pairs_val.parquet",columns=["anchor","label","target_code"])
A=pd.read_parquet(ROOT/"artifacts/anchors_val.parquet"); a=D.anchor.to_numpy(); y=D.label.to_numpy(); nt=A.n_true.to_numpy(); codes=D.target_code.to_numpy(); n=len(A)
p0=np.load(ROOT/"artifacts/val_scores_v1.npy")
def scores(prob,t):
    keep=one_owner(prob>=t,prob,codes,a)
    tp=np.bincount(a[keep],weights=y[keep],minlength=n); fp=np.bincount(a[keep],weights=1-y[keep],minlength=n)
    return entity_f05_counts(tp,fp,nt)
fold=np.random.default_rng(918).permutation(n); tune=np.zeros(n,bool); tune[fold[:n//2]]=True
baseline=scores(p0,.8); report={"baseline":float(baseline.mean()),"model_iteration":model.best_iteration,"context_version":CONTEXT_VERSION,"features":feature_names,"results":[]}
best=None
for alpha in [0.,.25,.5,.75,1.]:
    prob=alpha*p+(1-alpha)*p0
    choices=[(float(scores(prob,float(t))[tune].mean()),float(t)) for t in np.arange(.5,.951,.01)]
    tune_score,t=max(choices); s=scores(prob,t)
    result={"new_model_weight":alpha,"threshold":t,"tune":tune_score,"holdout":float(s[~tune].mean()),"overall":float(s.mean()),"holdout_delta":float((s-baseline)[~tune].mean()),"countries":{c:float(s[(A.country==c).to_numpy()].mean()) for c in A.country.unique()}}
    report["results"].append(result); log(result)
    if best is None or tune_score>best[0]: best=(tune_score,alpha,t,s)
_,alpha,t,s=best
# Bootstrap businesses for the selected policy, selection only used tuning half.
delta=(s-baseline)[~tune]; rng=np.random.default_rng(7); means=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)])
report["selected"]={"new_model_weight":alpha,"threshold":t,"holdout_delta":float(delta.mean()),"delta_ci95":np.quantile(means,[.025,.975]).tolist()}
(out/"report.json").write_text(json.dumps(report,indent=2)); log(report["selected"])
