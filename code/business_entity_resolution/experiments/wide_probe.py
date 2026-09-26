"""Full-pool wider retrieval, cached features, and held-out novel-edge rescue.

Never inject ground truth into candidates. Keep the v1 decisions as baseline;
only consider additional retrieved edges using an independently trained matcher.
"""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
from business_er.retrieve import TokenFreq, build_index, compute_keys, generate_candidates, SOURCE_BASE
from business_er.io import iter_source
from business_er.features import prepare, compute_features, FEATURE_NAMES
from business_er.predict import _prepare_targets
from business_er.evaluate import one_owner
from business_er.metrics import entity_f05_counts
ap=argparse.ArgumentParser(); ap.add_argument('--country',default='India'); ap.add_argument('--anchors',type=int,default=4000)
ap.add_argument('--k',type=int,default=150); ap.add_argument('--workers',type=int,default=2)
ap.add_argument('--disk-index',action='store_true'); ap.add_argument('--fresh',action='store_true'); ap.add_argument('--locked-threshold',type=float)
ap.add_argument('--model',default='artifacts/pair_only_v1/matcher.lgb'); ap.add_argument('--build-only',action='store_true')
args=ap.parse_args(); suffix='_fresh' if args.fresh else ''; out=ROOT/f'artifacts/wide_{args.country}_{args.anchors}_k{args.k}{suffix}'; out.mkdir(parents=True,exist_ok=True)
t0=time.time()
def log(s): print(f'[{time.time()-t0:.1f}s] {s}',flush=True)
D=ROOT/'amazon_ml_dataset/student_resource/dataset/train'; paths={s:D/f'train_source{s}.tsv' for s in (2,3)}
A0=pd.read_parquet(ROOT/'artifacts/anchors_val.parquet'); sel=np.flatnonzero((A0.country==args.country).to_numpy())
if len(sel)>args.anchors: sel=np.sort(np.random.default_rng(611).choice(sel,args.anchors,replace=False))
A=A0.iloc[sel].reset_index(drop=True)
if args.fresh:
    if args.locked_threshold is None: raise ValueError("fresh evaluation requires a locked threshold")
    from business_er.splits import load_split
    sp=load_split(ROOT/'artifacts/split_5fold.npz'); held=sp.s1_values[sp.s1_fold==0]; del sp
    eligible=[]
    for df in iter_source(D/'train_source1.tsv',chunksize=100000):
        values=df.entity_id.str.slice(3).astype(np.int64).to_numpy()
        mask=(df.country==args.country).to_numpy()&np.isin(values,held)&~np.isin(values,A0.s1_value.to_numpy())
        eligible.extend(values[mask].tolist())
    chosen=np.sort(np.random.default_rng(2718).choice(eligible,args.anchors,replace=False))
    A=pd.DataFrame({'s1_value':chosen,'country':args.country,'n_true':0}); del held,eligible
n=len(A); nt=A.n_true.to_numpy()
lookup=dict(zip(A.s1_value.astype(int),range(n))); truthkeys=[]
with open(D/'train_ground_truth.tsv') as f:
    next(f)
    for line in f:
        sid,ids=line.rstrip('\n').split('\t'); idx=lookup.get(int(sid[3:]))
        if idx is not None and ids:
            truthkeys.extend(idx*10**10+int(t[1])*SOURCE_BASE+int(t[3:]) for t in ids.split(','))
truthkeys=np.asarray(truthkeys,np.int64)
if args.fresh: A['n_true']=np.bincount(truthkeys//10**10,minlength=n)
nt=A.n_true.to_numpy(); A.to_parquet(out/'anchors.parquet',index=False)
# Pin caches to actual source code and frequency table; do not silently reuse stale features.
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
manifest={'country':args.country,'anchors':A.s1_value.tolist(),'k':args.k,
          'code':{name:sha(ROOT/'code/business_entity_resolution/src/business_er'/name) for name in ('features.py','normalize.py','retrieve.py','io.py','predict.py')},
          'frequency':sha(ROOT/'artifacts/token_freq_excl_fold0.npz')}
mp=out/'manifest.json'
if mp.exists():
    previous=json.loads(mp.read_text())
    previous['code']={name:previous['code'].get(name) for name in manifest['code']}
    if previous!=manifest: raise RuntimeError('wide-probe cache differs; choose a new output configuration')
mp.write_text(json.dumps(manifest))
if not (out/'features_done.json').exists():
    freq=TokenFreq.load(ROOT/'artifacts/token_freq_excl_fold0.npz')
    frames=[]
    for df in iter_source(D/'train_source1.tsv',chunksize=100000):
        values=df.entity_id.str.slice(3).astype(np.int64)
        m=values.isin(A.s1_value)
        if m.any():
            part=df[m].copy(); part['s1_value']=values[m]; frames.append(part)
    text=pd.concat(frames).set_index('s1_value').loc[A.s1_value].reset_index(); del frames
    cp=out/'candidates.npz'
    if cp.exists():
        z=np.load(cp); c={k:z[k] for k in z.files}
    else:
        if args.disk_index:
            from business_er.disk_index import build_disk_index
            index=build_disk_index(paths,args.country,freq,out/'index',log=log)
        else:
            index=build_index(paths,country=args.country,freq=freq,workers=args.workers,read_chunk=100000,log=log)
        log(f'index ready: {index.n_targets} targets')
        ak=compute_keys(text.business_name.tolist(),text.business_address.tolist(),text.country.tolist(),freq=freq,workers=args.workers)
        cand=generate_candidates(index,ak,n,cap=3000,k=args.k,batch=30,workers=args.workers)
        c={'anchor':cand.anchor,'code':index.codes[cand.target],'score':cand.score,'rank':cand.rank,'channels':cand.channels}
        np.savez(cp,**c); del index,ak,cand; gc.collect()
    log(f'{len(c["anchor"])} candidates')
    need=np.unique(c['code']); tcodes,T=_prepare_targets(paths,args.country,need,args.workers); order=np.argsort(tcodes)
    target_pos=order[np.searchsorted(tcodes[order],c['code'])]
    prepared=prepare(text.business_name.tolist(),text.business_address.tolist(),countries=text.country.tolist(),workers=args.workers)
    label=np.isin(c['anchor'].astype(np.int64)*10**10+c['code'],truthkeys).astype(np.int8)
    bounds=np.searchsorted(c['anchor'],np.arange(0,n+500,500)); count=0
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        if lo==hi: continue
        sl=slice(lo,hi); meta={k:c[k][sl] for k in ('anchor','score','rank','channels')}; meta['source']=(c['code'][sl]//SOURCE_BASE).astype(np.int8)
        X=compute_features(prepared,T,c['anchor'][sl],target_pos[sl],[args.country]*(hi-lo),meta,freq=freq,workers=args.workers)
        frame=pd.DataFrame(X,columns=list(FEATURE_NAMES)); frame.insert(0,'anchor',c['anchor'][sl]); frame.insert(1,'target_code',c['code'][sl]); frame.insert(2,'label',label[sl])
        frame.to_parquet(out/f'features_{count:03d}.parquet',index=False); count+=1; del X,frame; log(f'features {hi}/{len(label)}')
    (out/'features_done.json').write_text(json.dumps({'parts':count})); del prepared,T,c,text; gc.collect()
if args.build_only: sys.exit(0)
model=lgb.Booster(model_file=str(ROOT/args.model)); names=model.feature_name(); parts=[]; pp=[]
for f in sorted(out.glob('features_*.parquet')):
    frame=pd.read_parquet(f); pp.append(model.predict(frame[names].to_numpy(np.float32),num_threads=args.workers)); parts.append(frame[['anchor','target_code','label']])
W=pd.concat(parts,ignore_index=True); prob=np.concatenate(pp); del parts,pp
if args.fresh:
    from business_er.contextual import refresh_context
    from business_er.train import Matcher
    baseline_model=Matcher.load(ROOT/'artifacts/models/matcher_v1.lgb'); baselines=[]
    for f in sorted(out.glob('features_*.parquet')):
        frame=pd.read_parquet(f); frame=refresh_context(frame[frame.ret_rank<25].reset_index(drop=True))
        frame['p']=baseline_model.predict(frame[list(FEATURE_NAMES)].to_numpy(np.float32),num_threads=args.workers)
        baselines.append(frame[['anchor','target_code','label','p']])
    base=pd.concat(baselines,ignore_index=True); bp=base.p.to_numpy(); del baselines,baseline_model
else:
    base=pd.read_parquet(ROOT/'artifacts/pairs_val.parquet',columns=['anchor','target_code','label']); bp=np.load(ROOT/'artifacts/val_scores_v1.npy'); m=base.anchor.isin(sel).to_numpy(); base=base[m].copy(); bp=bp[m]; base['anchor']=np.searchsorted(sel,base.anchor.to_numpy()); base['p']=bp
expected_base=np.isin(base.anchor.to_numpy(np.int64)*10**10+base.target_code.to_numpy(),truthkeys).astype(np.int8)
if not np.array_equal(expected_base,base.label.to_numpy()): raise RuntimeError('baseline ID alignment or truth labels differ')
widekeys=W.anchor.to_numpy(np.int64)*10**10+W.target_code.to_numpy(); basekeys=base.anchor.to_numpy(np.int64)*10**10+base.target_code.to_numpy(); novel=~np.isin(widekeys,basekeys)
if not np.isin(basekeys,widekeys).all(): raise RuntimeError('wide retrieval lost a baseline candidate')
extra=W[novel]; ep=prob[novel]
a=np.r_[base.anchor.to_numpy(),extra.anchor.to_numpy()]; code=np.r_[base.target_code.to_numpy(),extra.target_code.to_numpy()]; label=np.r_[base.label.to_numpy(),extra.label.to_numpy()]
def scores(t):
    p=np.r_[bp,np.where(ep>=t,ep,0.)]; keep=one_owner(p>=.8,p,code,a)
    tp=np.bincount(a[keep],weights=label[keep],minlength=n); fp=np.bincount(a[keep],weights=1-label[keep],minlength=n)
    return entity_f05_counts(tp,fp,nt), int(((ep>=t)&(extra.label.to_numpy()==1)).sum()),int(((ep>=t)&(extra.label.to_numpy()==0)).sum())
baseline=scores(2.)[0]; tune=np.zeros(n,bool); tune[np.random.default_rng(918).permutation(n)[:n//2]]=True
found=np.bincount(W.anchor,weights=W.label,minlength=n)
report={'model':args.model,'model_sha256':sha(ROOT/args.model),'country':args.country,'n':n,'baseline':float(baseline.mean()),'wide_ceiling':float(entity_f05_counts(found,np.zeros(n),nt).mean()),'novel_true_pairs':int(extra.label.sum()),'results':[]}
# Isolate the ranker's value at an unchanged final candidate budget.
wa=W.anchor.to_numpy(); src=W.target_code.to_numpy()//SOURCE_BASE
ranking=np.lexsort((-prob,src,wa)); group=wa[ranking].astype(np.int64)*4+src[ranking]
starts=np.flatnonzero(np.r_[True,group[1:]!=group[:-1]])
rank=np.empty(len(W),np.int64); rank[ranking]=np.arange(len(W))-np.repeat(starts,np.diff(np.r_[starts,len(W)]))
report['reranked_oracles']={}
for budget in [25,50,75]:
    m=rank<budget; hit=np.bincount(wa[m],weights=W.label.to_numpy()[m],minlength=n)
    report['reranked_oracles'][str(budget)]=float(entity_f05_counts(hit,np.zeros(n),nt).mean())
log({'reranked_oracles':report['reranked_oracles']})

best=None
thresholds=[args.locked_threshold] if args.locked_threshold is not None else [.8,.9,.95,.97,.98,.99,.995,.999,2.]
for t in thresholds:
    s,tp,fp=scores(t); r={'threshold':t,'tune':float(s[tune].mean()),'holdout':float(s[~tune].mean()),'delta_holdout':float((s-baseline)[~tune].mean()),'all':float(s.mean()),'new_tp':tp,'new_fp':fp}; report['results'].append(r); log(r)
    if best is None or r['tune']>best[0]: best=(r['tune'],t,s)
_,t,s=best; eval_mask=np.ones(n,bool) if args.locked_threshold is not None else ~tune; delta=(s-baseline)[eval_mask]; rng=np.random.default_rng(7); means=[rng.choice(delta,len(delta),replace=True).mean() for _ in range(1000)]
report['selected']={'threshold':t,'fresh_anchors':args.fresh,'evaluation_businesses':int(eval_mask.sum()),'holdout_delta':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}
(out/'report.json').write_text(json.dumps(report,indent=2)); tag=Path(args.model).parent.name; (out/f'report_{tag}.json').write_text(json.dumps(report,indent=2)); np.save(out/f'scores_{tag}.npy',prob); log(report['selected'])
