"""Evaluate adding only novel, unclaimed character-retrieved matches.

Freeze candidate files first. Use a retrieval-independent training-fold model;
select the rescue threshold on half the pilot businesses, then evaluate the
other half. Existing baseline decisions and claimed targets are preserved.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
from business_er.features import FEATURE_NAMES, prepare, compute_features
from business_er.io import iter_source, file_sha256
from business_er.predict import _prepare_targets
from business_er.retrieve import TokenFreq, SOURCE_BASE
from business_er.train import Matcher
from business_er.rescue import unclaimed_rescue
from business_er.metrics import entity_f05_counts


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probes', nargs='+', default=['artifacts/char_probe_name_1000','artifacts/char_probe_address_1000_fast'])
    ap.add_argument('--out',default='artifacts/char_rescue_1000')
    ap.add_argument('--locked-threshold',type=float,help='Evaluate a threshold fixed by an earlier experiment without retuning')
    args=ap.parse_args()
    out=ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    baseline=ROOT/'artifacts/error_audit_sel'
    modelpath=ROOT/'artifacts/raw_matcher_50k/matcher.lgb'
    model=Matcher.load(modelpath)
    assert not set(model.feature_names)&{'ret_score','ret_rank','combo_rank','avail_mean'}
    manifests=[json.loads((ROOT/p/'manifest.json').read_text()) for p in args.probes]
    vals=manifests[0]['anchors']
    assert all(m['anchors']==vals for m in manifests)
    a=pd.read_parquet(ROOT/args.probes[0]/'anchors.parquet')
    d=pd.read_parquet(baseline/'predictions.parquet')
    all_a=pd.read_parquet(baseline/'anchors.parquet')
    mapping=dict(zip(a.anchor.to_numpy(),range(len(a))))
    selected=d[d.anchor.isin(mapping)].copy()
    selected['anchor']=selected.anchor.map(mapping)
    oldkeys=selected.anchor.to_numpy(np.int64)*10**10+selected.target_code.to_numpy(np.int64)
    c=pd.concat([pd.read_parquet(ROOT/p/'candidates.parquet')[['anchor','target_code']] for p in args.probes]).drop_duplicates().sort_values(['anchor','target_code']).reset_index(drop=True)
    keys=c.anchor.to_numpy(np.int64)*10**10+c.target_code.to_numpy(np.int64)
    # Already claimed targets cannot be reassigned by this conservative rescue.
    claimed=np.unique(d.loc[d.keep,'target_code'].to_numpy())
    c=c[~np.isin(keys,oldkeys)&~c.target_code.isin(claimed)].copy().reset_index(drop=True)
    del d,selected
    data=ROOT/'amazon_ml_dataset/student_resource/dataset/train'
    text=[]
    for chunk in iter_source(data/'train_source1.tsv',chunksize=100000):
        chunk['s1_value']=chunk.entity_id.str.slice(3).astype(np.int64)
        m=chunk.s1_value.isin(vals)
        if m.any(): text.append(chunk[m])
    text=pd.concat(text).set_index('s1_value').loc[vals].reset_index()
    freq=TokenFreq.load(ROOT/'artifacts/token_freq_excl_fold0.npz')
    c['p']=0.
    for country in sorted(text.country.unique()):
        mask=(text.country.to_numpy()[c.anchor.to_numpy()]==country)
        ci=np.flatnonzero(mask)
        if not len(ci): continue
        sub=c.iloc[ci]
        codes,T=_prepare_targets({s:data/f'train_source{s}.tsv' for s in (2,3)},country,np.unique(sub.target_code),2)
        order=np.argsort(codes); tp=order[np.searchsorted(codes[order],sub.target_code)]
        A=prepare(text.business_name.tolist(),text.business_address.tolist(),countries=text.country.tolist(),workers=0)
        n=len(sub); anchor=sub.anchor.to_numpy(np.int32)
        # Context is intentionally unused by the raw matcher. All input feature
        # names are checked explicitly rather than assuming an old column slice.
        meta={'anchor':anchor,'score':np.zeros(n,np.float32),'rank':np.zeros(n,np.int16),
              'channels':np.zeros(n,np.uint16),'source':(sub.target_code.to_numpy()//SOURCE_BASE).astype(np.int8)}
        X=compute_features(A,T,anchor,tp,[country]*n,meta,freq=freq,workers=2)
        pos=[FEATURE_NAMES.index(f) for f in model.feature_names]
        c.loc[ci,'p']=model.predict(X[:,pos],model.feature_names,num_threads=2)
        print(f'{country}: scored {n:,} novel, unclaimed candidates',flush=True)
    # Labels enter only after candidate generation and scoring have finished.
    errors=pd.read_parquet(baseline/'errors.parquet')
    errors=errors[(errors.reason=='not_retrieved')&errors.anchor.isin(mapping)].copy()
    truekeys=errors.anchor.map(mapping).to_numpy(np.int64)*10**10+errors.target_code.to_numpy(np.int64)
    c['label']=np.isin(c.anchor.to_numpy(np.int64)*10**10+c.target_code.to_numpy(np.int64),truekeys)
    c.to_parquet(out/'scored_novel.parquet',index=False)
    base=all_a.set_index('anchor').loc[a.anchor]
    nt=base.n_true.to_numpy(); n=len(base)
    bs=base.score.to_numpy(); tune=np.random.default_rng(419).random(n)<.5
    anc=c.anchor.to_numpy(np.int64); target=c.target_code.to_numpy(np.int64)
    p=c.p.to_numpy(); y=c.label.to_numpy()
    trials=[]; vectors=[]
    thresholds = [args.locked_threshold] if args.locked_threshold is not None else [.8,.9,.95,.97,.99,.995,1.01]
    for t in thresholds:
        keep=unclaimed_rescue(claimed,anc,target,p,threshold=t)
        tp=base.tp.to_numpy()+np.bincount(anc[keep],weights=y[keep],minlength=n)
        fp=base.fp.to_numpy()+np.bincount(anc[keep],weights=1-y[keep],minlength=n)
        s=entity_f05_counts(tp,fp,nt)
        trials.append({'threshold':t,'tune':float(s[tune].mean())}); vectors.append(s)
    # On an exact tie prefer no rescue / the higher threshold.
    best=max(range(len(trials)),key=lambda i:(trials[i]['tune'],trials[i]['threshold']))
    delta=(vectors[best]-bs)[~tune]
    rng=np.random.default_rng(123)
    boot=[delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)]
    report={'selected_on_tune':trials[best], 'baseline_holdout':float(bs[~tune].mean()),
            'rescued_holdout':float(vectors[best][~tune].mean()), 'delta':float(delta.mean()),
            'delta_ci95':np.quantile(boot,[.025,.975]).tolist(), 'tune_trials':trials,
            'novel_unclaimed_pairs':len(c),'novel_unclaimed_true_pairs':int(y.sum()),
            'model_sha256':file_sha256(modelpath),'probes':args.probes,
            'locked_threshold':args.locked_threshold,
            'frequency_sha256':file_sha256(ROOT/'artifacts/token_freq_excl_fold0.npz'),
            'note':'Small development pilot, not untouched validation or a public score. Existing baseline matches preserved; production promotion requires independent confirmation.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
