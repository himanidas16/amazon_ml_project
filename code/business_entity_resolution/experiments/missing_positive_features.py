"""Training-only augmentation: expose genuine positives missed by v1 retrieval.

These pairs are never inserted into validation/test candidate sets. The extra
features are intended for the pair-only matcher, which ignores retrieval ranks.
"""
import sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
from business_er.retrieve import SOURCE_BASE, TokenFreq, target_code
from business_er.features import prepare,compute_features,FEATURE_NAMES
from business_er.io import iter_source
D=ROOT/'amazon_ml_dataset/student_resource/dataset/train'; out=ROOT/'artifacts/missing_positive_train.parquet'; t0=time.time()
def log(s): print(f'[{time.time()-t0:.1f}s] {s}',flush=True)
A=pd.read_parquet(ROOT/'artifacts/anchors_train.parquet'); lookup=dict(zip(A.s1_value.astype(int),A.anchor.astype(int)))
pairs=pd.read_parquet(ROOT/'artifacts/pairs_train.parquet',columns=['anchor','target_code','label']); pairs=pairs[pairs.label==1]
existing=pairs.anchor.to_numpy(np.int64)*10**10+pairs.target_code.to_numpy(); del pairs
anc=[]; codes=[]
with open(D/'train_ground_truth.tsv') as f:
    next(f)
    for line in f:
        sid,ids=line.rstrip('\n').split('\t'); idx=lookup.get(int(sid[3:]))
        if idx is not None and ids:
            for tok in ids.split(','): anc.append(idx); codes.append(int(tok[1])*SOURCE_BASE+int(tok[3:]))
anc=np.asarray(anc,np.int64); codes=np.asarray(codes,np.int64); missing=~np.isin(anc*10**10+codes,existing); anc=anc[missing]; codes=codes[missing]
order=np.argsort(anc,kind='stable'); anc=anc[order]; codes=codes[order]; del existing
log(f'{len(anc)} training positives absent from retrieved pairs')
need_a=A.s1_value.to_numpy()[np.unique(anc)]; need_t=np.unique(codes)
def texts(source,need):
    parts=[]
    for df in iter_source(D/f'train_source{source}.tsv',chunksize=100000):
        values=df.entity_id.str.slice(3).astype(np.int64).to_numpy(); keys=values if source==1 else target_code(source,values)
        mask=np.isin(keys,need)
        if mask.any():
            part=df.loc[mask,['business_name','business_address','country']].copy(); part['key']=keys[mask]; parts.append(part)
    return pd.concat(parts,ignore_index=True)
at=texts(1,need_a); tt=pd.concat([texts(s,need_t) for s in (2,3)],ignore_index=True)
aidx=pd.Index(at.key).get_indexer(A.s1_value.to_numpy()[anc]); tidx=pd.Index(tt.key).get_indexer(codes)
assert (aidx>=0).all() and (tidx>=0).all()
assert np.array_equal(at.country.to_numpy()[aidx],tt.country.to_numpy()[tidx])
PA=prepare(at.business_name.tolist(),at.business_address.tolist(),countries=at.country.tolist())
PT=prepare(tt.business_name.tolist(),tt.business_address.tolist(),countries=tt.country.tolist())
freq=TokenFreq.load(ROOT/'artifacts/token_freq_excl_fold0.npz')
meta={'anchor':anc,'score':np.zeros(len(anc)),'rank':np.full(len(anc),100),'channels':np.zeros(len(anc),np.uint16),'source':codes//SOURCE_BASE}
X=compute_features(PA,PT,aidx,tidx,at.country.to_numpy()[aidx],meta,freq=freq,workers=0)
frame=pd.DataFrame(X,columns=list(FEATURE_NAMES)); frame.insert(0,'anchor',anc); frame.insert(1,'target_code',codes); frame.insert(2,'label',np.ones(len(anc),np.int8)); frame.to_parquet(out,index=False); log(f'saved {out}')
