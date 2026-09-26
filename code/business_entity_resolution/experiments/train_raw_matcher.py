"""Train a retrieval-independent matcher for new candidate channels.

Uses ONLY supplied training-fold pairs. Explicitly excludes retrieval scores,
ranks, block frequencies, and candidate context: fabricated zero retrieval
metadata must not influence predictions on a new candidate distribution.
"""
import json
import sys
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
from business_er.features import FEATURE_NAMES
from business_er.train import Matcher, dev_split
from business_er.io import file_sha256


def main():
    out=ROOT/'artifacts/raw_matcher_50k'
    out.mkdir(parents=True,exist_ok=True)
    if (out/'matcher.lgb').exists():
        raise RuntimeError('refusing to overwrite a trained model')
    pairs=ROOT/'artifacts/pairs_sel'
    anchors=pd.read_parquet(pairs/'anchors_train.parquet')
    chosen=np.sort(np.random.default_rng(2801).choice(anchors.anchor.to_numpy(),min(50000,len(anchors)),replace=False))
    names=list(FEATURE_NAMES[:FEATURE_NAMES.index('ret_score')])
    assert not any(n.startswith(('ret_','ch_','avail_')) for n in names)
    parts=[]
    for f in sorted(pairs.glob('train_*.parquet')):
        for b in pq.ParquetFile(f).iter_batches(columns=['anchor','label','weight',*names],batch_size=50000):
            d=b.to_pandas()
            d=d[d.anchor.isin(chosen)]
            if len(d): parts.append(d)
        print(f'loaded {f.name}',flush=True)
    d=pd.concat(parts,ignore_index=True); del parts
    a=d.pop('anchor').to_numpy(); y=d.pop('label').to_numpy(); w=d.pop('weight').to_numpy()
    X=d[names].to_numpy(np.float32); del d
    dev=dev_split(a)
    params={'objective':'binary','learning_rate':.05,'num_leaves':63,'min_data_in_leaf':100,
            'feature_fraction':.9,'bagging_fraction':.8,'bagging_freq':1,'lambda_l2':1.,
            'max_bin':127,'num_threads':3,'seed':2801,'deterministic':True,'force_col_wise':True,'verbosity':-1}
    full=lgb.Dataset(X,label=y,weight=w,feature_name=names,params={'max_bin':127,'num_threads':3},free_raw_data=True)
    train=full.subset(np.flatnonzero(~dev)); valid=full.subset(np.flatnonzero(dev))
    model=lgb.train(params,train,num_boost_round=1600,valid_sets=[valid],
                    callbacks=[lgb.early_stopping(100),lgb.log_evaluation(200)])
    meta={'purpose':'retrieval-independent rescue experiment','train_businesses':len(chosen),'n_pairs':len(a),
          'params':params,'best_iteration':model.best_iteration,'source_build_sha256':file_sha256(pairs/'build_report.json')}
    Matcher(model,names,meta).save(out/'matcher.lgb')
    print(json.dumps(meta,indent=2),flush=True)

if __name__=='__main__': main()
