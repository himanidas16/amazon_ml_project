"""Reproducible, streaming audit of a saved candidate set and matcher.

Labels are used only to evaluate frozen predictions. Writes all missed true
edges and false predictions for offline diagnosis, never into candidate data.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/business_entity_resolution/src'))
from business_er.evaluate import one_owner
from business_er.io import file_sha256, iter_source
from business_er.metrics import entity_f05_counts
from business_er.retrieve import SOURCE_BASE
from business_er.train import Matcher


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pairs', default='artifacts/pairs_sel')
    ap.add_argument('--model', default='artifacts/models/matcher_sel_300k.lgb')
    ap.add_argument('--out', default='artifacts/error_audit_sel')
    ap.add_argument('--threshold', type=float, default=0.75)
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pairs = ROOT / args.pairs
    model = Matcher.load(ROOT / args.model)
    anchors = pd.read_parquet(pairs / 'anchors_val.parquet').sort_values('anchor').reset_index(drop=True)
    assert np.array_equal(anchors.anchor, np.arange(len(anchors)))
    frames = []
    for path in sorted(pairs.glob('val_*.parquet')):
        for b in pq.ParquetFile(path).iter_batches(batch_size=25000):
            d = b.to_pandas()
            p = model.predict(d[list(model.feature_names)].to_numpy(np.float32), model.feature_names, num_threads=2)
            m = d[['anchor', 'target_code', 'label']].copy()
            m['p'] = p.astype(np.float32)
            frames.append(m)
        print(f'scored {path.name}', flush=True)
    d = pd.concat(frames, ignore_index=True).sort_values(['anchor','target_code']).reset_index(drop=True)
    del frames
    a, c, y, p = [d[k].to_numpy() for k in ['anchor','target_code','label','p']]
    keep = one_owner(p >= args.threshold, p, c, a)
    n = len(anchors)
    nt = anchors.n_true.to_numpy()
    found = np.bincount(a, weights=y, minlength=n)
    tp = np.bincount(a[keep], weights=y[keep], minlength=n)
    fp = np.bincount(a[keep], weights=1-y[keep], minlength=n)
    metric = entity_f05_counts(tp, fp, nt)
    oracle = entity_f05_counts(found, np.zeros(n), nt)
    d['keep'] = keep
    d.to_parquet(out / 'predictions.parquet', index=False)
    anchors['score'] = metric
    anchors['oracle'] = oracle
    anchors['tp'] = tp.astype(int)
    anchors['fp'] = fp.astype(int)
    anchors['retrieved_true'] = found.astype(int)
    anchors.to_parquet(out / 'anchors.parquet', index=False)
    lookup = dict(zip(anchors.s1_value.astype(int), range(n)))
    true_edges = []
    data = ROOT / 'amazon_ml_dataset/student_resource/dataset/train'
    with (data / 'train_ground_truth.tsv').open() as f:
        next(f)
        for line in f:
            sid, ids = line.rstrip('\n').split('\t')
            idx = lookup.get(int(sid[3:]))
            if idx is not None:
                true_edges.extend((idx, int(t[1])*SOURCE_BASE + int(t[3:])) for t in ids.split(',') if t)
    truth = pd.DataFrame(true_edges, columns=['anchor','target_code'])
    assert np.array_equal(np.bincount(truth.anchor, minlength=n), nt)
    missed = truth.merge(d[['anchor','target_code','p','keep']], on=['anchor','target_code'], how='left', validate='one_to_one')
    missed = missed[~missed.keep.fillna(False).astype(bool)].copy()
    missed['reason'] = np.where(missed.p.isna(), 'not_retrieved', np.where(missed.p < args.threshold, 'below_threshold', 'lost_ownership'))
    false = d.loc[keep & (y == 0), ['anchor','target_code','p']].copy()
    false['reason'] = 'false_positive'
    errors = pd.concat([missed.drop(columns='keep'), false], ignore_index=True)
    errors['s1_value'] = anchors.s1_value.to_numpy()[errors.anchor]
    errors['country'] = anchors.country.to_numpy()[errors.anchor]
    # Attach only records actually involved in errors; bounded TSV scans.
    for source in (1,2,3):
        ids = (np.unique(errors.s1_value) if source == 1 else
               np.unique(errors.loc[errors.target_code // SOURCE_BASE == source, 'target_code'] % SOURCE_BASE))
        text = []
        for chunk in iter_source(data / f'train_source{source}.tsv', chunksize=100000):
            vals = chunk.entity_id.str.slice(3).astype(np.int64)
            mask = vals.isin(ids)
            if mask.any():
                z = chunk.loc[mask, ['business_name','business_address']].copy()
                z['key'] = vals[mask].to_numpy() + (0 if source == 1 else source*SOURCE_BASE)
                text.append(z)
        table = pd.concat(text, ignore_index=True) if text else pd.DataFrame(columns=['key','business_name','business_address'])
        if source == 1:
            errors = errors.merge(table.rename(columns={'key':'s1_value','business_name':'anchor_name','business_address':'anchor_address'}), on='s1_value', how='left', validate='many_to_one')
        else:
            table = table.set_index('key')
            m = errors.target_code // SOURCE_BASE == source
            errors.loc[m,'target_name'] = errors.loc[m,'target_code'].map(table.business_name)
            errors.loc[m,'target_address'] = errors.loc[m,'target_code'].map(table.business_address)
        print(f'attached S{source} error text', flush=True)
    errors.to_parquet(out / 'errors.parquet', index=False)
    report = {'threshold': args.threshold, 'macro_f05': float(metric.mean()), 'candidate_ceiling': float(oracle.mean()),
              'n_anchors': n, 'errors': errors.reason.value_counts().to_dict(),
              'per_country': {ct: {'score':float(metric[anchors.country == ct].mean()), 'ceiling':float(oracle[anchors.country == ct].mean())} for ct in sorted(anchors.country.unique())},
              'model_sha256':file_sha256(ROOT / args.model),
              'inputs': {f.name:file_sha256(f) for f in sorted(pairs.glob('val_*.parquet'))},
              'note':'Development audit of repeatedly used validation, not a fresh generalization estimate.'}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
