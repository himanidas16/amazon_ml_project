"""Measure complementary character retrieval against the FULL target pool.

Random development anchors, no label-dependent query selection. Ground truth
is opened only AFTER candidates are frozen. Results are candidate ceilings,
not model scores. No production model or output is modified.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/business_entity_resolution/src'))
from business_er.char_retrieve import CHAR_VERSION, QueryGramIndex
from business_er.io import file_sha256, iter_source
from business_er.metrics import entity_f05_counts
from business_er.retrieve import SOURCE_BASE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--anchors', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=991)
    ap.add_argument('--cap', type=int, default=256)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--k', type=int, default=100)
    ap.add_argument('--fields', nargs='+', choices=['name','address'], default=['name'])
    ap.add_argument('--seed-predictions', help='Optional frozen baseline predictions; use confident kept matches as extra query views')
    ap.add_argument('--seed-threshold', type=float, default=.995)
    ap.add_argument('--pairs', default='artifacts/pairs_sel')
    ap.add_argument('--out', default='artifacts/char_probe_name_1000')
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    pairs = ROOT / args.pairs
    data = ROOT / 'amazon_ml_dataset/student_resource/dataset/train'
    t0 = time.time()
    def log(msg):
        print(f'[{time.time()-t0:.1f}s] {msg}', flush=True)
    base = pd.read_parquet(pairs / 'anchors_val.parquet').sort_values('anchor')
    pick = np.sort(np.random.default_rng(args.seed).choice(len(base), min(args.anchors,len(base)), replace=False))
    a = base.iloc[pick].reset_index(drop=True)
    mapping = dict(zip(a.anchor.astype(int),range(len(a))))
    vals = a.s1_value.to_numpy()
    manifest = {'version':CHAR_VERSION,'seed':args.seed,'cap':args.cap,'k':args.k,'fields':args.fields,
                'anchors':vals.tolist(), 'baseline':{f.name:file_sha256(f) for f in sorted(pairs.glob('val_*.parquet'))},
                'code':{f:file_sha256(ROOT / 'code/business_entity_resolution/src/business_er' / f) for f in ['char_retrieve.py','normalize.py','retrieve.py']},
                'data':{p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in sorted(data.glob('train_source*.tsv'))}}
    manifest['seed_predictions'] = file_sha256(ROOT / args.seed_predictions) if args.seed_predictions else None
    manifest['seed_threshold'] = args.seed_threshold
    mp = out / 'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise RuntimeError('cache configuration or input changed; use a fresh --out directory')
    mp.write_text(json.dumps(manifest, indent=2))
    cp = out / 'candidates.parquet'
    if not cp.exists():
        frames = []
        for chunk in iter_source(data / 'train_source1.tsv', chunksize=100000):
            chunk['s1_value'] = chunk.entity_id.str.slice(3).astype(np.int64)
            m = chunk.s1_value.isin(vals)
            if m.any():
                frames.append(chunk[m])
        text = pd.concat(frames).set_index('s1_value').loc[vals]
        query_text = text.reset_index()
        query_owner = np.arange(len(a),dtype=np.int32)
        if args.seed_predictions:
            # Read ONLY predictions and identity, never the label column.
            seeds = pd.read_parquet(ROOT / args.seed_predictions, columns=['anchor','target_code','p','keep'])
            seeds = seeds[seeds.anchor.isin(mapping) & seeds.keep & (seeds.p >= args.seed_threshold)].copy()
            seeds['owner'] = seeds.anchor.map(mapping)
            chunks = []
            for source in (2,3):
                need = seeds.loc[seeds.target_code // SOURCE_BASE == source, 'target_code']
                for block in iter_source(data / f'train_source{source}.tsv', chunksize=100000):
                    code = source*SOURCE_BASE + block.entity_id.str.slice(3).astype(np.int64)
                    mask = code.isin(need)
                    if mask.any():
                        z = block.loc[mask,['business_name','business_address','country']].copy()
                        z['target_code'] = code[mask].to_numpy()
                        chunks.append(z)
            if chunks:
                extra = seeds[['owner','target_code']].merge(pd.concat(chunks),on='target_code',validate='many_to_one')
                query_owner = np.r_[query_owner,extra.owner.to_numpy(np.int32)]
                query_text = pd.concat([query_text,extra],ignore_index=True)
            log(f'added {len(query_owner)-len(a):,} frozen high-confidence seed views')
        idx = QueryGramIndex(query_text.business_name.tolist(),query_text.business_address.tolist(),query_text.country.tolist(),cap=args.cap,fields=args.fields)
        log(f'{len(query_owner)} query views, {len(idx.counts):,} distinct character keys')
        def tasks(source):
            for chunk in iter_source(data / f'train_source{source}.tsv', chunksize=50000):
                codes = source*SOURCE_BASE + chunk.entity_id.str.slice(3).astype(np.int64).to_numpy()
                yield (codes,chunk.business_name.tolist(),chunk.business_address.tolist(),chunk.country.tolist())
        if args.workers > 1:
            from multiprocessing import get_context
            from business_er.char_retrieve import init_gram_worker, build_gram_partial, merge_partial
            with get_context('spawn').Pool(args.workers, initializer=init_gram_worker,
                 initargs=(query_text.business_name.tolist(),query_text.business_address.tolist(),query_text.country.tolist(),args.cap,args.fields)) as pool:
                for source in (2,3):
                    count = 0
                    for counts,postings,n_country in pool.imap(build_gram_partial,tasks(source),chunksize=1):
                        merge_partial(idx,counts,postings,n_country)
                        count += sum(n_country.values())
                        if count % 250000 == 0:
                            log(f'S{source}: {count:,} records; {len(idx.postings):,} live keys')
                    log(f'S{source} complete: {count:,} records')
        else:
            for source in (2,3):
                count = 0
                for task in tasks(source):
                    idx.add(*task)
                    count += len(task[0])
                    if count % 250000 == 0:
                        log(f'S{source}: {count:,} records; {len(idx.postings):,} live keys')
                log(f'S{source} complete: {count:,} records')
        c = idx.query(k=args.k)
        cand = pd.DataFrame({'anchor':query_owner[c.anchor],'target_code':c.code,'score':c.score,'shared':c.shared})
        cand = cand.sort_values(['anchor','score','target_code'],ascending=[True,False,True]).drop_duplicates(['anchor','target_code'])
        cand['rank'] = cand.groupby(['anchor', cand.target_code // SOURCE_BASE]).cumcount()
        cand.to_parquet(cp,index=False)
        log(f'froze {len(cand):,} candidates without consulting labels')
        del idx
    else:
        cand = pd.read_parquet(cp)
    old = []
    for f in sorted(pairs.glob('val_*.parquet')):
        for b in pq.ParquetFile(f).iter_batches(columns=['anchor','target_code'], batch_size=100000):
            z = b.to_pandas()
            z = z[z.anchor.isin(mapping)].copy()
            z['anchor'] = z.anchor.map(mapping)
            old.append(z)
    old = pd.concat(old,ignore_index=True)
    basekeys = np.unique(old.anchor.to_numpy(np.int64)*10**10 + old.target_code.to_numpy(np.int64))
    lookup = dict(zip(vals.astype(int),range(len(vals))))
    truth = []
    with (data / 'train_ground_truth.tsv').open() as f:
        next(f)
        for line in f:
            sid, ids = line.rstrip('\n').split('\t')
            i = lookup.get(int(sid[3:]))
            if i is not None:
                truth.extend(i*10**10 + int(t[1])*SOURCE_BASE + int(t[3:]) for t in ids.split(',') if t)
    truth = np.array(truth,np.int64)
    nt = a.n_true.to_numpy()
    assert np.array_equal(np.bincount(truth//10**10,minlength=len(a)),nt)
    def measure(keys):
        found = truth[np.isin(truth,keys)]
        counts = np.bincount(found//10**10,minlength=len(a))
        scores = entity_f05_counts(counts,np.zeros(len(a)),nt)
        return {'ceiling':float(scores.mean()),'pair_recall':float(len(found)/max(1,len(truth))),
                'candidates_per_anchor':len(keys)/len(a), 'true_pairs_found':len(found)}, scores
    baseline, bs = measure(basekeys)
    report = {'baseline':baseline,'n_anchors':len(a),'note':'Development candidate-recall probe. No final matcher gain measured. Labels never used for candidate generation.', 'variants':{}}
    for k in sorted(set([v for v in [10,30,100,args.k] if v <= args.k])):
        rows = cand[cand['rank'] < k]
        ck = rows.anchor.to_numpy(np.int64)*10**10 + rows.target_code.to_numpy(np.int64)
        union = np.union1d(basekeys,ck)
        r, scores = measure(union)
        delta = scores-bs
        rng = np.random.default_rng(772)
        boots = np.array([delta[rng.integers(0,len(a),len(a))].mean() for _ in range(1000)])
        r.update(delta=float(delta.mean()), delta_ci95=np.quantile(boots,[.025,.975]).tolist(),
                 new_true_pairs=r['true_pairs_found']-baseline['true_pairs_found'],
                 per_country={ct:float(scores[a.country==ct].mean()) for ct in sorted(a.country.unique())})
        report['variants'][str(k)] = r
    a.to_parquet(out / 'anchors.parquet',index=False)
    (out / 'report.json').write_text(json.dumps(report,indent=2))
    log(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
