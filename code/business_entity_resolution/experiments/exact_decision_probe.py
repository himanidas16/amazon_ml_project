"""Tune an exact expected-F0.5 rule on one half, report on the other.

The source predictions come from error_audit.py. All ownership decisions use
all candidates before restricting metric rows to a half. This is a development
experiment: the parent validation set has been used in previous experiments.
"""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
from business_er.exact_decision import calibrated_probabilities, expected_f05_exact
from business_er.evaluate import one_owner
from business_er.metrics import entity_f05_counts


def main():
    out = ROOT/'artifacts/exact_decision_sel'
    out.mkdir(parents=True,exist_ok=True)
    d = pd.read_parquet(ROOT/'artifacts/error_audit_sel/predictions.parquet')
    a = pd.read_parquet(ROOT/'artifacts/error_audit_sel/anchors.parquet')
    anchor,target,p,y = [d[k].to_numpy() for k in ['anchor','target_code','p','label']]
    n=len(a)
    def scores(keep,prob):
        keep=one_owner(keep,prob,target,anchor)
        tp=np.bincount(anchor[keep],weights=y[keep],minlength=n)
        fp=np.bincount(anchor[keep],weights=1-y[keep],minlength=n)
        return entity_f05_counts(tp,fp,a.n_true.to_numpy())
    tune=np.random.default_rng(731).random(n)<.5
    baseline=scores(p>=.75,p)
    rows=[]; vectors=[]
    start=time.time()
    for temp in (1.,1.5,2.):
        for shift in (-.5,0.,.5):
            q=calibrated_probabilities(p,temp,shift)
            keep=expected_f05_exact(q,anchor)
            s=scores(keep,q)
            rows.append({'temperature':temp,'shift':shift,'tune':float(s[tune].mean())})
            vectors.append(s)
            print(f'{time.time()-start:.1f}s: {rows[-1]}',flush=True)
    selected=int(np.argmax([r['tune'] for r in rows]))
    best=vectors[selected]
    delta=(best-baseline)[~tune]
    rng=np.random.default_rng(916)
    boots=[delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)]
    report={'selected_on_tune':rows[selected], 'baseline_tune':float(baseline[tune].mean()),
            'baseline_holdout':float(baseline[~tune].mean()),'exact_holdout':float(best[~tune].mean()),
            'holdout_delta':float(delta.mean()),'delta_ci95':np.quantile(boots,[.025,.975]).tolist(),
            'tune_trials':rows,'note':'Development evaluation; the parent validation businesses were used for earlier experiments. No production promotion performed.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
