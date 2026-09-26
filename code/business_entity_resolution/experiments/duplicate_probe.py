"""Measure conservative exact-text duplicate expansion against complete validation truth."""
import sys,time,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/"code/business_entity_resolution/src"))
from business_er.retrieve import target_code, SOURCE_BASE
from business_er.metrics import entity_f05_counts
from business_er.evaluate import one_owner
out=ROOT/"artifacts/duplicate_probe"; out.mkdir(parents=True,exist_ok=True); t0=time.time()
def log(s): print(f"[{time.time()-t0:.1f}s] {s}",flush=True)
path=out/"target_fingerprints.npz"
if not path.exists():
    hashes=[]; codes=[]
    for source in (2,3):
        for df in pd.read_csv(ROOT/f"amazon_ml_dataset/student_resource/dataset/train/train_source{source}.tsv",sep="\t",keep_default_na=False,dtype=str,chunksize=100000,quoting=3):
            # Preserve punctuation and accents; only case/whitespace normalization.
            for field in ("business_name","business_address"):
                df[field]=df[field].str.normalize("NFKC").str.casefold().str.replace(r"\s+"," ",regex=True).str.strip()
            valid=(df.business_name.str.len()>=5)&(df.business_address.str.len()>=12)
            df=df[valid]
            hashes.append(pd.util.hash_pandas_object(df[["country","business_name","business_address"]],index=False).to_numpy())
            codes.append(target_code(source,df.entity_id.str.slice(3).astype(np.int64).to_numpy()))
        log(f"fingerprinted source {source}")
    h=np.concatenate(hashes); cd=np.concatenate(codes); del hashes,codes
    # Keep only duplicate groups: singletons cannot expand anything.
    order=np.argsort(h); h=h[order]; cd=cd[order]
    dup=np.r_[False,h[1:]==h[:-1]]|np.r_[h[:-1]==h[1:],False]
    h=h[dup]; cd=cd[dup]; np.savez(path,hash=h,code=cd); log(f"{len(cd)} records in duplicate groups")
else:
    z=np.load(path); h=z["hash"]; cd=z["code"]
D=pd.read_parquet(ROOT/"artifacts/pairs_val.parquet",columns=["anchor","target_code","label"]); A=pd.read_parquet(ROOT/"artifacts/anchors_val.parquet"); p=np.load(ROOT/"artifacts/val_scores_v1.npy")
a=D.anchor.to_numpy(); code=D.target_code.to_numpy(); y=D.label.to_numpy(); nt=A.n_true.to_numpy(); n=len(A)
order=np.argsort(cd); pos=np.searchsorted(cd[order],code); valid=pos<len(cd); valid &= cd[order[np.minimum(pos,len(cd)-1)]]==code
pair_hash=h[order[pos[valid]]]
# One group per (anchor, exact text), with its highest existing score.
groups=pd.DataFrame({"anchor":a[valid],"hash":pair_hash,"p":p[valid]}).groupby(["anchor","hash"],sort=False).p.max().reset_index()
lo=np.searchsorted(h,groups["hash"].to_numpy(),"left"); hi=np.searchsorted(h,groups["hash"].to_numpy(),"right"); size=hi-lo
rows=np.repeat(np.arange(len(groups)),size); off=np.arange(size.sum())-np.repeat(np.cumsum(size)-size,size)
extra_code=cd[np.repeat(lo,size)+off]; extra_anchor=groups.anchor.to_numpy()[rows]; extra_p=groups.p.to_numpy()[rows]
expanded=pd.DataFrame({"anchor":np.r_[a,extra_anchor],"code":np.r_[code,extra_code],"p":np.r_[p,extra_p]}).groupby(["anchor","code"],sort=True).p.max().reset_index()
aa=expanded.anchor.to_numpy(np.int64); cc=expanded.code.to_numpy(); pp=expanded.p.to_numpy()
lookup=dict(zip(A.s1_value.astype(int),range(n))); truthkeys=[]
with open(ROOT/"amazon_ml_dataset/student_resource/dataset/train/train_ground_truth.tsv") as f:
    next(f)
    for line in f:
        sid,ids=line.rstrip("\n").split("\t"); idx=lookup.get(int(sid[3:]))
        if idx is not None and ids:
            truthkeys.extend(idx*10**10+int(t[1])*SOURCE_BASE+int(t[3:]) for t in ids.split(","))
yy=np.isin(aa*10**10+cc,np.array(truthkeys,np.int64)).astype(np.int8)
def score(a,c,p,y,t):
    k=one_owner(p>=t,p,c,a); tp=np.bincount(a[k],weights=y[k],minlength=n); fp=np.bincount(a[k],weights=1-y[k],minlength=n)
    return entity_f05_counts(tp,fp,nt)
b=score(a,code,p,y,.8)
found=np.bincount(a,weights=y,minlength=n); expanded_found=np.bincount(aa,weights=yy,minlength=n)
report={"pairs_before":len(a),"pairs_after":len(aa),"new_true_pairs":int(yy.sum()-y.sum()),"ceiling_before":float(entity_f05_counts(found,np.zeros(n),nt).mean()),"ceiling_after":float(entity_f05_counts(expanded_found,np.zeros(n),nt).mean()),"baseline":float(b.mean()),"thresholds":[]}
for t in [.75,.8,.85,.9,.95]:
    s=score(aa,cc,pp,yy,t); report["thresholds"].append({"t":t,"score":float(s.mean()),"delta":float((s-b).mean()),"countries":{c:float(s[(A.country==c).to_numpy()].mean()) for c in A.country.unique()}})
(out/"report.json").write_text(json.dumps(report,indent=2)); np.savez(out/"expanded_val.npz",anchor=aa,code=cc,p=pp,label=yy); log(report)
