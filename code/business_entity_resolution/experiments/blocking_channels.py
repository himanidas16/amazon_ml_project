"""Step 5 experiment: per-channel blocking recall, measured with numpy instead of Python dicts.

Keys are hashed to int64 and postings held as flat arrays, so 13M postings cost
~160 MB rather than several GB.  PYTHONHASHSEED=0 makes hash() reproducible.

Run from repo root:  PYTHONHASHSEED=0 python3 code/business_entity_resolution/experiments/blocking_channels.py
"""
import sys, time, resource, random
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code/business_entity_resolution/src'))
import numpy as np
from business_er.normalize import name_views, address_views
from business_er.splits import load_split

D=str(ROOT/'amazon_ml_dataset/student_resource/dataset/train')
t0=time.time()
def mark(m): print(f"[{time.time()-t0:6.1f}s {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:4.1f}GB] {m}",flush=True)

MASK=(1<<62)-1
def h(s): return hash(s) & MASK

sp = load_split(ROOT/'artifacts/split_25fold.npz')
f0_s1 = set(sp.s1_values[sp.s1_fold==0].tolist())
f0_t  = {s:set(sp.tgt_values[s][sp.tgt_fold[s]==0].tolist()) for s in (2,3)}
mark(f"dev fold: {len(f0_s1):,} S1, {len(f0_t[2]):,} S2, {len(f0_t[3]):,} S3")

TKEY = lambda src,val: src*2_000_000_000 + val     # < 8e9
# (anchor, target) packed into one int64.  Must exceed max TKEY (~7e9) or
# pairs from different anchors collide -- the old 4e9 base did exactly that.
PAIR_BASE = 10_000_000_000

# ---------------- key design -------------------------------------------------
def keys_of(name, addr, country):
    nv, av = name_views(name), address_views(addr)
    nf, af = nv['folded'], av['folded']
    ntok = [t for t in nf.split() if len(t)>=3]
    atok = [t for t in af.split() if len(t)>=4 and not t.isdigit()]
    nums = av['numbers'].split()
    k=defaultdict(list); c=country
    if nf:
        k['name_exact'].append(f"{c}|{nf}")
        k['name_sorted'].append(f"{c}|{' '.join(sorted(set(ntok)))}")
    for t in sorted(set(ntok))[:6]:        k['name_token'].append(f"{c}|{t}")
    st=sorted(set(ntok))[:4]
    for i in range(len(st)):
        for j in range(i+1,len(st)):       k['name_pair'].append(f"{c}|{st[i]}|{st[j]}")
    for t in sorted(set(atok))[:6]:        k['addr_token'].append(f"{c}|a|{t}")
    for n in sorted(set(nums))[:2]:
        for t in sorted(set(atok))[:3]:    k['num_token'].append(f"{c}|{n}|{t}")
    if nums: k['num_set'].append(f"{c}|#|{'-'.join(sorted(set(nums)))}")
    for t in sorted(set(ntok))[:3]:
        for n in sorted(set(nums))[:2]:    k['name_num'].append(f"{c}|{t}|{n}")
    return k

CH=['name_exact','name_sorted','name_token','name_pair','addr_token','num_token','num_set','name_num']

# ---------------- index the targets (streaming, text never retained) ---------
tgt_keys=[]                       # int64 TKEY per target row
post_k={c:[] for c in CH}; post_i={c:[] for c in CH}
for s in (2,3):
    keep=f0_t[s]
    with open(f'{D}/train_source{s}.tsv',encoding='utf-8') as f:
        next(f)
        for line in f:
            eid,rest=line.split('\t',1)
            v=int(eid[3:])
            if v not in keep: continue
            nm,ad,ct=rest.rstrip('\n').split('\t')
            idx=len(tgt_keys); tgt_keys.append(TKEY(s,v))
            for ch,kk in keys_of(nm,ad,ct).items():
                pk=post_k[ch]; pi=post_i[ch]
                for key in kk: pk.append(h(key)); pi.append(idx)
    mark(f"indexed S{s} (targets so far {len(tgt_keys):,})")
tgt_keys=np.asarray(tgt_keys,dtype=np.int64)

IDX={}
for ch in CH:
    k=np.asarray(post_k[ch],dtype=np.int64); i=np.asarray(post_i[ch],dtype=np.int32)
    o=np.argsort(k,kind='stable'); IDX[ch]=(k[o], i[o])
    post_k[ch]=post_i[ch]=None
mark(f"index built: {sum(len(v[0]) for v in IDX.values()):,} postings")
print("\n  channel        distinct keys    mean block   p99 block     max block")
for ch in CH:
    k,_=IDX[ch]
    _,cnt=np.unique(k,return_counts=True)
    print(f"  {ch:<14}{len(cnt):>13,}{cnt.mean():>14.2f}{np.percentile(cnt,99):>12.0f}{cnt.max():>14,}")

# ---------------- anchors ----------------------------------------------------
truth={}
with open(f'{D}/train_ground_truth.tsv',encoding='utf-8') as f:
    next(f)
    for line in f:
        s1,raw=line.rstrip('\n').split('\t')
        v=int(s1[3:])
        if v in f0_s1: truth[v]=[x for x in raw.split(',') if x]
random.seed(11)
anchors=random.sample(sorted(truth),20000)
apos={v:i for i,v in enumerate(anchors)}
akeys={ch:([],[]) for ch in CH}
with open(f'{D}/train_source1.tsv',encoding='utf-8') as f:
    next(f)
    for line in f:
        eid,rest=line.split('\t',1)
        v=int(eid[3:])
        if v not in apos: continue
        nm,ad,ct=rest.rstrip('\n').split('\t')
        ai=apos[v]
        for ch,kk in keys_of(nm,ad,ct).items():
            for key in kk: akeys[ch][0].append(h(key)); akeys[ch][1].append(ai)
for ch in CH: akeys[ch]=(np.asarray(akeys[ch][0],np.int64),np.asarray(akeys[ch][1],np.int32))
mark(f"anchor keys built for {len(anchors):,} anchors")

truth_codes=[]; n_true=0
for v,ids in ((v,truth[v]) for v in anchors):
    ai=apos[v]
    for t in ids:
        truth_codes.append(ai*PAIR_BASE + TKEY(int(t[1]), int(t[3:])))
    n_true+=len(ids)
truth_codes=np.sort(np.asarray(truth_codes,np.int64))

def probe(channels, cap, batch=500):
    """Blocking recall for a channel set.  Anchors are processed in batches so
    memory is bounded: at cap 5000 the full expansion is billions of pairs,
    which is what OOM-killed the earlier run (and likely froze the laptop)."""
    hit=0; per=np.zeros(len(anchors),np.int64)
    for b0 in range(0,len(anchors),batch):
        codes=[]
        for ch in channels:
            sk,si=IDX[ch]; ak,ai=akeys[ch]
            sel=(ai>=b0)&(ai<b0+batch); ak,ai=ak[sel],ai[sel]
            lo=np.searchsorted(sk,ak,'left'); hi=np.searchsorted(sk,ak,'right')
            sz=hi-lo
            ok=(sz>0)&(sz<=cap)
            lo,sz,ai2=lo[ok],sz[ok],ai[ok]
            if not len(lo): continue
            tot=int(sz.sum())
            starts=np.repeat(lo,sz)
            offs=np.arange(tot)-np.repeat(np.cumsum(sz)-sz,sz)
            rows=si[starts+offs]
            codes.append(np.repeat(ai2,sz).astype(np.int64)*PAIR_BASE + tgt_keys[rows])
        if not codes: continue
        allc=np.unique(np.concatenate(codes))
        hit+=len(np.intersect1d(allc,truth_codes,assume_unique=True))
        per+=np.bincount(allc//PAIR_BASE, minlength=len(anchors))
    return hit/n_true, per.mean(), np.percentile(per,95)

print(f"\n=== per-channel recall ({len(anchors):,} anchors, cap 300) ===")
print(f"  {'channel':<14}{'pair recall':>13}{'cand/S1':>10}{'p95':>8}")
for ch in CH:
    r,m,p=probe([ch],300)
    print(f"  {ch:<14}{r:>12.4f}{m:>10.1f}{p:>8.0f}")

print(f"\n=== cumulative union (cap 300) ===")
print(f"  {'channels':<56}{'recall':>9}{'cand/S1':>10}{'p95':>8}")
use=[]
for ch in ['name_token','num_token','addr_token','name_pair','name_num','num_set','name_sorted','name_exact']:
    use.append(ch); r,m,p=probe(use,300)
    print(f"  {'+'.join(use):<56}{r:>9.4f}{m:>10.1f}{p:>8.0f}")

print(f"\n=== block-size cap sweep on the full union ===")
print(f"  {'cap':>6}{'recall':>9}{'cand/S1':>10}{'p95':>8}")
for cap in (50,100,300,1000,5000):
    r,m,p=probe(CH,cap)
    print(f"  {cap:>6}{r:>9.4f}{m:>10.1f}{p:>8.0f}")
mark("done")