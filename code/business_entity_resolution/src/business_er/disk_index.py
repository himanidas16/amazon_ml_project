"""Build the exact retrieval index source by source, backed by memory-mapped arrays.

Keeps the original target order and stable posting order. This changes memory
use, not keys, block sizes, candidate scores, or the retrieval policy.
"""
import gc
import json
from pathlib import Path
import numpy as np
from .retrieve import CHANNELS, KeyIndex, build_index


def build_disk_index(source_paths, country, freq, directory, log=lambda s:None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    complete=directory/'complete.json'
    if not complete.exists():
        sources=sorted(source_paths); offsets={}; total=0
        for source in sources:
            sub=directory/f's{source}'; sub.mkdir(exist_ok=True)
            if not (sub/'complete.json').exists():
                index=build_index({source:source_paths[source]},country=country,freq=freq,
                                  workers=0,read_chunk=100000,log=log)
                np.save(sub/'codes.npy',index.codes)
                for channel in CHANNELS:
                    np.save(sub/f'{channel}_keys.npy',index.keys[channel])
                    np.save(sub/f'{channel}_rows.npy',index.rows[channel])
                (sub/'complete.json').write_text(json.dumps({'targets':index.n_targets}))
                del index; gc.collect()
            count=json.loads((sub/'complete.json').read_text())['targets']
            offsets[source]=total; total+=count
        codes=np.lib.format.open_memmap(directory/'codes.npy',mode='w+',dtype=np.int64,shape=(total,))
        for source in sources:
            source_codes=np.load(directory/f's{source}/codes.npy',mmap_mode='r')
            codes[offsets[source]:offsets[source]+len(source_codes)]=source_codes
        codes.flush(); del codes,source_codes
        for channel in CHANNELS:
            kp=[np.load(directory/f's{s}/{channel}_keys.npy',mmap_mode='r') for s in sources]
            rp=[np.load(directory/f's{s}/{channel}_rows.npy',mmap_mode='r') for s in sources]
            keys=np.concatenate(kp); rows=np.concatenate([r+np.int32(offsets[s]) for s,r in zip(sources,rp)])
            order=np.argsort(keys,kind='stable')
            np.save(directory/f'{channel}_keys.npy',keys[order]); np.save(directory/f'{channel}_rows.npy',rows[order])
            del kp,rp,keys,rows,order; gc.collect(); log(f'merged {channel}')
        complete.write_text(json.dumps({'country':country,'targets':total}))
    meta=json.loads(complete.read_text())
    if meta['country']!=country: raise ValueError('index cache country differs')
    return KeyIndex(
        keys={c:np.load(directory/f'{c}_keys.npy',mmap_mode='r') for c in CHANNELS},
        rows={c:np.load(directory/f'{c}_rows.npy',mmap_mode='r') for c in CHANNELS},
        codes=np.load(directory/'codes.npy',mmap_mode='r'),country=country)
