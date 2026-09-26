"""Load feature shards with one preallocated matrix, avoiding pandas concat copies."""
import numpy as np
import pyarrow.parquet as pq
from .features import FEATURE_NAMES

def load_pair_parts(files, sort_anchors=False, mmap_path=None):
    """mmap_path: put the feature matrix in a disk-backed array instead of RAM,
    so a large training set cannot push the process over its memory cap (the
    OS can drop and re-read those pages)."""
    files=list(files)
    if not files: raise ValueError("no pair shards found")
    metadata=[pq.ParquetFile(path) for path in files]
    weighted=["weight" in p.schema_arrow.names for p in metadata]
    if len(set(weighted)) != 1: raise ValueError("inconsistent weight columns across shards")
    total=sum(p.metadata.num_rows for p in metadata)
    X=(np.lib.format.open_memmap(mmap_path, mode="w+", dtype=np.float32, shape=(total,len(FEATURE_NAMES)))
       if mmap_path is not None else np.empty((total,len(FEATURE_NAMES)),np.float32))
    y=np.empty(total,np.int8); anchor=np.empty(total,np.int64)
    weight=np.empty(total,np.float32) if weighted[0] else None
    columns=list(FEATURE_NAMES)+["label","anchor"]+(["weight"] if weighted[0] else [])
    pos=0
    for part in metadata:
        for batch in part.iter_batches(batch_size=50000,columns=columns):
            frame=batch.to_pandas(); end=pos+len(frame)
            X[pos:end]=frame[list(FEATURE_NAMES)].to_numpy(np.float32)
            y[pos:end]=frame.label.to_numpy(np.int8); anchor[pos:end]=frame.anchor.to_numpy(np.int64)
            if weight is not None: weight[pos:end]=frame.weight.to_numpy(np.float32)
            pos=end
    if sort_anchors and np.any(anchor[1:]<anchor[:-1]):
        order=np.argsort(anchor,kind="stable")
        X,y,anchor=X[order],y[order],anchor[order]
        if weight is not None: weight=weight[order]
    return X,y,anchor,weight
