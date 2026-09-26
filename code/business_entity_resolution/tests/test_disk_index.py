import numpy as np
import pandas as pd
from business_er.disk_index import build_disk_index
from business_er.retrieve import CHANNELS,build_index,compute_keys,generate_candidates

def test_disk_index_exactly_preserves_scores_caps_and_ties(tmp_path):
    paths={}
    for source in (2,3):
        frame=pd.DataFrame({'entity_id':[f'S{source}-4',f'S{source}-8'],
            'business_name':['Acme Services','Acme Services'],'business_address':['12 Main Road','19 Hill Lane'],'country':['US','US']})
        path=tmp_path/f's{source}.tsv'; frame.to_csv(path,sep='\t',index=False); paths[source]=path
    ordinary=build_index(paths,country='US')
    disk=build_disk_index(paths,'US',None,tmp_path/'index')
    np.testing.assert_equal(ordinary.codes,disk.codes)
    for ch in CHANNELS:
        np.testing.assert_equal(ordinary.keys[ch],disk.keys[ch]); np.testing.assert_equal(ordinary.rows[ch],disk.rows[ch])
    query=compute_keys(['Acme Services'],['12 Main Road'],['US'])
    for cap in (1,3,10):
        a=generate_candidates(ordinary,query,1,cap=cap,k=2)
        b=generate_candidates(disk,query,1,cap=cap,k=2)
        for field in ('anchor','target','score','channels','rank'):
            np.testing.assert_equal(getattr(a,field),getattr(b,field))
    resumed=build_disk_index(paths,'US',None,tmp_path/'index')
    np.testing.assert_equal(resumed.codes,ordinary.codes)
