import numpy as np
import pytest
from business_er.char_retrieve import QueryGramIndex
from business_er.retrieve import SOURCE_BASE


def test_typo_recovery_country_and_source_identity():
    idx = QueryGramIndex(['Xylophonic'], [''], ['France'], fields=('name',))
    codes = [2*SOURCE_BASE+7, 3*SOURCE_BASE+7, 2*SOURCE_BASE+8]
    idx.add(codes, ['Xylophnoic']*3, ['']*3, ['France','France','US'])
    result = idx.query(k=1)
    assert set(result.code) == set(codes[:2])
    assert list(result.anchor) == [0,0]


def test_common_grams_are_dropped_not_truncated_and_order_independent():
    names = ['Same Business', 'Same Business', 'Same Business']
    codes = np.array([2*SOURCE_BASE+3, 2*SOURCE_BASE+1, 2*SOURCE_BASE+2])
    for order in ([0,1,2],[2,1,0]):
        idx = QueryGramIndex(['Same Business'], [''], ['India'], cap=2, fields=('name',))
        for i in order:
            idx.add([codes[i]], [names[i]], [''], ['India'])
        assert len(idx.query().code) == 0
        assert not idx.postings


def test_address_fallback_and_empty_records():
    idx = QueryGramIndex(['',''], ['123 Boulevard Voltaire',''], ['France','France'])
    idx.add([2*SOURCE_BASE+1], [''], ['123 BD Voltaire'], ['France'])
    r = idx.query()
    assert r.code.tolist() == [2*SOURCE_BASE+1]
    assert r.anchor.tolist() == [0]


def test_ties_and_batching_are_deterministic():
    results = []
    for order in ([0,1], [1,0]):
        idx = QueryGramIndex(['Uncommon Trading'], [''], ['US'], fields=('name',))
        codes = [2*SOURCE_BASE+2, 2*SOURCE_BASE+1]
        for i in order:
            idx.add([codes[i]], ['Uncommon Trading'], [''], ['US'])
        results.append(idx.query(k=1).code.tolist())
    assert results == [[2*SOURCE_BASE+1]]*2


def test_column_mismatch_rejected():
    with pytest.raises(ValueError):
        QueryGramIndex(['a'], [], ['US'])


def test_parallel_shard_merge_matches_full_pool_in_any_order():
    from business_er.char_retrieve import merge_partial
    args=(['Xylophone Trading','Quantum Biology'],['',''],['US','US'])
    codes=[2*SOURCE_BASE+i for i in range(1,5)]
    names=['Xylophone Trading']*3+['Quantum Biolgy']
    full=QueryGramIndex(*args,cap=2,fields=('name',))
    full.add(codes,names,['']*4,['US']*4)
    pieces=[]
    for lo,hi in [(0,2),(2,4)]:
        part=QueryGramIndex(*args,cap=2,fields=('name',))
        part.add(codes[lo:hi],names[lo:hi],['']*(hi-lo),['US']*(hi-lo))
        pieces.append(part)
    for order in ([0,1],[1,0]):
        merged=QueryGramIndex(*args,cap=2,fields=('name',))
        for i in order:
            part=pieces[i]
            merge_partial(merged,{k:v for k,v in part.counts.items() if v},part.postings,part.n_country)
        assert merged.counts==full.counts
        assert dict(merged.n_country)==dict(full.n_country)
        np.testing.assert_array_equal(merged.query().code,full.query().code)
        np.testing.assert_allclose(merged.query().score,full.query().score)
