"""Complementary character retrieval with bounded postings for pilot queries.

The index retains only grams requested by the supplied queries. Target rows
are streamed over the FULL pool: document frequencies and common-gram caps
are therefore exact, not estimated from a conveniently small target sample.
No labels or match probabilities enter retrieval. This module deliberately
stays separate from the production candidate distribution until evaluated.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np

from .normalize import address_views, expand_address_tokens, name_views
from .retrieve import SOURCE_BASE, glued_name

CHAR_VERSION = '1.0.0'


def character_keys(name: str, address: str, country: str, fields=('name', 'address')):
    """Distinct field-scoped grams, symmetric on query and target records.

    Interior grams tolerate word boundaries, punctuation and local typos.
    All positions are retained; long records are not silently prefix-truncated.
    """
    out = set()
    if 'name' in fields:
        folded = name_views(name)['folded']
        text = glued_name(folded) or ''.join(folded.split())
        out.update((country, 'n', text[i:i+4]) for i in range(len(text)-3))
    if 'address' in fields:
        folded = address_views(address)['folded']
        text = ''.join(expand_address_tokens(folded, country).split())
        out.update((country, 'a', text[i:i+5]) for i in range(len(text)-4))
    return out


@dataclass
class CharCandidates:
    anchor: np.ndarray
    code: np.ndarray
    score: np.ndarray
    shared: np.ndarray


class QueryGramIndex:
    """A query-filtered exact inverted index, bounded by cap * query grams.

    Frequent grams are discarded completely, never truncated to whichever
    target rows happened to arrive first. Fits a modest pilot on a laptop.
    For all-test deployment use disk-backed postings rather than increasing
    the query batch and exceeding the memory budget.
    """
    def __init__(self, names, addresses, countries, cap=256, fields=('name','address')):
        if cap < 1:
            raise ValueError('cap must be positive')
        if not fields or set(fields) - {'name','address'}:
            raise ValueError('fields must contain name and/or address')
        if not (len(names) == len(addresses) == len(countries)):
            raise ValueError('query columns differ in length')
        self.cap, self.fields = cap, tuple(fields)
        self.queries = [character_keys(n,a,c,self.fields) for n,a,c in zip(names,addresses,countries)]
        self.postings = {k: array('q') for q in self.queries for k in q}
        self.counts = dict.fromkeys(self.postings, 0)
        self.n_country = defaultdict(int)

    def add(self, codes, names, addresses, countries):
        if not (len(codes) == len(names) == len(addresses) == len(countries)):
            raise ValueError('target columns differ in length')
        for code, name, addr, country in zip(codes,names,addresses,countries):
            self.n_country[country] += 1
            for key in character_keys(name,addr,country,self.fields):
                if key not in self.counts:
                    continue
                count = self.counts[key] + 1
                self.counts[key] = count
                if count <= self.cap:
                    self.postings[key].append(int(code))
                elif count == self.cap + 1:
                    del self.postings[key]

    def query(self, k=30, min_shared=2):
        if k < 1 or min_shared < 1:
            raise ValueError('k and min_shared must be positive')
        anchors, codes, scores, shared = [], [], [], []
        for ai, keys in enumerate(self.queries):
            weights = defaultdict(float)
            hits = defaultdict(int)
            for key in sorted(keys):
                rows = self.postings.get(key)
                if not rows:
                    continue
                weight = math.log1p(self.n_country[key[0]] / self.counts[key])
                for code in rows:
                    weights[code] += weight
                    hits[code] += 1
            for source in (2,3):
                ranked = sorted((c for c in weights if c//SOURCE_BASE == source and hits[c] >= min_shared),
                                key=lambda c: (-weights[c], c))[:k]
                anchors.extend([ai]*len(ranked)); codes.extend(ranked)
                scores.extend(weights[c] for c in ranked); shared.extend(hits[c] for c in ranked)
        return CharCandidates(np.asarray(anchors,np.int32), np.asarray(codes,np.int64),
                              np.asarray(scores,np.float32), np.asarray(shared,np.int32))


def merge_partial(index, counts, postings, n_country):
    """Merge a streamed shard exactly, applying the cap to GLOBAL counts.

    A locally discarded gram necessarily exceeds the global cap too. Thus
    splitting targets across workers cannot change which candidates survive.
    """
    for country, n in n_country.items():
        index.n_country[country] += n
    for key, count in counts.items():
        if key not in index.counts or count < 1:
            raise ValueError('invalid partial index')
        total = index.counts[key] + count
        index.counts[key] = total
        if total > index.cap:
            index.postings.pop(key,None)
        else:
            rows = postings[key]
            if len(rows) != count:
                raise ValueError('partial posting count mismatch')
            index.postings[key].extend(rows)


_WORKER_ARGS = None


def init_gram_worker(names, addresses, countries, cap, fields):
    global _WORKER_ARGS
    _WORKER_ARGS = (names,addresses,countries,cap,fields)


def build_gram_partial(task):
    names,addresses,countries,cap,fields = _WORKER_ARGS
    idx = QueryGramIndex(names,addresses,countries,cap=cap,fields=fields)
    idx.add(*task)
    return ({k:v for k,v in idx.counts.items() if v},
            {k:v for k,v in idx.postings.items() if len(v)},dict(idx.n_country))
