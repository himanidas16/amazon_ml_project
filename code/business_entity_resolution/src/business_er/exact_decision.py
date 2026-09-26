"""Exact expected macro-F0.5 cardinality under independent pair probabilities.

Unlike a ratio of expected counts, this integrates the actual F0.5 over
Bernoulli outcomes. It cannot account for unretrieved matches or repair bad
probability calibration; both must be checked on validation before adoption.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=64)
def _quadrature(nodes):
    x, w = np.polynomial.legendre.leggauss(nodes)
    return (x + 1) / 2, w / 2


def calibrated_probabilities(p, temperature=1.0, shift=0.0):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    p = np.asarray(p, np.float64)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('probabilities must be finite and between zero and one')
    if temperature == 1 and shift == 0:
        return p
    q = np.clip(p,1e-12,1-1e-12)
    z = (np.log(q)-np.log1p(-q))/temperature + shift
    return np.exp(-np.logaddexp(0,-z))


def cardinality_utilities(p, max_k=20):
    """Expected scores for predicting zero, then top 1..max_k pairs.

    Input p is descending. For k>0, F0.5 = 5*TP/(4*k + N).
    The identity 1/d = integral_0^1 u**(d-1) du yields a polynomial:
      5*u**(4*k) * product(1-p+p*u) * sum_topk(p/(1-p+p*u)).
    Gauss-Legendre with sufficient nodes integrates it exactly (up to float
    rounding), including p=0/1. The zero-prediction utility is P(N=0).
    """
    p = calibrated_probabilities(p)
    if max_k < 1:
        raise ValueError('max_k must be positive')
    if np.any(p[1:] > p[:-1]):
        raise ValueError('probabilities must be descending')
    m, kmax = len(p), min(max_k,len(p))
    if not m:
        return np.array([1.0])
    needed = (4*kmax+m+1)//2
    nodes = ((needed+15)//16)*16
    u, w = _quadrature(nodes)
    factors = 1-p[:,None]+p[:,None]*u
    product = factors.prod(axis=0)
    prefix = np.cumsum(p[:kmax,None]/factors[:kmax],axis=0)
    k = np.arange(1,kmax+1)
    utilities = 5*((u[None,:]**(4*k[:,None]))*prefix*product*w).sum(axis=1)
    return np.r_[np.prod(1-p),utilities]


def expected_f05_exact(p, anchor, max_k=20):
    """Choose the smallest optimal cardinality for each sorted anchor group."""
    p = calibrated_probabilities(p)
    anchor = np.asarray(anchor)
    if len(p) != len(anchor):
        raise ValueError('probability and anchor lengths differ')
    if np.any(anchor[1:] < anchor[:-1]):
        raise ValueError('pairs must be grouped by anchor')
    keep = np.zeros(len(p),bool)
    bounds = np.r_[0,np.flatnonzero(anchor[1:]!=anchor[:-1])+1,len(anchor)]
    for lo,hi in zip(bounds[:-1],bounds[1:]):
        if lo == hi:
            continue
        order = np.argsort(-p[lo:hi],kind='stable')
        utilities = cardinality_utilities(p[lo:hi][order],max_k=max_k)
        k = int(np.argmax(utilities))
        keep[lo+order[:k]] = True
    return keep
