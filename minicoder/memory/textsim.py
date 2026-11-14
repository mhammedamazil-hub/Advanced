"""Tiny dependency-free text similarity + vector index.

Used by working/episodic/semantic memory for relevance scoring.  A signed
hashing trick maps tokens to a fixed-size sparse vector; cosine similarity on
these vectors blends with exact token-overlap (Jaccard) for robust retrieval.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
EMB_DIM = 512
STOP = {
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "were", "and", "or",
    "for", "on", "with", "as", "by", "at", "from", "that", "this", "it", "be",
}


def tokenize(text: str) -> list[str]:
    toks = [t.lower() for t in _TOKEN_RE.findall(text or "")]
    return [t for t in toks if t not in STOP]


def embed(text: str) -> list[float]:
    """Signed hashing embedding into EMB_DIM dims (sparse, L2-normalized)."""
    vec = [0.0] * EMB_DIM
    for tok in tokenize(text):
        h = int(hashlib.blake2b(tok.encode(), digest_size=8).hexdigest(), 16)
        idx = h % EMB_DIM
        sign = 1.0 if (h >> 63) & 1 == 0 else -1.0
        vec[idx] += sign * 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    xa, xb = list(a), list(b)
    if not xa or not xb:
        return 0.0
    return float(sum(p * q for p, q in zip(xa, xb)))


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def hybrid_similarity(query: str, text: str, query_vec=None, text_vec=None) -> float:
    """Blend semantic (cosine on hashing vectors) with lexical (Jaccard)."""
    qv = query_vec if query_vec is not None else embed(query)
    tv = text_vec if text_vec is not None else embed(text)
    cos = max(0.0, cosine(qv, tv))
    jac = jaccard(query, text)
    return 0.6 * cos + 0.4 * jac
