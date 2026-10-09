"""Three ways to rank documents for a query: keywords, vectors, and both."""
import re

import numpy as np
from rank_bm25 import BM25Okapi

STOP_WORDS = frozenset("""a an and are as at be by for from in into is it its of on or that the this to we
with our their can which these via using based towards toward""".split())


def tokenize(text):
    """Lower-case word tokens with common function words removed."""
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOP_WORDS]


class KeywordRetriever:
    """BM25 over tokenised documents."""

    def __init__(self, documents):
        self.bm25 = BM25Okapi([tokenize(document) for document in documents])

    def scores(self, query):
        return np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")


class VectorRetriever:
    """Cosine similarity over unit-length embeddings."""

    def __init__(self, embedder, document_vectors):
        self.embedder = embedder
        self.vectors = np.asarray(document_vectors, dtype="float32")

    def embed(self, texts):
        return np.asarray(self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False),
                          dtype="float32")

    def scores(self, query=None, query_vector=None):
        if query_vector is None:
            query_vector = self.embed([query])[0]
        return self.vectors @ query_vector


def top_k(scores, k=10, mask=None):
    """Positions of the k highest scores. With a mask, only allowed positions compete."""
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
    if k <= 0:
        return np.array([], dtype=int)
    best = np.argpartition(-scores, k - 1)[:k]
    return best[np.argsort(-scores[best])]


def min_max(scores):
    low, high = float(scores.min()), float(scores.max())
    return np.zeros_like(scores) if high == low else (scores - low) / (high - low)


def weighted_fusion(keyword_scores, vector_scores, alpha=0.5):
    """Blend the two score lists after scaling each to 0-1. alpha is the weight on vectors."""
    return alpha * min_max(vector_scores) + (1 - alpha) * min_max(keyword_scores)


def reciprocal_rank_fusion(keyword_scores, vector_scores, k=60):
    """Combine by rank instead of score, so the two scales never have to be compared."""
    fused = np.zeros(len(keyword_scores), dtype="float32")
    for scores in (keyword_scores, vector_scores):
        ranks = np.empty(len(scores), dtype="int64")
        ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
        fused += 1.0 / (k + ranks)
    return fused
