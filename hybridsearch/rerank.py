"""Second-stage reranking: rescore the top candidates with a cross-encoder."""
import numpy as np

from .retrievers import top_k

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def rerank_scores(scores, candidates, candidate_scores):
    """Scores in which `candidates` are reordered by `candidate_scores` and placed above everything else.

    Documents outside the candidate set keep their first-stage order below them, so ranks
    and metrics can be computed over the whole collection exactly as before.
    """
    candidate_scores = np.asarray(candidate_scores, dtype="float32")
    spread = float(candidate_scores.max() - candidate_scores.min()) or 1.0
    out = np.array(scores, dtype="float32")
    out[candidates] = out.max() + 1 + (candidate_scores - candidate_scores.min()) / spread
    return out


class CrossEncoderReranker:
    """Reads the query and each candidate together, which is slower but more precise than comparing vectors."""

    def __init__(self, documents, model=None, depth=20):
        if model is None:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(DEFAULT_RERANKER)
        self.model = model
        self.documents = documents
        self.depth = depth

    def rerank(self, query, scores):
        candidates = top_k(scores, self.depth)
        pairs = [(query, self.documents[i]) for i in candidates]
        return rerank_scores(scores, candidates, self.model.predict(pairs, batch_size=64, show_progress_bar=False))
