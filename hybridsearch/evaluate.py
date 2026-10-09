"""Known-item evaluation: does a paper's title bring back that paper's abstract?"""
import numpy as np

from .retrievers import top_k


def rank_of(scores, target):
    """1-based rank of the target document (ties counted against it)."""
    return int((scores >= scores[target]).sum())


def summarise(ranks, ks=(1, 5, 10)):
    ranks = np.asarray(ranks)
    summary = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    summary["MRR"] = float((1.0 / ranks).mean())
    return summary


def category_precision(scores, categories, target_category, k=10, exclude=None):
    """Share of the top k results that are in the query paper's own category."""
    if exclude is not None:
        scores = scores.copy()
        scores[exclude] = -np.inf
    return float((categories[top_k(scores, k)] == target_category).mean())
