import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybridsearch.backends import NumpyBackend  # noqa: E402
from hybridsearch.evaluate import category_precision, rank_of, summarise  # noqa: E402
from hybridsearch.fetch import parse_feed  # noqa: E402
from hybridsearch.retrievers import (KeywordRetriever, VectorRetriever, min_max,  # noqa: E402
                                     reciprocal_rank_fusion, tokenize, top_k, weighted_fusion)

FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00001v2</id>
    <title>A Study of
      Index Structures</title>
    <summary>  We study indexes.
    They are useful. </summary>
    <published>2026-01-05T10:00:00Z</published>
    <author><name>Ada Lovelace</name></author><author><name>Alan Turing</name></author>
    <arxiv:primary_category term="cs.DB"/>
  </entry>
</feed>"""

DOCS = ["graph neural networks for molecules", "query optimisation in relational databases",
        "image segmentation with transformers", "transformers for database query generation"]


def test_parse_feed_cleans_whitespace_and_reads_fields():
    assert parse_feed(FEED) == [{
        "arxiv_id": "2601.00001v2", "title": "A Study of Index Structures",
        "abstract": "We study indexes. They are useful.", "authors": "Ada Lovelace, Alan Turing",
        "published": "2026-01-05", "category": "cs.DB"}]


def test_tokenize_lowercases_and_drops_function_words():
    assert tokenize("The Cost of Training: A Study for LLMs") == ["cost", "training", "study", "llms"]


def test_keyword_retriever_prefers_documents_with_the_query_terms():
    scores = KeywordRetriever(DOCS).scores("relational databases")
    assert int(np.argmax(scores)) == 1 and scores[0] == 0


def test_vector_retriever_scores_by_cosine():
    vectors = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype="float32")
    scores = VectorRetriever(None, vectors).scores(query_vector=np.array([1, 0], dtype="float32"))
    assert scores.tolist() == pytest.approx([1.0, 0.0, 0.6])


def test_top_k_orders_results_and_respects_the_mask():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert top_k(scores, 2).tolist() == [1, 3]
    assert top_k(scores, 3, mask=np.array([True, False, True, False])).tolist() == [2, 0]
    assert top_k(scores, 2, mask=np.zeros(4, dtype=bool)).tolist() == []


def test_min_max_handles_constant_scores():
    assert min_max(np.array([2.0, 4.0, 6.0])).tolist() == [0.0, 0.5, 1.0]
    assert min_max(np.array([3.0, 3.0])).tolist() == [0.0, 0.0]


def test_weighted_fusion_endpoints_match_the_single_retrievers():
    keyword, vector = np.array([3.0, 1.0, 2.0]), np.array([0.1, 0.9, 0.5])
    assert int(np.argmax(weighted_fusion(keyword, vector, alpha=0.0))) == 0
    assert int(np.argmax(weighted_fusion(keyword, vector, alpha=1.0))) == 1


def test_reciprocal_rank_fusion_prefers_good_in_both_over_great_in_one():
    # Document 1 is second in both lists; documents 0 and 2 are first in one list and last in the other
    keyword = np.array([10.0, 9.0, 1.0, 5.0, 4.0])
    vector = np.array([0.0, 0.9, 0.95, 0.5, 0.4])
    assert int(np.argmax(reciprocal_rank_fusion(keyword, vector))) == 1


def test_rank_and_summary():
    assert rank_of(np.array([0.2, 0.9, 0.5]), target=2) == 2
    summary = summarise([1, 2, 20], ks=(1, 10))
    assert summary["recall@1"] == pytest.approx(1 / 3) and summary["recall@10"] == pytest.approx(2 / 3)
    assert summary["MRR"] == pytest.approx((1 + 0.5 + 0.05) / 3)


def test_category_precision_can_exclude_the_query_paper():
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    categories = np.array(["a", "a", "b", "a"])
    assert category_precision(scores, categories, "a", k=2) == 1.0
    assert category_precision(scores, categories, "a", k=2, exclude=0) == 0.5


def test_exact_backend_applies_filters_before_ranking():
    vectors = np.eye(4, dtype="float32")
    backend = NumpyBackend(vectors, ["x", "y", "x", "y"], [20260101, 20260201, 20260301, 20260401])
    query = np.array([0.9, 0.8, 0.7, 0.6], dtype="float32")
    assert backend.search(query, k=2) == [0, 1]
    assert backend.search(query, k=2, category="y") == [1, 3]
    assert backend.search(query, k=5, category="x", date_from=20260201) == [2]
