"""The same filtered vector search on three engines, for timing and agreement checks."""
import time

import numpy as np

from .retrievers import top_k


class NumpyBackend:
    """Exact search: filter with a boolean mask, then score what is left."""
    name = "NumPy (exact)"

    def __init__(self, vectors, categories, dates):
        self.vectors, self.categories, self.dates = vectors, np.asarray(categories), np.asarray(dates)

    def search(self, query_vector, k=10, category=None, date_from=None):
        mask = np.ones(len(self.vectors), dtype=bool)
        if category is not None:
            mask &= self.categories == category
        if date_from is not None:
            mask &= self.dates >= date_from
        return top_k(self.vectors @ query_vector, k, mask).tolist()


class ChromaBackend:
    """ChromaDB with an HNSW index and metadata filters."""
    name = "ChromaDB (HNSW)"

    def __init__(self, vectors, categories, dates):
        import chromadb
        self.collection = chromadb.EphemeralClient().create_collection(
            "papers", metadata={"hnsw:space": "cosine"})
        ids = [str(i) for i in range(len(vectors))]
        metadata = [{"category": c, "date": int(d)} for c, d in zip(categories, dates)]
        for start in range(0, len(ids), 2000):
            end = start + 2000
            self.collection.add(ids=ids[start:end], embeddings=vectors[start:end].tolist(), metadatas=metadata[start:end])

    def search(self, query_vector, k=10, category=None, date_from=None):
        conditions = []
        if category is not None:
            conditions.append({"category": category})
        if date_from is not None:
            conditions.append({"date": {"$gte": int(date_from)}})
        where = None if not conditions else conditions[0] if len(conditions) == 1 else {"$and": conditions}
        result = self.collection.query(query_embeddings=[query_vector.tolist()], n_results=k, where=where, include=[])
        return [int(i) for i in result["ids"][0]]


class QdrantBackend:
    """Qdrant in local (in-process) mode with payload filters."""
    name = "Qdrant (local mode)"

    def __init__(self, vectors, categories, dates):
        from qdrant_client import QdrantClient, models
        self.models = models
        self.client = QdrantClient(":memory:")
        self.client.create_collection("papers", vectors_config=models.VectorParams(
            size=vectors.shape[1], distance=models.Distance.COSINE))
        points = [models.PointStruct(id=i, vector=vectors[i].tolist(),
                                     payload={"category": categories[i], "date": int(dates[i])})
                  for i in range(len(vectors))]
        for start in range(0, len(points), 1000):
            self.client.upsert("papers", points[start:start + 1000])

    def search(self, query_vector, k=10, category=None, date_from=None):
        m, conditions = self.models, []
        if category is not None:
            conditions.append(m.FieldCondition(key="category", match=m.MatchValue(value=category)))
        if date_from is not None:
            conditions.append(m.FieldCondition(key="date", range=m.Range(gte=int(date_from))))
        result = self.client.query_points("papers", query=query_vector.tolist(), limit=k,
                                          query_filter=m.Filter(must=conditions) if conditions else None)
        return [point.id for point in result.points]


def benchmark(backend, query_vectors, exact, k=10, **filters):
    """Median latency in milliseconds, and how much of the exact top k the backend returns."""
    timings, overlap = [], []
    for query_vector in query_vectors:
        start = time.perf_counter()
        found = backend.search(query_vector, k, **filters)
        timings.append((time.perf_counter() - start) * 1000)
        truth = exact.search(query_vector, k, **filters)
        overlap.append(len(set(found) & set(truth)) / max(len(truth), 1))
    return {"median ms": float(np.median(timings)), "p95 ms": float(np.percentile(timings, 95)),
            "agreement with exact": float(np.mean(overlap))}
