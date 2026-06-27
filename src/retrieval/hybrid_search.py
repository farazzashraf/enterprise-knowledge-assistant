"""
Hybrid search implementation (dense + sparse) over Qdrant and BM25.
Uses Reciprocal Rank Fusion (RRF) to combine results.
"""

import logging
from typing import List, Dict, Any

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class HybridSearcher:
    """
    Performs hybrid search by combining:
    1. Dense vector search via Qdrant
    2. Sparse keyword search via BM25
    3. RRF (Reciprocal Rank Fusion) for merging
    """
    
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str | None = None,
        collection_name: str = "knowledge_base",
        bm25_index_path: str = "data/bm25_index.pkl",
        bm25_corpus_path: str = "data/bm25_corpus.pkl",
    ):
        self.collection_name = collection_name
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60,
        )
        
        # We load BM25 at init time so it's ready in memory
        from src.ingest.indexer import BM25Indexer
        self.bm25 = BM25Indexer(
            index_path=bm25_index_path,
            corpus_path=bm25_corpus_path,
        )
        try:
            self.bm25.load()
            self._bm25_loaded = True
            logger.info("BM25 index loaded successfully for retrieval")
        except FileNotFoundError:
            self._bm25_loaded = False
            logger.warning("BM25 index not found. Keyword search will be skipped.")

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 20,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Execute hybrid search and merge using RRF.
        
        Args:
            query: The raw string query (for BM25).
            query_vector: The embedded query vector (for Qdrant).
            top_k: How many results to return.
            rrf_k: The k parameter for Reciprocal Rank Fusion.
        """
        # 1. Dense Search (Qdrant) — query_points is the current API
        # (the older .search() was removed in qdrant-client 1.12+).
        dense_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k * 2,  # Fetch more to get good overlap for RRF
        ).points
        
        # 2. Sparse Search (BM25)
        sparse_results = []
        if self._bm25_loaded:
            sparse_results = self.bm25.search(query, top_k=top_k * 2)

        # 3. RRF Merging
        # We need a unified dict to track scores for each point ID
        # point ID -> { 'payload': {...}, 'rrf_score': 0.0 }
        merged: Dict[int, Dict[str, Any]] = {}
        
        # Process dense ranks
        for rank, hit in enumerate(dense_results):
            point_id = hit.id
            if point_id not in merged:
                merged[point_id] = {
                    "id": point_id,
                    "payload": hit.payload or {},
                    "rrf_score": 0.0,
                    "dense_score": hit.score,
                    "sparse_score": 0.0,
                }
            merged[point_id]["rrf_score"] += 1.0 / (rrf_k + rank + 1)
            
        # Process sparse ranks
        # BM25 search returns list of (id, score) tuples.
        # Sparse-only hits (not found by dense search) need their payloads fetched
        # from Qdrant. Batch them into a SINGLE retrieve() rather than one network
        # round-trip per id — the latter is an N+1 that dominates latency against a
        # remote (cloud) Qdrant.
        missing_ids = [
            point_id for point_id, _ in sparse_results if point_id not in merged
        ]
        payloads: Dict[int, Dict[str, Any]] = {}
        if missing_ids:
            try:
                points = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=missing_ids,
                )
                payloads = {p.id: (p.payload or {}) for p in points}
            except Exception:
                logger.warning("Failed to fetch payloads for sparse-only hits", exc_info=True)

        for rank, (point_id, score) in enumerate(sparse_results):
            if point_id not in merged:
                merged[point_id] = {
                    "id": point_id,
                    "payload": payloads.get(point_id, {}),
                    "rrf_score": 0.0,
                    "dense_score": 0.0,
                    "sparse_score": score,
                }
            merged[point_id]["rrf_score"] += 1.0 / (rrf_k + rank + 1)
            
        # Sort by RRF score descending
        sorted_results = sorted(merged.values(), key=lambda x: x["rrf_score"], reverse=True)
        
        # Return top_k
        return sorted_results[:top_k]

def get_hybrid_searcher() -> HybridSearcher:
    """Factory using app config."""
    from src.config import get_settings
    settings = get_settings()
    return HybridSearcher(
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=settings.qdrant_api_key or None,
        collection_name=settings.qdrant_collection_name,
        bm25_index_path=settings.bm25_index_path,
        bm25_corpus_path=settings.bm25_corpus_path,
    )
