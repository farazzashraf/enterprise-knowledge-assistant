"""
Script to test the retrieval pipeline end-to-end.

Usage:
    python test_retrieval.py "What is the company policy on remote work?"
"""

import sys
import time
import argparse
import logging
from src.logger import setup_logging, get_logger
from src.config import get_settings
from src.ingest.embedder import GeminiEmbedder
from src.retrieval.hybrid_search import get_hybrid_searcher
from src.retrieval.reranker import get_reranker

logger = get_logger(__name__)

def test_query(query: str, top_k: int = 5, rrf_k: int = 60):
    setup_logging(level="INFO")
    settings = get_settings()
    
    logger.info("=" * 60)
    logger.info(f"TESTING RETRIEVAL PIPELINE")
    logger.info(f"Query: '{query}'")
    logger.info("=" * 60)
    
    start = time.time()
    
    # 1. Embed query
    logger.info("1. Embedding query with Gemini...")
    embedder = GeminiEmbedder(
        api_key=settings.gemini_api_key,
        model=settings.gemini_embedding_model,
        dim=settings.gemini_embedding_dim,
    )
    query_vector = embedder.embed_query(query)
    
    # 2. Hybrid search
    logger.info("2. Executing hybrid search (Qdrant + BM25) with RRF...")
    searcher = get_hybrid_searcher()
    hybrid_results = searcher.search(
        query=query,
        query_vector=query_vector,
        top_k=top_k * 2,  # Fetch more for reranking
        rrf_k=rrf_k,
    )
    
    logger.info(f"   Found {len(hybrid_results)} candidates.")
    
    # 3. Rerank
    logger.info("3. Reranking candidates with Cross-Encoder...")
    reranker = get_reranker()
    final_results = reranker.rerank(
        query=query,
        results=hybrid_results,
        top_k=top_k,
        confidence_threshold=-10.0, # Accept all for testing
    )
    
    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.2f}s")
    
    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("TOP RESULTS:")
    logger.info("=" * 60)
    
    for i, res in enumerate(final_results, 1):
        payload = res.get("payload", {})
        doc = payload.get("doc", "Unknown")
        page = payload.get("page", "?")
        text = payload.get("text", "")[:150] + "..."
        score = res.get("rerank_score", 0.0)
        
        print(f"[{i}] Score: {score:+.2f} | Doc: {doc} (Page {page})")
        print(f"    {text}")
        print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the retrieval pipeline.")
    parser.add_argument("query", type=str, help="The query string")
    parser.add_argument("--top_k", type=int, default=3, help="Number of final results")
    args = parser.parse_args()
    
    test_query(args.query, top_k=args.top_k)
