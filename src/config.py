"""
Configuration management using pydantic-settings.
Environment variables loaded from .env file.
All sensitive keys stay in .env (never committed to git).
"""

from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # ── Gemini API ──────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2"
    gemini_embedding_dim: int = 768

    # ── Qdrant ──────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "knowledge_base"

    # ── Retrieval ───────────────────────────────────────────────
    chunk_size: int = 600          # tokens (tiktoken cl100k_base) per chunk
    chunk_overlap: float = 0.15    # fraction of chunk_size to overlap (15%)
    top_k_retrieval: int = 20      # candidates from hybrid search
    top_k_rerank: int = 5          # final passages after reranking
    # Layer 1 guard: a low floor that only catches empty/near-random retrieval.
    # Semantic out-of-scope is handled by the LLM's grounded-refusal prompt (layer 2),
    # because cross-encoder confidence overlaps for in/out-of-scope on this corpus.
    confidence_threshold: float = 0.50

    # ── Reranking ───────────────────────────────────────────────
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"

    # ── BM25 Persistence ────────────────────────────────────────
    bm25_index_path: str = "data/bm25_index.pkl"
    bm25_corpus_path: str = "data/bm25_corpus.pkl"

    # ── API ─────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # ── Streamlit ───────────────────────────────────────────────
    streamlit_server_port: int = 8501

    # ── Logging ─────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Feedback ────────────────────────────────────────────────
    feedback_log_path: str = "data/feedback.jsonl"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings singleton.
    Uses lru_cache so .env is only read once, and won't crash
    at import time if .env is missing.
    """
    return Settings()
