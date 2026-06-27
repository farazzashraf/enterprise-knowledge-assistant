"""
Upload chunks + embeddings to Qdrant Cloud and serialize BM25 index to disk.
"""

import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    HnswConfigDiff,
)

from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class QdrantIndexer:
    """
    Manages Qdrant collection creation and dense vector uploads.
    Supports hybrid search setup (dense + sparse vectors).
    """

    DEFAULT_COLLECTION = "knowledge_base"
    DEFAULT_DIM = 768
    DEFAULT_DISTANCE = Distance.COSINE

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION,
        vector_dim: int = DEFAULT_DIM,
        distance: Distance = DEFAULT_DISTANCE,
    ):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.distance = distance

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=60,
        )

    def _collection_exists(self) -> bool:
        """Check if the collection already exists."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return False

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create the Qdrant collection with dense vector configuration.
        If recreate=True, delete existing collection first.
        """
        if recreate and self._collection_exists():
            logger.warning(f"Deleting existing collection '{self.collection_name}'")
            self.client.delete_collection(self.collection_name)

        if not self._collection_exists():
            logger.info(
                f"Creating Qdrant collection '{self.collection_name}' "
                f"({self.vector_dim}d, {self.distance.name})"
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_dim,
                    distance=self.distance,
                    on_disk=True,  # Persist vectors to disk
                ),
                optimizers_config={"indexing_threshold": 20000},
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                ),
            )
            logger.info("Collection created successfully")
        else:
            logger.info(f"Collection '{self.collection_name}' already exists")

    def upload_points(
        self,
        points: List[PointStruct],
        batch_size: int = 100,
    ) -> None:
        """
        Upload points to Qdrant in batches.
        """
        if not points:
            logger.warning("No points to upload")
            return

        total = len(points)
        logger.info(f"Uploading {total} points to Qdrant...")

        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )
                logger.info(f"Uploaded batch {i // batch_size + 1}/{(total - 1) // batch_size + 1}")
            except Exception as e:
                logger.error(f"Failed to upload batch at offset {i}: {e}")
                raise

        logger.info("Upload complete")

    def build_points(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> List[PointStruct]:
        """
        Build Qdrant PointStruct objects from chunks and embeddings.

        Args:
            chunks: List of dicts with keys 'text', 'metadata'.
            embeddings: Parallel list of 768-dim vectors.

        Returns:
            List of PointStruct ready for upload.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        points: List[PointStruct] = []
        for idx, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            if len(vec) != self.vector_dim:
                raise ValueError(
                    f"Embedding dim mismatch at idx {idx}: "
                    f"expected {self.vector_dim}, got {len(vec)}"
                )

            metadata = chunk.get("metadata", {})
            payload = {
                "text": chunk["text"],
                "doc": metadata.get("doc", "unknown"),
                "page": metadata.get("page", 1),
                "chunk_idx": metadata.get("chunk_idx", 0),
                "chunk_total": metadata.get("chunk_total", 1),
                **metadata,  # Merge any additional metadata
            }

            points.append(
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload=payload,
                )
            )

        return points

    def get_collection_info(self) -> Dict[str, Any]:
        """Return collection statistics."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
            "points_count": getattr(info, "points_count", 0),
            "status": str(info.status),
        }


class BM25Indexer:
    """
    Builds and persists a BM25 sparse index from chunk texts.
    """

    def __init__(
        self,
        index_path: str = "data/bm25_index.pkl",
        corpus_path: str = "data/bm25_corpus.pkl",
    ):
        self.index_path = Path(index_path)
        self.corpus_path = Path(corpus_path)
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[str] = []
        self.ids: List[int] = []

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization for BM25."""
        return text.lower().split()

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Build BM25 index from chunk texts.

        Args:
            chunks: List of dicts with 'text' and metadata. 'id' used for mapping.
        """
        if not chunks:
            logger.warning("No chunks provided for BM25 indexing")
            return

        self.corpus = [c["text"] for c in chunks]
        self.ids = [c.get("id", i) for i, c in enumerate(chunks)]
        tokenized = [self._tokenize(t) for t in self.corpus]

        logger.info(f"Building BM25 index from {len(tokenized)} documents...")
        self.bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built")

    def save(self) -> None:
        """Serialize BM25 index and corpus to disk."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call .build() first.")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, "wb") as f:
            pickle.dump(self.bm25, f)
        with open(self.corpus_path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "ids": self.ids}, f)

        logger.info(f"BM25 index saved to {self.index_path}")
        logger.info(f"BM25 corpus saved to {self.corpus_path}")

    def load(self) -> None:
        """Load BM25 index and corpus from disk."""
        if not self.index_path.exists() or not self.corpus_path.exists():
            raise FileNotFoundError(
                f"BM25 files not found: {self.index_path} or {self.corpus_path}"
            )

        with open(self.index_path, "rb") as f:
            self.bm25 = pickle.load(f)
        with open(self.corpus_path, "rb") as f:
            data = pickle.load(f)
            self.corpus = data["corpus"]
            self.ids = data["ids"]

        logger.info(f"BM25 index loaded from {self.index_path}")

    def search(self, query: str, top_k: int = 20) -> List[tuple]:
        """
        Search BM25 index and return (doc_id, score) tuples.

        Returns:
            List of (id, score) sorted by relevance descending.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not loaded. Call .load() or .build() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.ids[idx], float(scores[idx])))

        return results


class HybridIndexer:
    """
    Orchestrates both dense (Qdrant) and sparse (BM25) indexing.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "knowledge_base",
        vector_dim: int = 768,
        bm25_index_path: str = "data/bm25_index.pkl",
        bm25_corpus_path: str = "data/bm25_corpus.pkl",
    ):
        self.qdrant = QdrantIndexer(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            vector_dim=vector_dim,
        )
        self.bm25 = BM25Indexer(
            index_path=bm25_index_path,
            corpus_path=bm25_corpus_path,
        )

    def index(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        recreate: bool = False,
    ) -> None:
        """
        Full indexing pipeline: Qdrant dense + BM25 sparse.

        Args:
            chunks: List of dicts with 'text' and 'metadata'.
            embeddings: Parallel list of dense vectors.
            recreate: If True, drop and recreate Qdrant collection.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")

        logger.info(f"Starting hybrid indexing of {len(chunks)} chunks...")

        # 1. Dense vectors → Qdrant
        self.qdrant.create_collection(recreate=recreate)
        points = self.qdrant.build_points(chunks, embeddings)
        self.qdrant.upload_points(points)

        # 2. Sparse keywords → BM25
        # Assign sequential IDs matching Qdrant point IDs
        chunks_with_ids = [
            {**chunk, "id": i} for i, chunk in enumerate(chunks)
        ]
        self.bm25.build(chunks_with_ids)
        self.bm25.save()

        logger.info("Hybrid indexing complete")

    def get_stats(self) -> Dict[str, Any]:
        """Return stats for both indexes."""
        return {
            "qdrant": self.qdrant.get_collection_info(),
            "bm25": {
                "index_path": str(self.bm25.index_path),
                "corpus_path": str(self.bm25.corpus_path),
                "documents": len(self.bm25.corpus) if self.bm25.corpus else 0,
            },
        }


# --- Factory from config ---

def get_indexer() -> HybridIndexer:
    """Factory that wires from src.config."""
    try:
        from src.config import get_settings
        settings = get_settings()
        return HybridIndexer(
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key or None,
            collection_name=settings.qdrant_collection_name,
            vector_dim=settings.gemini_embedding_dim,
            bm25_index_path=settings.bm25_index_path,
            bm25_corpus_path=settings.bm25_corpus_path,
        )
    except ImportError:
        logger.warning("src.config not available, using defaults")
        return HybridIndexer()