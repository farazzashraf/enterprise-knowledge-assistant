"""
Generate 768-dim vectors via Gemini embedding-2.
"""

import time
import math
import logging
from typing import List, Optional
from dataclasses import dataclass

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """A single embedding result with provenance."""
    text: str
    vector: List[float]
    index: int


class GeminiEmbedder:
    """
    Client for Gemini embedding-2 with batching and rate-limit handling.
    Uses the new unified google-genai SDK (>=1.0.0).
    """

    DEFAULT_MODEL = "gemini-embedding-2"
    DEFAULT_DIM = 768
    # Native (un-truncated) output size. gemini-embedding returns unit-norm
    # vectors only at full dimensionality; any smaller (MRL-truncated) output
    # must be L2-normalized by the caller before use in cosine/dot-product search.
    FULL_DIM = 3072
    DEFAULT_BATCH_SIZE = 100          # Gemini supports up to 100 per batch
    DEFAULT_RPM = 15                  # Free tier requests per minute
    DEFAULT_TIMEOUT = 60.0

    # gemini-embedding-2 IGNORES EmbedContentConfig.task_type — task optimization
    # is driven by instruction prefixes baked into the text instead. Asymmetric
    # retrieval: queries get a `task: ... | query:` prefix, documents get a
    # `title: ... | text:` structure. This is a Q&A RAG, so the query task is
    # "question answering". Other valid query tasks: "search result",
    # "fact checking", "code retrieval".
    QUERY_TASK = "question answering"

    @staticmethod
    def _l2_normalize(vector: List[float]) -> List[float]:
        """Scale a vector to unit L2 norm (no-op for a zero vector)."""
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    @staticmethod
    def _format_document(text: str, title: Optional[str] = None) -> str:
        """Wrap a document/chunk in the gemini-embedding-2 document structure."""
        return f"title: {title or 'none'} | text: {text}"

    @classmethod
    def _format_query(cls, text: str) -> str:
        """Wrap a query in the gemini-embedding-2 task instruction structure."""
        return f"task: {cls.QUERY_TASK} | query: {text}"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dim: int = DEFAULT_DIM,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rpm: int = DEFAULT_RPM,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self.sleep_seconds = 60.0 / rpm if rpm > 0 else 0
        self.client = genai.Client(
            api_key=api_key,
            http_options={"timeout": int(timeout * 1000)},
        )

    def embed(
        self,
        texts: List[str],
        mode: str = "document",
        titles: Optional[List[Optional[str]]] = None,
    ) -> List[EmbeddingResult]:
        """
        Embed a list of texts with automatic batching and rate limiting.

        Applies the gemini-embedding-2 instruction prefixes:
          - mode="document": ``title: {title} | text: {content}``
          - mode="query":    ``task: question answering | query: {content}``

        Args:
            texts: List of strings to embed.
            mode: "document" for chunks being indexed, "query" for user
                  questions. Drives the instruction prefix (task_type is
                  deprecated and ignored by gemini-embedding-2).
            titles: Optional per-text titles, only used in document mode.
                    Falls back to "none" when missing.

        Returns:
            List of EmbeddingResult in the same order as input.
        """
        if not texts:
            return []
        if mode not in ("document", "query"):
            raise ValueError(f"mode must be 'document' or 'query', got {mode!r}")

        # Strip, then apply the task-specific instruction prefix.
        if mode == "query":
            clean_texts = [self._format_query(t.strip()) for t in texts]
        else:
            titles = titles or [None] * len(texts)
            clean_texts = [
                self._format_document(t.strip(), title)
                for t, title in zip(texts, titles)
            ]
        # Guard against an all-empty input (prefixes are never empty, so check raw)
        if not any(t.strip() for t in texts):
            return []

        results: List[EmbeddingResult] = []
        total = len(clean_texts)

        # Free tier is 100 RPM for gemini-embedding-2.
        # We need to sleep to stay under this limit.
        sleep_time = 60.0 / 90.0  # Target 90 RPM to be safe

        for i in range(0, total, self.batch_size):
            batch = clean_texts[i : i + self.batch_size]
            batch_len = len(batch)

            logger.info(
                f"Embedding batch {i // self.batch_size + 1}/"
                f"{(total - 1) // self.batch_size + 1} "
                f"({batch_len} texts)"
            )

            try:
                for j, text in enumerate(batch):
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=text,
                        config=types.EmbedContentConfig(
                            # task_type intentionally omitted: ignored by
                            # gemini-embedding-2 (prefixes drive task tuning).
                            output_dimensionality=self.dim,
                        ),
                    )
                    vector = list(response.embeddings[0].values)
                    # MRL-truncated outputs aren't unit-norm; normalize so cosine
                    # /dot-product search behaves correctly. Full-dim is already
                    # normalized, so skip the work there.
                    if self.dim < self.FULL_DIM:
                        vector = self._l2_normalize(vector)
                    results.append(
                        EmbeddingResult(
                            text=text,
                            vector=vector,
                            index=i + j,
                        )
                    )
                    # Pace requests to avoid 429 Too Many Requests — but only
                    # *between* calls. A single live query (or the final chunk of
                    # an ingest) has nothing after it, so don't make it wait.
                    if (i + j) < total - 1:
                        time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Embedding batch failed at offset {i}: {e}")
                raise

        # Ensure original order is preserved
        results.sort(key=lambda r: r.index)
        return results

    def embed_single(
        self, text: str, mode: str = "document", title: Optional[str] = None
    ) -> List[float]:
        """Embed a single text string."""
        results = self.embed([text], mode=mode, titles=[title])
        return results[0].vector if results else []

    def embed_query(self, text: str) -> List[float]:
        """Convenience method for query embedding (question-answering task)."""
        return self.embed_single(text, mode="query")

    def embed_documents(
        self, texts: List[str], titles: Optional[List[Optional[str]]] = None
    ) -> List[List[float]]:
        """
        Returns raw vectors only, matching common vector-store interfaces.

        Pass `titles` (e.g. source document names) to enrich the document
        structure; missing titles fall back to "none".
        """
        results = self.embed(texts, mode="document", titles=titles)
        return [r.vector for r in results]


# --- Factory that wires from config ---

def get_embedder(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> GeminiEmbedder:
    """
    Factory that creates an embedder from explicit args or src.config.
    """
    try:
        from src.config import get_settings
        settings = get_settings()
        _api_key = api_key or settings.gemini_api_key
        _model = model or settings.gemini_embedding_model
        _dim = settings.gemini_embedding_dim
    except ImportError:
        if not api_key:
            raise ValueError("api_key required when src.config is unavailable")
        _api_key = api_key
        _model = model or GeminiEmbedder.DEFAULT_MODEL
        _dim = GeminiEmbedder.DEFAULT_DIM

    return GeminiEmbedder(api_key=_api_key, model=_model, dim=_dim)