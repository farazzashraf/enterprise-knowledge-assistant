"""
Shared RAG pipeline: embed -> hybrid search (dense + BM25, RRF) ->
cross-encoder rerank -> confidence floor -> grounded generation.

Both the FastAPI `/ask` endpoint and the evaluation harness use this single
implementation so retrieval/generation behaviour can never drift between them.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from src.config import get_settings
from src.ingest.embedder import GeminiEmbedder
from src.retrieval.hybrid_search import HybridSearcher, get_hybrid_searcher
from src.retrieval.reranker import CrossEncoderReranker, get_reranker
from src.generation.generator import AnswerGenerator, get_generator
from src.generation.guardrails import compute_confidence, ABSTAIN_MESSAGE

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end retrieval-augmented answering with a hallucination guardrail."""

    def __init__(
        self,
        embedder: GeminiEmbedder,
        searcher: HybridSearcher,
        reranker: CrossEncoderReranker,
        generator: AnswerGenerator,
        confidence_threshold: float = 0.3,
    ):
        self.embedder = embedder
        self.searcher = searcher
        self.reranker = reranker
        self.generator = generator
        self.confidence_threshold = confidence_threshold

    @classmethod
    def from_config(cls) -> "RAGPipeline":
        """Build a pipeline wired entirely from `src.config` / .env."""
        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        embedder = GeminiEmbedder(
            api_key=settings.gemini_api_key,
            model=settings.gemini_embedding_model,
            dim=settings.gemini_embedding_dim,
        )
        return cls(
            embedder=embedder,
            searcher=get_hybrid_searcher(),
            reranker=get_reranker(),
            generator=get_generator(),
            confidence_threshold=settings.confidence_threshold,
        )

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Hybrid retrieval + reranking. Returns the top_k passages (no floor)."""
        query_vector = self.embedder.embed_query(question)
        hybrid = self.searcher.search(
            query=question, query_vector=query_vector, top_k=top_k * 2
        )
        # Keep every candidate through reranking; the confidence floor (a single
        # decision on the top score), not a per-doc cutoff, decides abstention.
        return self.reranker.rerank(
            query=question,
            results=hybrid,
            top_k=top_k,
            confidence_threshold=float("-inf"),
        )

    def answer(
        self,
        question: str,
        top_k: int = 5,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Agentic pipeline: the LLM decides whether to consult the knowledge base.

        - Greetings / small talk → answered directly, no retrieval, no floor.
        - Factual questions → the agent calls the search tool (embed → hybrid
          search → rerank), then answers grounded in the returned passages.
        - `history` (optional prior turns) enables multi-turn follow-ups. The
          server stays stateless — the client supplies the conversation.

        Returns a dict with:
          answer, sources, confidence, context_used, abstained,
          retrieval_ms, generation_ms.
        """
        question = (question or "").strip()
        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
                "confidence": 0.0,
                "context_used": [],
                "abstained": True,
                "retrieval_ms": 0.0,
                "generation_ms": 0.0,
            }

        t0 = time.perf_counter()
        gen = self.generator.generate_agentic(
            question,
            tool_executor=lambda q: self.retrieve(q, top_k=top_k),
            history=history,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        retrieval_ms = gen.get("retrieval_ms", 0.0)
        generation_ms = round(max(total_ms - retrieval_ms, 0.0), 1)

        context_used = gen.get("context_used", [])

        # Confidence is only meaningful when the agent actually retrieved. For a
        # direct conversational reply there is no grounded claim to score.
        if not gen.get("used_search"):
            return {
                "answer": gen["answer"],
                "sources": [],
                "confidence": 1.0,
                "context_used": [],
                "abstained": False,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
            }

        confidence = compute_confidence(context_used)

        # Layer-1 backstop: if retrieval was essentially empty / near-random, abstain
        # regardless of what the model wrote. The grounded prompt is layer 2.
        if confidence < self.confidence_threshold:
            logger.info(
                f"Abstaining (backstop): confidence {confidence} < {self.confidence_threshold}"
            )
            return {
                "answer": ABSTAIN_MESSAGE,
                "sources": [],
                "confidence": confidence,
                "context_used": [],
                "abstained": True,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
            }

        abstained = bool(gen.get("used_search") and not gen.get("sources"))
        return {
            "answer": gen["answer"],
            "sources": gen.get("sources", []),
            "confidence": confidence,
            "context_used": context_used,
            "abstained": abstained,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
        }


_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Process-wide singleton (heavy models load once)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline.from_config()
    return _pipeline
