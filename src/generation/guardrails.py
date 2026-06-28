"""
Confidence scoring and hallucination guardrails for the RAG pipeline.

`sentence_transformers.CrossEncoder.predict()` already applies a sigmoid for the
single-label bge-reranker, so each passage's `rerank_score` is a calibrated 0-1
relevance probability. Confidence is simply the top passage's score; we abstain
when it falls below the configured floor -- this is what stops the LLM from
answering on weak or empty context.

(Historical note: an earlier version applied a *second* sigmoid here, which
crushed every score into [0.50, 0.73] and destroyed the in/out-of-scope signal.
The reranker probability is used directly now.)
"""

from typing import List, Dict, Any, Tuple

ABSTAIN_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that."
)


def compute_confidence(results: List[Dict[str, Any]]) -> float:
    """
    Confidence = highest reranker probability among the retrieved passages.

    `rerank_score` is already a 0-1 probability (CrossEncoder.predict applies a
    sigmoid for the 1-label bge-reranker), so we use it directly -- do NOT squash
    it again. Returns 0.0 when there is nothing to score.
    """
    if not results:
        return 0.0
    return round(max(float(r.get("rerank_score", 0.0)) for r in results), 4)


def passes_floor(confidence: float, threshold: float) -> bool:
    """True when the answer is allowed to be generated."""
    return confidence >= threshold


def evaluate(
    results: List[Dict[str, Any]], threshold: float
) -> Tuple[float, bool]:
    """
    Convenience: returns (confidence, should_answer) for a reranked result set.
    """
    confidence = compute_confidence(results)
    return confidence, passes_floor(confidence, threshold)
