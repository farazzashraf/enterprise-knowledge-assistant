"""
Confidence scoring and hallucination guardrails for the RAG pipeline.

The cross-encoder reranker emits an unbounded relevance *logit* per passage
(roughly -11 .. +11 for bge-reranker-base). We squash the top passage's logit
through a sigmoid to get a calibrated 0-1 confidence, then abstain when it falls
below the configured floor -- this is what stops the LLM from answering on weak
or empty context.
"""

import math
from typing import List, Dict, Any, Tuple

ABSTAIN_MESSAGE = (
    "I don't have enough information in the knowledge base to answer that."
)


def sigmoid(x: float) -> float:
    """Numerically stable logistic squashing of a reranker logit into 0-1."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def compute_confidence(results: List[Dict[str, Any]]) -> float:
    """
    Confidence = sigmoid(highest rerank logit among the retrieved passages).

    Returns 0.0 when there is nothing to score.
    """
    if not results:
        return 0.0
    top_score = max(float(r.get("rerank_score", 0.0)) for r in results)
    return round(sigmoid(top_score), 4)


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
