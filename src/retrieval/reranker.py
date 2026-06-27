"""
Cross-encoder reranking for refining hybrid search results.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """
    Reranks documents using a Cross-Encoder model.
    By default, uses BAAI/bge-reranker-base.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None):
        self.model_name = model_name
        self.device = device or self._auto_device()
        self.model = None
        self._load_model()

    @staticmethod
    def _auto_device() -> str:
        """Use the GPU when a CUDA build of torch can see one; else CPU."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _load_model(self):
        """Lazy load the sentence-transformers cross-encoder."""
        logger.info(f"Loading Cross-Encoder model: {self.model_name} on {self.device}")
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Cross-Encoder loaded successfully on {self.device}")
        except ImportError:
            logger.error("sentence-transformers is not installed. Reranking will fail.")
            raise
            
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        confidence_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Rerank a list of retrieved documents against the query.
        
        Args:
            query: The user's question.
            results: List of result dicts containing at least 'payload' with 'text'.
            top_k: Number of results to return after reranking.
            confidence_threshold: Minimum cross-encoder score to keep a document.
            
        Returns:
            Sorted list of result dicts, with updated 'rerank_score'.
        """
        if not results or self.model is None:
            return []
            
        # Prepare pairs for cross-encoder (query, document_text)
        pairs = []
        for res in results:
            text = res.get("payload", {}).get("text", "")
            pairs.append((query, text))
            
        # Calculate scores
        scores = self.model.predict(pairs)
        
        # Attach scores and filter
        reranked = []
        for res, score in zip(results, scores):
            res["rerank_score"] = float(score)
            if score >= confidence_threshold:
                reranked.append(res)
                
        # Sort descending by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked[:top_k]

def get_reranker() -> CrossEncoderReranker:
    """Factory using app config."""
    from src.config import get_settings
    settings = get_settings()
    return CrossEncoderReranker(model_name=settings.reranker_model)
