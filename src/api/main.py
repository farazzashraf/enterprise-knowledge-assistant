"""
FastAPI application for AnthraSync.
Exposes REST endpoints for search and answer generation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field, model_validator
import uvicorn

from src.logger import setup_logging, get_logger
from src.config import get_settings
from src.pipeline import RAGPipeline, get_pipeline

logger = get_logger(__name__)

app = FastAPI(
    title="AnthraSync API",
    description="Enterprise Knowledge Assistant API",
    version="1.0.0",
)

# --- Dependency Injection ---
# A single shared RAGPipeline owns the heavy models (reranker, clients) and is
# reused by every endpoint, so they load exactly once.
def get_shared_pipeline() -> RAGPipeline:
    return get_pipeline()

# --- Request/Response Models ---
class SearchRequest(BaseModel):
    query: str = Field(..., description="The user query")
    top_k: int = Field(5, description="Number of results to return")
    rrf_k: int = Field(60, description="RRF constant for hybrid search")

class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]

class Source(BaseModel):
    document: str = Field(..., description="Source document filename")
    page: Any = Field(..., description="Page number within the source document")

class Turn(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The visible text of that turn")

class AskRequest(BaseModel):
    question: str = Field(..., description="The user question")
    top_k: int = Field(5, description="Number of context documents to use")
    # Prior conversation turns for multi-turn follow-ups. The server is stateless:
    # the client owns the conversation and replays it here on each request.
    history: List[Turn] = Field(default_factory=list, description="Prior conversation turns")

    @model_validator(mode="after")
    def _require_question(self):
        # Normalize and reject empty / whitespace-only questions.
        self.question = (self.question or "").strip()
        if not self.question:
            raise ValueError("Question must not be empty.")
        return self

class AskResponse(BaseModel):
    answer: str = Field(..., description="Generated answer grounded in retrieved context")
    sources: List[Source] = Field(default_factory=list, description="Cited source documents")
    confidence: float = Field(..., description="0-1 retrieval confidence (top reranker probability)")
    # Extra (not in the brief example) — full context passages for UI inspection.
    context_used: List[Dict[str, Any]] = Field(default_factory=list)

class FeedbackRequest(BaseModel):
    question: str = Field(..., description="The question that was asked")
    answer: str = Field("", description="The answer that was rated")
    rating: str = Field(..., description="'up' or 'down'")
    comment: str = Field("", description="Optional free-text comment")
    sources: List[Source] = Field(default_factory=list, description="Sources shown with the answer")

    @model_validator(mode="after")
    def _check_rating(self):
        if self.rating not in ("up", "down"):
            raise ValueError("rating must be 'up' or 'down'.")
        return self

# --- Endpoints ---
@app.on_event("startup")
async def startup_event():
    setup_logging(level="INFO")
    logger.info("Starting AnthraSync API...")
    # Pre-load the pipeline (heavy models) so the first request is fast.
    get_shared_pipeline()
    logger.info("Pipeline loaded.")

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(
    request: SearchRequest,
    pipeline: RAGPipeline = Depends(get_shared_pipeline),
):
    """Hybrid search (dense + BM25, RRF) with cross-encoder reranking."""
    logger.info(f"Search request: '{request.query}'")
    try:
        results = pipeline.retrieve(request.query, top_k=request.top_k)
        return SearchResponse(query=request.query, results=results)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(
    request: AskRequest,
    pipeline: RAGPipeline = Depends(get_shared_pipeline),
):
    """
    Answer a question via the full RAG pipeline:
    embed -> hybrid search (dense + BM25, RRF) -> cross-encoder rerank ->
    confidence floor -> grounded generation with citations.
    """
    logger.info(f"Ask request: '{request.question}' ({len(request.history)} prior turns)")
    try:
        history = [{"role": t.role, "content": t.content} for t in request.history]
        result = pipeline.answer(request.question, top_k=request.top_k, history=history)
        return AskResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result.get("sources", [])],
            confidence=result["confidence"],
            context_used=result.get("context_used", []),
        )
    except Exception as e:
        logger.error(f"Ask failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Persist a thumbs up/down on an answer as a JSONL line (for later review)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
        "sources": [s.model_dump() for s in request.sources],
    }
    try:
        path = Path(get_settings().feedback_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Feedback recorded: {request.rating}")
        return {"status": "recorded"}
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")
        raise HTTPException(status_code=500, detail="Could not record feedback.")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
