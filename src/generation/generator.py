"""
Answer generation using Gemini with RAG context.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Callable

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


def _is_retryable(err: Exception) -> bool:
    """Detect transient Gemini errors worth retrying (rate limit / overload)."""
    s = str(err)
    return any(t in s for t in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))


def _is_abstention(answer: str) -> bool:
    """True when the model declined to answer because the context lacked the info."""
    normalized = (answer or "").strip().rstrip(".").lower()
    return normalized.startswith("i don't have enough information")

class AnswerGenerator:
    """
    Generates answers to user queries using retrieved context.
    Provides citation metadata tracing back to source documents.
    """
    
    # Prompt template designed to prevent hallucinations and enforce citations
    SYSTEM_PROMPT = (
        "You are AnthraSync, an Enterprise Knowledge Assistant.\n"
        "For greetings, thanks, or casual small talk (e.g. \"hi\", \"hello\", "
        "\"thanks\", \"how are you\"), reply naturally and briefly as a friendly "
        "assistant. You do NOT need any context documents for these, and you must "
        "NOT use the refusal message for them.\n"
        "For any request that seeks information or facts, answer using ONLY the "
        "provided context documents. Do not use any outside knowledge and never guess.\n"
        "If the context does not actually contain the information needed to answer "
        "such a question, you MUST reply exactly: "
        "\"I don't have enough information in the knowledge base to answer that.\" "
        "and nothing else. Do not try to be helpful by inventing an answer.\n"
        "When the answer IS in the context, be concise and accurate, format with "
        "Markdown (bullet points, bold where helpful)."
    )

    # Agentic prompt: the model decides whether to consult the knowledge base via
    # the `search_knowledge_base` tool, instead of context being force-fed.
    AGENT_SYSTEM_PROMPT = (
        "You are AnthraSync, an Enterprise Knowledge Assistant.\n"
        "For greetings, thanks, or casual small talk (e.g. \"hi\", \"hello\", "
        "\"thanks\", \"how are you\"), reply naturally and briefly as a friendly "
        "assistant. Do NOT call any tool and do NOT use the refusal message for these.\n"
        "For any request that seeks information or facts about company policies, "
        "guides, compliance, or FAQs, you MUST call the `search_knowledge_base` "
        "tool to fetch relevant passages, and answer using ONLY those passages. "
        "Never use outside knowledge and never guess.\n"
        "If the returned passages do not actually contain the information needed, "
        "you MUST reply exactly: "
        "\"I don't have enough information in the knowledge base to answer that.\" "
        "and nothing else.\n"
        "When the answer IS in the passages, be concise and accurate, format with "
        "Markdown (bullet points, bold where helpful)."
    )

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", max_retries: int = 3):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = genai.Client(api_key=self.api_key)

    def _call_with_retry(self, contents, config: "types.GenerateContentConfig"):
        """Call Gemini, backing off on transient 429/503s."""
        for attempt in range(self.max_retries):
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                if _is_retryable(e) and attempt < self.max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s — recover fast from transient 503s
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{self.max_retries}); "
                        f"retrying in {wait}s"
                    )
                    time.sleep(wait)
                    continue
                raise

    @staticmethod
    def _build_citations(context: List[Dict[str, Any]]):
        """Return (citation_strings, sources) deduplicated by (document, page)."""
        citations: List[str] = []
        sources: List[Dict[str, Any]] = []
        seen = set()
        for doc in context:
            payload = doc.get("payload", {})
            doc_name = payload.get("doc", "Unknown")
            page = payload.get("page", "?")
            citation_str = f"{doc_name} (Page {page})"
            if citation_str not in citations:
                citations.append(citation_str)
            key = (doc_name, page)
            if key not in seen:
                seen.add(key)
                sources.append({"document": doc_name, "page": page})
        return citations, sources

    @staticmethod
    def _format_passages(context: List[Dict[str, Any]]) -> str:
        """Render passages as numbered, labeled blocks for the LLM."""
        if not context:
            return "No relevant passages found."
        parts = []
        for idx, doc in enumerate(context, 1):
            payload = doc.get("payload", {})
            label = f"{payload.get('doc', 'Unknown')} (Page {payload.get('page', '?')})"
            parts.append(f"--- Document [{idx}]: {label} ---\n{payload.get('text', '')}\n")
        return "\n".join(parts)

    def generate(self, query: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer from pre-retrieved context (non-agentic / fixed chain).

        Args:
            query: The user's question.
            context: List of reranked document dicts from the retrieval pipeline.

        Returns:
            Dict containing 'answer', 'citations', and 'sources'.
        """
        if not context:
            return {
                "answer": "I don't have enough context to answer that question.",
                "citations": [],
                "sources": [],
            }

        citations, sources = self._build_citations(context)
        prompt = (
            f"Context Documents:\n\n{self._format_passages(context)}\n\n"
            f"User Question: {query}\n\n"
            f"Answer:"
        )

        logger.info(f"Generating answer using {self.model} with {len(context)} context documents.")

        try:
            response = self._call_with_retry(
                prompt,
                types.GenerateContentConfig(
                    system_instruction=self.SYSTEM_PROMPT,
                    temperature=0.1,
                ),
            )
            answer_text = response.text
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            answer_text = "I encountered an error while trying to generate the answer."

        # If the model abstained, there are no real sources to cite — drop them.
        if _is_abstention(answer_text):
            return {"answer": answer_text, "citations": [], "sources": []}

        return {"answer": answer_text, "citations": citations, "sources": sources}

    @staticmethod
    def _to_contents(
        question: str, history: Optional[List[Dict[str, str]]] = None
    ) -> List["types.Content"]:
        """
        Build a multi-turn `contents` list from prior turns + the new question.

        `history` is a list of {"role": "user"|"assistant", "content": str};
        "assistant" maps to Gemini's "model" role. Only the visible text of each
        turn is carried (not intermediate tool calls), keeping context small while
        preserving conversational continuity for follow-ups.
        """
        contents: List["types.Content"] = []
        for turn in history or []:
            text = (turn.get("content") or "").strip()
            if not text:
                continue
            role = "model" if turn.get("role") in ("assistant", "model") else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        return contents

    def generate_agentic(
        self,
        question: str,
        tool_executor: Callable[[str], List[Dict[str, Any]]],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Agentic generation: the model decides whether to consult the knowledge
        base via a `search_knowledge_base` tool. Greetings / small talk are
        answered directly with no retrieval; factual questions trigger the tool.

        Args:
            question: The raw user message.
            tool_executor: Callable that runs retrieval for a query string and
                returns reranked passage dicts (each with a 'payload' and a
                'rerank_score').
            history: Optional prior conversation turns for multi-turn follow-ups,
                as a list of {"role": "user"|"assistant", "content": str}.

        Returns:
            Dict with 'answer', 'sources', 'citations', 'context_used',
            'used_search', and 'retrieval_ms'.
        """
        captured: List[Dict[str, Any]] = []   # passages the model actually retrieved
        retrieval_s = 0.0                       # seconds spent inside retrieval

        def search_knowledge_base(query: str) -> str:
            """Search the company knowledge base (HR policies, customer policies,
            technical guides, compliance rules, FAQs) and return the most relevant
            passages. Call this ONLY for questions seeking information or facts that
            would live in company documents. Do NOT call it for greetings, thanks,
            or small talk.

            Args:
                query: A focused search query capturing the user's information need.
            """
            nonlocal retrieval_s
            t0 = time.perf_counter()
            try:
                passages = tool_executor(query)
            except Exception as e:
                logger.error(f"Knowledge search failed: {e}")
                passages = []
            finally:
                retrieval_s += time.perf_counter() - t0
            captured.extend(passages)
            return self._format_passages(passages)

        try:
            response = self._call_with_retry(
                self._to_contents(question, history),
                types.GenerateContentConfig(
                    system_instruction=self.AGENT_SYSTEM_PROMPT,
                    temperature=0.1,
                    tools=[search_knowledge_base],  # SDK auto-runs the tool loop
                ),
            )
            answer_text = response.text
        except Exception as e:
            logger.error(f"Agentic generation failed: {e}")
            answer_text = "I encountered an error while trying to generate the answer."

        used_search = bool(captured)
        citations, sources = self._build_citations(captured)
        # On a refusal, drop sources so the UI stays honest.
        if _is_abstention(answer_text):
            citations, sources = [], []

        return {
            "answer": answer_text,
            "sources": sources,
            "citations": citations,
            "context_used": captured,
            "used_search": used_search,
            "retrieval_ms": round(retrieval_s * 1000, 1),
        }

def get_generator() -> AnswerGenerator:
    """Factory using app config."""
    from src.config import get_settings
    settings = get_settings()

    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")

    return AnswerGenerator(
        api_key=settings.gemini_api_key,
        model=settings.gemini_llm_model,
    )
