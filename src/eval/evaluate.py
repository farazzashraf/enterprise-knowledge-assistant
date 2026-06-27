"""
Evaluation harness for the AnthraSync RAG pipeline.

Runs the hand-written test set (data/test_queries.json) through the *exact*
production pipeline (src.pipeline.RAGPipeline) and reports metrics grouped by
category. Two layers of metrics:

  1. Deterministic (no LLM, always available):
     - source_accuracy   : expected source document is among the cited sources
     - keyword_recall     : fraction of `expected_answer_contains` phrases present
     - abstention         : out-of-scope / empty questions correctly refused
     - latency            : retrieval + generation ms

  2. LLM-as-judge (optional, --judge): uses Gemini to score, on answerable
     questions, the *faithfulness* of the answer to the retrieved context
     (a RAGAS-style groundedness check). Costs one LLM call per answered query.

Usage:
    python -m src.eval.evaluate
    python -m src.eval.evaluate --judge
    python -m src.eval.evaluate --test-file data/test_queries.json --top-k 5
"""

import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.logger import setup_logging, get_logger
from src.config import get_settings
from src.pipeline import RAGPipeline

logger = get_logger(__name__)

# Categories where the correct behaviour is to abstain / not produce a grounded answer.
ABSTAIN_CATEGORIES = {"out_of_scope"}

# Phrases that signal the model declined to answer (layer-2 grounded refusal).
REFUSAL_MARKERS = (
    "don't have enough information",
    "do not have enough information",
    "not contain",
    "no information",
    "cannot find",
    "couldn't find",
    "not available in the knowledge base",
    "unable to answer",
)


def _is_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(m in t for m in REFUSAL_MARKERS)


# ─────────────────────────── metric helpers ────────────────────────────

def _cited_documents(result: Dict[str, Any]) -> List[str]:
    return [s.get("document", "") for s in result.get("sources", [])]


def source_hit(result: Dict[str, Any], case: Dict[str, Any]) -> Optional[bool]:
    """True if any expected source document was cited. None if case has no expectation."""
    expected = case.get("expected_source")
    expected_list = case.get("expected_sources")
    targets = []
    if expected:
        targets.append(expected)
    if expected_list:
        targets.extend(expected_list)
    if not targets:
        return None
    cited = _cited_documents(result)
    return any(t in cited for t in targets)


def keyword_recall(result: Dict[str, Any], case: Dict[str, Any]) -> Optional[float]:
    """Fraction of expected phrases that appear (case-insensitive) in the answer."""
    phrases = case.get("expected_answer_contains")
    if not phrases:
        return None
    answer = result.get("answer", "").lower()
    hits = sum(1 for p in phrases if str(p).lower() in answer)
    return hits / len(phrases)


def abstained_correctly(result: Dict[str, Any], case: Dict[str, Any]) -> Optional[bool]:
    """
    For abstain categories / empty questions: did the system refuse?
    Counts EITHER layer: the confidence-floor guardrail OR a textual LLM refusal.
    """
    is_abstain_case = (
        case.get("category") in ABSTAIN_CATEGORIES
        or not str(case.get("question", "")).strip()
    )
    if not is_abstain_case:
        return None
    return bool(result.get("abstained")) or _is_refusal(result.get("answer", ""))


# ─────────────────────────── optional LLM judge ────────────────────────

FAITHFULNESS_PROMPT = (
    "You are a strict RAG evaluator. Given a CONTEXT and an ANSWER, decide whether "
    "every factual claim in the ANSWER is directly supported by the CONTEXT.\n"
    "Respond with ONLY a JSON object: {{\"faithful\": <0.0-1.0>, \"reason\": \"<short>\"}}.\n"
    "1.0 = fully grounded, 0.0 = unsupported/contradicted.\n\n"
    "CONTEXT:\n{context}\n\nANSWER:\n{answer}\n"
)


def judge_faithfulness(client, model: str, result: Dict[str, Any]) -> Optional[float]:
    """LLM-as-judge groundedness score for an answered query. None if not applicable."""
    if result.get("abstained") or not result.get("context_used"):
        return None
    context = "\n\n".join(
        c.get("payload", {}).get("text", "") for c in result["context_used"]
    )
    prompt = FAITHFULNESS_PROMPT.format(context=context, answer=result.get("answer", ""))
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = (resp.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
        return float(json.loads(text).get("faithful"))
    except Exception as e:  # judge is best-effort; never fail the run
        logger.warning(f"Judge failed: {e}")
        return None


# ─────────────────────────── runner ────────────────────────────

def run_eval(
    test_file: str = "data/test_queries.json",
    top_k: int = 5,
    use_judge: bool = False,
    sleep: float = 2.0,
) -> Dict[str, Any]:
    setup_logging(level="WARNING")  # quiet the per-request httpx noise
    settings = get_settings()

    cases = json.loads(Path(test_file).read_text(encoding="utf-8"))["test_cases"]
    logger.info(f"Loaded {len(cases)} test cases from {test_file}")

    pipeline = RAGPipeline.from_config()

    judge_client = None
    if use_judge:
        from google import genai
        judge_client = genai.Client(api_key=settings.gemini_api_key)

    rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases):
        if i and sleep:
            time.sleep(sleep)  # pace requests under the per-minute quota
        q = case.get("question", "")
        result = pipeline.answer(q, top_k=top_k)
        row = {
            "id": case["id"],
            "category": case.get("category"),
            "question": q,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "abstained": result["abstained"],
            "sources": _cited_documents(result),
            "latency_ms": round(result["retrieval_ms"] + result["generation_ms"], 1),
            "source_hit": source_hit(result, case),
            "keyword_recall": keyword_recall(result, case),
            "correct_abstention": abstained_correctly(result, case),
        }
        if use_judge:
            row["faithfulness"] = judge_faithfulness(
                judge_client, settings.gemini_llm_model, result
            )
        rows.append(row)
        _print_row(row)

    summary = _summarize(rows)
    _print_summary(summary)
    _write_report(rows, summary, top_k, use_judge)
    return {"rows": rows, "summary": summary}


# ─────────────────────────── reporting ────────────────────────────

def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def rate(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None

    summary = {
        "n_cases": len(rows),
        "source_accuracy": rate("source_hit"),
        "keyword_recall": _mean([r["keyword_recall"] for r in rows]),
        "abstention_accuracy": rate("correct_abstention"),
        "avg_latency_ms": _mean([r["latency_ms"] for r in rows]),
        "avg_confidence": _mean([r["confidence"] for r in rows]),
    }
    if any("faithfulness" in r for r in rows):
        summary["faithfulness"] = _mean([r.get("faithfulness") for r in rows])
    return summary


def _fmt(v):
    # ASCII-only: the console may be cp1252 (Windows) and can't encode ✓/✗.
    if v is None:
        return " -  "
    if isinstance(v, bool):
        return "yes " if v else "no  "
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _print_row(row: Dict[str, Any]) -> None:
    print(
        f"[{row['id']}] {row['category']:<14} "
        f"src={_fmt(row['source_hit'])} kw={_fmt(row['keyword_recall'])} "
        f"abst={_fmt(row['correct_abstention'])} conf={row['confidence']:.2f} "
        f"{row['latency_ms']:.0f}ms"
    )


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<22}: {v}")
    print("=" * 60)


def _write_report(rows, summary, top_k, use_judge) -> None:
    out_json = Path("data/eval_results.json")
    out_json.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )

    lines = [
        "# Evaluation Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M')} | top_k={top_k} | "
        f"LLM judge={'on' if use_judge else 'off'}_",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    labels = {
        "n_cases": "Test cases",
        "source_accuracy": "Source accuracy (retrieval)",
        "keyword_recall": "Answer keyword recall",
        "abstention_accuracy": "Abstention accuracy (hallucination guard)",
        "avg_latency_ms": "Avg latency (ms)",
        "avg_confidence": "Avg confidence",
        "faithfulness": "Faithfulness (LLM judge)",
    }
    for k, v in summary.items():
        lines.append(f"| {labels.get(k, k)} | {v} |")

    lines += [
        "",
        "## Per-question results",
        "",
        "| ID | Category | Source hit | Keyword recall | Correct abstention | Confidence | Latency |",
        "|----|----------|:----------:|:--------------:|:------------------:|:----------:|:-------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {_md(r['source_hit'])} | "
            f"{_md(r['keyword_recall'])} | {_md(r['correct_abstention'])} | "
            f"{r['confidence']:.2f} | {r['latency_ms']:.0f}ms |"
        )
    Path("eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote eval_report.md and {out_json}")


def _md(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✅" if v else "❌"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the AnthraSync RAG pipeline.")
    parser.add_argument("--test-file", default="data/test_queries.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--judge", action="store_true",
        help="Enable Gemini LLM-as-judge faithfulness scoring (extra API calls).",
    )
    parser.add_argument(
        "--sleep", type=float, default=2.0,
        help="Seconds to pause between questions to respect rate limits.",
    )
    args = parser.parse_args()
    run_eval(
        test_file=args.test_file, top_k=args.top_k,
        use_judge=args.judge, sleep=args.sleep,
    )


if __name__ == "__main__":
    main()
