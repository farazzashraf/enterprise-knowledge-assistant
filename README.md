# AnthraSync — Enterprise Knowledge Assistant

An agentic **Retrieval-Augmented Generation (RAG)** assistant that answers natural-language
questions from a company's internal documents (HR policies, product/technical guides,
customer FAQs, compliance) — with **source citations**, a **confidence score**, and an
honest *"I don't know"* when the answer isn't in the knowledge base.

> 📐 **Full design document:** [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) — architecture,
> data flow, every component, design decisions, evaluation, scalability, and limitations in one place.
> This README is the **setup & run guide**.

---

## What it does

```
Documents (PDF/DOCX/TXT)                Live question + chat history
      │  (one-time ingestion)                    │
      ▼                                          ▼
 load → chunk → embed → index            Gemini agent: "do I need the KB?"
 (Qdrant + BM25)                          ├─ greeting/chit-chat → answer directly
                                          └─ real question → search tool:
                                                hybrid search (dense + BM25, RRF)
                                                → cross-encoder re-rank
                                                → grounded answer + sources + confidence
```

### Key features
- **Hybrid retrieval** — dense (semantic) + BM25 (keyword) fused with Reciprocal Rank Fusion.
- **Cross-encoder re-ranking** (`bge-reranker-base`) — the biggest accuracy boost.
- **Agentic tool use** — the LLM decides whether to search; greetings skip retrieval entirely.
- **Conversation memory** — multi-turn follow-ups (client-held history; server stays stateless).
- **Hallucination guardrails** — grounded-only prompt + a confidence-floor backstop.
- **Source citations** — every grounded answer lists `document` + `page`.
- **User feedback** — 👍/👎 on each answer, logged for later review.
- **Evaluation harness** — 15 test questions, retrieval/answer metrics, optional LLM-judge.
- **API + UI + Docker** — FastAPI, Streamlit, and a one-command Compose stack.

### Tech stack
| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| LLM | Google **Gemini 2.5 Flash** |
| Embeddings | **Gemini embedding-2** (768-dim, task-prefixed, L2-normalized) |
| Vector DB | **Qdrant** (local via Docker, or Qdrant Cloud) |
| Keyword search | **BM25** (`rank-bm25`) |
| Re-ranker | **`BAAI/bge-reranker-base`** cross-encoder (runs locally) |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, token-based (tiktoken) |
| API / UI | FastAPI + Uvicorn / Streamlit |
| Packaging | Docker + docker-compose |

---

## Architecture

```
                         ┌─────────────────────────────────────────────┐
  INGESTION (offline)    │  ANSWERING (live, agentic)                  │
  ingest_documents.py    │  src/pipeline.py · src/api/main.py          │
                         │                                             │
  PDF/DOCX/TXT           │   Streamlit UI (app.py)  ◀── employee       │
      │ load (loaders)   │        │ question + chat history            │
      ▼                  │        ▼                                    │
  chunk (~600 tok)       │   FastAPI  /ask                             │
      │ (chunker)        │        ▼                                    │
      ▼                  │   Gemini agent ── "need the KB?" ──┐        │
  embed (Gemini 768-d)   │        │ greeting               yes│        │
      │ (embedder)       │        │ → reply directly          ▼        │
      ▼                  │        │            search_knowledge_base   │
  ┌───────────┐          │        │                    │ embed query  │
  │  Qdrant   │◀─────────┼────────┼──── hybrid search ─┤              │
  │ (dense)   │  index   │        │   (dense + BM25, RRF)              │
  ├───────────┤          │        │            │                      │
  │ BM25 .pkl │◀─────────┘        │            ▼                      │
  └───────────┘                   │   cross-encoder rerank (top 5)    │
                                  │            │                      │
                                  │            ▼                      │
                                  │   confidence floor (guardrail)    │
                                  │            │                      │
                                  │            ▼                      │
                                  │   grounded answer + sources + conf│
                                  └─────────────────────────────────────┘

         Gemini API (embeddings + generation) ── over the network
```

A full, annotated **Mermaid diagram** (with the data-flow and component tables) is in
[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) §2–§4.

---

## Technical decisions

Key choices and the reasoning behind them (full rationale in
[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) §9):

| Decision | Why |
|---|---|
| **Hybrid retrieval (dense + BM25, fused with RRF)** | Semantic search misses exact acronyms/IDs; keyword search misses paraphrases. RRF merges them without reconciling two incompatible score scales. |
| **Cross-encoder re-ranking** (`bge-reranker-base`) | Biggest accuracy boost — it reads question + passage *together*. Runs only on the ~10 fused candidates, so cost stays flat as the corpus grows. Local & free. |
| **Agentic tool use over a fixed RAG chain** | The LLM decides whether to search, so greetings/small talk skip retrieval (no pointless search, no false refusal) and the model can rewrite the query. Native `google-genai` function calling, no LangChain. Trade-off: ~2 model calls per factual question. |
| **Two anti-hallucination nets** | A deterministic confidence floor (top reranker probability < `0.10` → abstain) plus a strict grounded-only prompt. Neither alone separates answerable from out-of-scope cleanly, so both run; abstention is credited if either fires. |
| **Token-based chunking (~600 tokens)** | The embedder has a *token* limit, so tokens (tiktoken `cl100k_base`) track it far better than characters — especially for dense content like tables. |
| **Gemini embeddings with task prefixes + L2-norm** | `gemini-embedding-2` ignores `task_type`, so intent is steered via instruction prefixes; the 768-dim (truncated MRL) vectors are L2-normalized for correct cosine search. |
| **Stateless server, client-held history** | Conversation memory is replayed by the client per request, so the API stores nothing and scales horizontally. |
| **Qdrant for vectors, FastAPI + Streamlit, Docker Compose** | Fast disk-backed vector search with a managed-cloud option; a modern auto-documented API; a quick clean UI; one-command startup. |

---

## Prerequisites
- **Python 3.11+** and **Docker Desktop** (for Qdrant and/or the full stack)
- A **Gemini API key** → https://aistudio.google.com/apikey
- *(Optional)* an NVIDIA GPU — speeds up re-ranking; CPU works fine otherwise.

---

## Quick start — Option A: Docker (recommended, fully self-contained)

```bash
# 1. Configure
cp .env.example .env            # then put your GEMINI_API_KEY in .env

# 2. Start the vector DB
docker compose up -d qdrant

# 3. Ingest the sample documents into Qdrant + BM25 (uses your Gemini key)
docker compose run --rm api python ingest_documents.py --data-dir data/sample_documents --recreate

# 4. Launch the API + UI
docker compose up --build
```
Open **http://localhost:8501** (UI) and **http://localhost:8000/docs** (API).

> The Compose stack runs Qdrant + API + Streamlit together. The API container reaches
> Qdrant by its service name (`qdrant:6333`), set automatically — no manual config.

## Quick start — Option B: Local (venv)

```bash
# 1. Environment
python -m venv venv
source venv/bin/activate         # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env             # add GEMINI_API_KEY

# 3. Start Qdrant (local container) — or set QDRANT_URL to Qdrant Cloud in .env
docker compose up -d qdrant

# 4. Ingest documents
python ingest_documents.py --data-dir data/sample_documents --recreate

# 5. Start the API (terminal 1)
python -m uvicorn src.api.main:app --reload      # http://localhost:8000/docs

# 6. Start the UI (terminal 2)
streamlit run app.py                             # http://localhost:8501
```

> **Windows note:** if the `uvicorn` command fails with a launcher error, use the module form
> shown above (`python -m uvicorn ...`) — it bypasses a stale console-script shim.

---

## ⏱️ Why is it slow? (expected, and how to speed it up)

The system is correct but has a few **one-time** and **inherent** costs worth knowing:

1. **First API startup downloads the re-ranker model.** `BAAI/bge-reranker-base` (~1 GB) is
   pulled from Hugging Face the first time the API boots, *before* it accepts requests — so the
   first start can take **30–90 s**. Watch for `Pipeline loaded.` in the logs. It's **cached
   afterward**, so later starts are fast.
2. **Each factual question makes ~2 Gemini calls.** The agent first decides + runs the search
   tool, then writes the grounded answer. That's the cost of letting the model choose whether to
   retrieve (and rewrite the query). **Greetings make 1 call and skip retrieval entirely.**
3. **Gemini free-tier rate limits.** The free tier allows ~20 generation calls/day per model and
   limited requests/minute. Ingestion **paces** embedding calls (~0.7 s each) to stay under the
   limit, and bulk evaluation is throttled. A paid tier removes this (`GEMINI_LLM_MODEL` is a setting).
4. **Re-ranking runs locally on CPU** by default. It only scores the top ~10 candidates, so it's
   cheap — but a **GPU** makes it noticeably faster (install a CUDA build of `torch`; the code
   auto-detects CUDA).
5. **First Docker build is large.** It installs `torch` + `sentence-transformers` (multiple GB),
   so `docker compose up --build` is slow the first time and cached after.

> TL;DR: the slow parts are the **one-time reranker download** and the **first Docker build**;
> per-question latency is mostly the **two Gemini calls** + network.

---

## API

### `POST /ask`
```json
{ "question": "What is the refund policy?", "top_k": 5,
  "history": [ {"role":"user","content":"..."}, {"role":"assistant","content":"..."} ] }
```
`history` is optional (prior turns for follow-ups). Response:
```json
{
  "answer": "Refunds are allowed within 30 days of purchase.",
  "sources": [ { "document": "Customer_Policy.pdf", "page": 5 } ],
  "confidence": 0.91,
  "context_used": [ /* the passages used, for inspection */ ]
}
```
`confidence` is the top cross-encoder relevance probability (0–1; `bge-reranker` already applies
the sigmoid). If a search ran but scored below `CONFIDENCE_THRESHOLD` (default 0.10), the API
abstains with the standard *"I don't have enough information…"* message and empty `sources`.
Greetings return `confidence: 1.0` and no sources.

### Other endpoints
- `POST /search` — hybrid search + rerank only (no answer); handy for debugging.
- `POST /feedback` — `{ "question", "answer", "rating": "up"|"down", "comment", "sources" }`
  → appends a line to `data/feedback.jsonl`.
- `GET /health` — liveness check.

Interactive docs: **http://localhost:8000/docs**.

---

## Evaluation

```bash
python -m src.eval.evaluate            # deterministic metrics (fits the free tier)
python -m src.eval.evaluate --judge    # + Gemini faithfulness (groundedness) score
```
Runs 15 hand-written test questions (`data/test_queries.json`) through the **real pipeline** and
reports `source_accuracy`, `keyword_recall`, `abstention_accuracy`, `avg_latency_ms`,
`avg_confidence` (and `faithfulness` with `--judge`). Writes `eval_report.md` +
`data/eval_results.json`. See [eval_report.md](eval_report.md) for a sample run.

---

## Project structure

```
anthrasync/
├── src/
│   ├── config.py              # settings (loaded from .env)
│   ├── logger.py              # logging setup
│   ├── pipeline.py            # RAGPipeline — the agentic answer loop
│   ├── ingest/
│   │   ├── loaders.py         # PDF / DOCX / TXT → pages (+ metadata)
│   │   ├── chunker.py         # token-based chunking
│   │   ├── embedder.py        # Gemini embeddings (task prefixes + L2 norm)
│   │   └── indexer.py         # Qdrant (dense) + BM25 (sparse)
│   ├── retrieval/
│   │   ├── hybrid_search.py   # dense + BM25, RRF fusion
│   │   └── reranker.py        # cross-encoder re-ranking
│   ├── generation/
│   │   ├── generator.py       # agent + tool + grounded prompt
│   │   └── guardrails.py      # confidence scoring
│   ├── api/main.py            # FastAPI: /ask /search /feedback /health
│   └── eval/evaluate.py       # evaluation harness
├── app.py                     # Streamlit chat UI
├── ingest_documents.py        # ingestion CLI
├── data/
│   ├── sample_documents/      # 5 sample company docs
│   └── test_queries.json      # evaluation test set
├── Dockerfile · docker-compose.yml · .dockerignore
├── requirements.txt · .env.example
└── README.md · SYSTEM_ARCHITECTURE.md
```

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | **Required.** Your Gemini API key. |
| `GEMINI_LLM_MODEL` | `gemini-2.5-flash` | Answer-generation model. |
| `GEMINI_EMBEDDING_MODEL` / `_DIM` | `gemini-embedding-2` / `768` | Embedding model + size. |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint (Compose overrides to `qdrant:6333`). |
| `QDRANT_API_KEY` | _(empty)_ | Only needed for Qdrant Cloud. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `600` / `0.15` | Chunk size in **tokens** / overlap fraction. |
| `CONFIDENCE_THRESHOLD` | `0.10` | Floor on the top reranker probability; abstains below it. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model. |
| `FEEDBACK_LOG_PATH` | `data/feedback.jsonl` | Where 👍/👎 feedback is stored. |

---

## Limitations
- **Free-tier quota** (~20 answers/day) limits bulk testing/demos; a paid tier removes it.
- **Memory is per-session & client-held** — capped to recent turns; not persisted server-side.
- **Text-file page numbers are approximate** (PDFs are exact).
- **No authentication** — the API is open; add a key/login before public exposure.
- **In-memory BM25 index** — fine for thousands of chunks; needs a dedicated store at much larger scale.

---

## Future improvements
- **Persistent, server-side conversation memory** — beyond the current client-held history, so long sessions don't forget early turns.
- **Close the feedback loop** — use the collected 👍/👎 (`data/feedback.jsonl`) to tune retrieval and prompts.
- **More agent tools** — e.g. separate HR vs. compliance indexes, a calculator, or document-scoped search.
- **Move keyword search into Qdrant** — drop the in-memory BM25 `.pkl` for a single scalable store.
- **Auth & per-document access control** — logins and per-department permissions before public exposure.
- **Production hosting** — managed Qdrant Cloud, horizontal API replicas, and a GPU for faster reranking.

See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for the full design, scalability notes, and roadmap.

---