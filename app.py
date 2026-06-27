"""
AnthraSync — Enterprise Knowledge Assistant (Streamlit UI).

A focused chat interface: ask a question, get a grounded answer with a
confidence indicator and source documents. Talks to the FastAPI backend
(POST /ask, POST /feedback), keeps conversation memory on the client, and
replays recent turns so follow-ups work.

Message handling uses a two-phase pattern (append user → rerun → fetch answer)
so the UI stays responsive and consistent across many messages.
"""

import os
import time

import requests
import streamlit as st

# Resolve the backend URL. API_URL wins if explicitly set; otherwise build it from
# API_HOST/API_PORT. Never try to *connect* to 0.0.0.0 — that's a server bind
# address, not a client target (it fails on Windows/macOS) — fall back to localhost.
# In docker-compose the UI service sets API_HOST=api, which still works here.
_api_host = os.getenv("API_HOST") or "localhost"
if _api_host in ("0.0.0.0", ""):
    _api_host = "localhost"
_api_port = os.getenv("API_PORT", "8000")
API_URL = os.getenv("API_URL") or f"http://{_api_host}:{_api_port}"
REQUEST_TIMEOUT = 120
HISTORY_LIMIT = 8  # messages (~4 turns) replayed to the backend as memory

EXAMPLE_QUESTIONS = [
    "What is the employee leave policy?",
    "What is the refund policy?",
    "What is the password policy?",
    "How do I contact support?",
]

st.set_page_config(
    page_title="AnthraSync - Knowledge Assistant",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Styling ---
st.markdown(
    """
    <style>
        .block-container { padding-top: 2.2rem; max-width: 820px; }
        .app-title { font-size: 2.2rem; font-weight: 800; margin-bottom: 0; color: #1f2933; }
        .app-subtitle { color: #6b7280; font-size: 1rem; margin-top: 0.2rem; }
        .source-pill {
            display: inline-block; background: #eef2ff; color: #4338ca;
            border-radius: 999px; padding: 2px 12px; margin: 3px 4px 0 0;
            font-size: 0.85rem; font-weight: 600;
        }
        .conf-chip {
            display: inline-block; border-radius: 999px; padding: 1px 10px;
            font-size: 0.78rem; font-weight: 700; margin-bottom: 4px;
        }
        .conf-high { background: #dcfce7; color: #166534; }
        .conf-med  { background: #fef9c3; color: #854d0e; }
        .conf-low  { background: #fee2e2; color: #991b1b; }
        /* Animated "thinking" three-dot indicator */
        .typing { display: inline-flex; gap: 5px; padding: 8px 2px; align-items: center; }
        .typing span {
            width: 8px; height: 8px; border-radius: 50%; background: #9ca3af;
            display: inline-block; animation: typing-bounce 1.2s infinite ease-in-out;
        }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typing-bounce {
            0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
            40% { transform: translateY(-6px); opacity: 1; }
        }
        /* Compact feedback buttons */
        div[data-testid="column"] .stButton button {
            padding: 0 8px; border: none; background: transparent; font-size: 1rem;
        }
        div[data-testid="column"] .stButton button:hover { background: #f3f4f6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────── backend calls ────────────────────────────

def ask_backend(question: str, history: list, top_k: int = 5) -> dict:
    """Call the /ask endpoint. Returns the JSON dict or raises for the caller."""
    resp = requests.post(
        f"{API_URL}/ask",
        json={"question": question, "top_k": top_k, "history": history},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def send_feedback(question: str, data: dict, rating: str) -> bool:
    """Send a thumbs up/down for an answer to the backend. Best-effort."""
    try:
        requests.post(
            f"{API_URL}/feedback",
            json={
                "question": question,
                "answer": data.get("answer", ""),
                "rating": rating,
                "sources": data.get("sources", []),
            },
            timeout=10,
        )
        return True
    except Exception:
        return False


# ──────────────────────────── helpers ────────────────────────────

def _build_history(msgs: list) -> list:
    """Flatten session messages into [{role, content}], capped to recent turns."""
    history = []
    for m in msgs:
        if m["role"] == "user":
            history.append({"role": "user", "content": m["content"]})
        else:
            history.append({"role": "assistant", "content": m["data"].get("answer", "")})
    return history[-HISTORY_LIMIT:]


def _typewriter(text: str) -> None:
    """Reveal `text` with a brief 'typing' effect, then render final markdown."""
    placeholder = st.empty()
    n = len(text)
    step = max(1, n // 150)  # cap frames so the animation stays ~1.5s
    for i in range(step, n + 1, step):
        placeholder.markdown(text[:i] + " ▌")
        time.sleep(0.012)
    placeholder.markdown(text)


def _confidence_chip(confidence: float) -> str:
    """Color-coded confidence badge (only shown for grounded answers)."""
    pct = int(round(confidence * 100))
    if confidence >= 0.7:
        cls, label = "conf-high", "High"
    elif confidence >= 0.4:
        cls, label = "conf-med", "Medium"
    else:
        cls, label = "conf-low", "Low"
    return f"<span class='conf-chip {cls}'>Confidence: {label} · {pct}%</span>"


def render_answer(
    data: dict, question: str = "", key_prefix: str = "", typewriter: bool = False
) -> None:
    """Render an assistant answer block: text, confidence, sources, feedback."""
    answer = data.get("answer", "")
    if typewriter and answer:
        _typewriter(answer)
    else:
        st.markdown(answer)

    sources = data.get("sources", [])
    # Confidence chip only when the answer is grounded in sources.
    if sources and isinstance(data.get("confidence"), (int, float)):
        st.markdown(_confidence_chip(float(data["confidence"])), unsafe_allow_html=True)

    if sources:
        unique_sources = list(
            dict.fromkeys(
                (s.get("document", "Unknown"), s.get("page", "?")) for s in sources
            )
        )
        pills = "".join(
            f"<span class='source-pill'>📄 {doc} · p.{page}</span>"
            for doc, page in unique_sources
        )
        st.markdown("**Sources**", help="Documents the answer is grounded in.")
        st.markdown(pills, unsafe_allow_html=True)

    # Thumbs up/down feedback (recorded to the backend's feedback log).
    if key_prefix:
        given = st.session_state.feedback_given.get(key_prefix)
        if given:
            st.caption("👍 Thanks for your feedback!" if given == "up"
                       else "👎 Thanks — we'll use this to improve.")
        else:
            up, down, _ = st.columns([1, 1, 10])
            if up.button("👍", key=f"{key_prefix}_up", help="Helpful"):
                send_feedback(question, data, "up")
                st.session_state.feedback_given[key_prefix] = "up"
                st.rerun()
            if down.button("👎", key=f"{key_prefix}_down", help="Not helpful"):
                send_feedback(question, data, "down")
                st.session_state.feedback_given[key_prefix] = "down"
                st.rerun()


# ──────────────────────────── state ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None          # example question queued to send
if "awaiting" not in st.session_state:
    st.session_state.awaiting = None         # user question awaiting an answer
if "animate_idx" not in st.session_state:
    st.session_state.animate_idx = None      # which answer to typewriter-reveal
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}     # key_prefix -> "up" | "down"


# ──────────────────────────── sidebar ────────────────────────────
with st.sidebar:
    st.markdown("### 💡 AnthraSync")
    st.caption("Enterprise Knowledge Assistant — answers grounded in your company's documents.")
    if st.button("🗑️  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.awaiting = None
        st.session_state.animate_idx = None
        st.session_state.feedback_given = {}
        st.rerun()
    st.divider()
    st.caption(f"Backend: `{API_URL}`")


# ──────────────────────────── header ────────────────────────────
st.markdown('<p class="app-title">💡 AnthraSync</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="app-subtitle">Ask anything about company policies, guides, and FAQs — '
    "get an answer with its sources.</p>",
    unsafe_allow_html=True,
)
st.divider()

# Input box (pinned to the bottom; drawn early so it stays visible while thinking).
typed = st.chat_input("Ask a question…")

# Example questions (only before the first message).
if not st.session_state.messages and not st.session_state.awaiting:
    st.caption("Try one of these:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(example, use_container_width=True, key=f"ex_{i}"):
            st.session_state.pending = example
            st.rerun()


# ──────────────────────────── render history ────────────────────────────
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(
                msg["data"],
                question=msg.get("question", ""),
                key_prefix=f"msg{idx}",
                typewriter=(idx == st.session_state.animate_idx),
            )
st.session_state.animate_idx = None  # animate each answer only once


# ──────────────── phase 2: answer a question that's awaiting ────────────────
if st.session_state.awaiting:
    q = st.session_state.awaiting
    history = _build_history(st.session_state.messages[:-1])  # exclude current Q
    with st.chat_message("assistant"):
        thinking = st.empty()
        thinking.markdown(
            "<div class='typing'><span></span><span></span><span></span></div>",
            unsafe_allow_html=True,
        )
        try:
            data = ask_backend(q, history)
            st.session_state.messages.append(
                {"role": "assistant", "data": data, "question": q}
            )
            st.session_state.animate_idx = len(st.session_state.messages) - 1
        except requests.exceptions.ConnectionError:
            st.session_state.messages.append({
                "role": "assistant",
                "question": q,
                "data": {"answer": "⚠️ Couldn't reach the backend. Start it with "
                                   "`uvicorn src.api.main:app --reload`."},
            })
        except requests.exceptions.HTTPError as e:
            st.session_state.messages.append({
                "role": "assistant",
                "question": q,
                "data": {"answer": f"⚠️ The assistant couldn't answer right now "
                                   f"({e.response.status_code})."},
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "question": q,
                "data": {"answer": f"⚠️ Something went wrong: {e}"},
            })
        finally:
            thinking.empty()
    st.session_state.awaiting = None
    st.rerun()


# ──────────────── phase 1: accept new input ────────────────
incoming = typed or st.session_state.pending
st.session_state.pending = None
if incoming and not st.session_state.awaiting:
    st.session_state.messages.append({"role": "user", "content": incoming})
    st.session_state.awaiting = incoming
    st.rerun()
