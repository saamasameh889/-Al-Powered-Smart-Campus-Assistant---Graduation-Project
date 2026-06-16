"""
rag_evaluator.py — Zewail City Campus Assistant
════════════════════════════════════════════════
LLM-judge RAG evaluation (no external eval framework needed).

Evaluates three dimensions via a single gpt-4o-mini call per question:
  • Faithfulness      — is every claim grounded in the retrieved context?
  • Answer Relevancy  — does the answer directly address the question?
  • Context Precision — are the retrieved chunks relevant to the question?

Public API (used by phase7_streamlit_app.py):
    from rag_evaluator import render_evaluation_tab
"""
from __future__ import annotations

import re
import time
from typing import Optional

import streamlit as st

# ── Default evaluation questions ───────────────────────────────────────────────
EVAL_QUESTIONS: list[tuple[str, str]] = [
    ("How many credit hours does a CSAI student need to graduate?",        "graduation"),
    ("What is the academic probation policy at Zewail City?",              "policy"),
    ("What scholarships are available for undergraduate students?",         "scholarships"),
    ("Who are the faculty members in the School of CSAI?",                 "faculty"),
    ("What courses can I take after completing CSAI 101 and CSAI 201?",    "prerequisites"),
    ("What are the admission requirements for Zewail City?",               "admissions"),
    ("What research institutes are available at Zewail City?",             "research"),
]


# ── LLM judge ──────────────────────────────────────────────────────────────────

def _llm_judge(question: str, answer: str, chunks: list, rag) -> dict[str, int]:
    """
    Score a single RAG response on three 1-5 dimensions using gpt-4o-mini.
    Returns {"faithfulness": int, "relevancy": int, "context_precision": int}.
    """
    context = "\n".join(c.text[:220] for c in chunks[:4])
    prompt = (
        "Evaluate this academic advisor RAG system response.\n\n"
        f"Question: {question}\n"
        f"Retrieved context (first 4 chunks):\n{context[:900]}\n"
        f"Answer: {answer[:700]}\n\n"
        "Score each dimension 1-5 (integers only):\n"
        "• Faithfulness: Is every claim in the answer supported by the context? "
        "(1=hallucinated facts, 5=fully grounded)\n"
        "• Relevancy: Does the answer directly address the question? "
        "(1=off-topic, 5=spot-on)\n"
        "• Context Precision: Are the retrieved chunks relevant to the question? "
        "(1=irrelevant, 5=highly relevant)\n\n"
        "Reply in this exact format with nothing else:\n"
        "faithfulness=X relevancy=X context_precision=X"
    )
    try:
        resp = rag._oai_chat.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()

        def _extract(key: str) -> int:
            m = re.search(rf'{key}=([1-5])', text)
            return int(m.group(1)) if m else 3

        return {
            "faithfulness":      _extract("faithfulness"),
            "relevancy":         _extract("relevancy"),
            "context_precision": _extract("context_precision"),
        }
    except Exception:
        return {"faithfulness": 0, "relevancy": 0, "context_precision": 0}


def run_evaluation(
    rag,
    questions: Optional[list[tuple[str, str]]] = None,
    progress_cb=None,
) -> list[dict]:
    """
    Run the full evaluation suite.

    Args:
        rag:         CampusRAG instance
        questions:   list of (question_text, category) tuples
        progress_cb: optional callable(i, total) for progress reporting

    Returns list of result dicts.
    """
    qs = questions or EVAL_QUESTIONS
    results = []
    for i, (q, cat) in enumerate(qs):
        t0 = time.time()
        answer, chunks = rag.answer(q)
        scores = _llm_judge(q, answer, chunks, rag)
        avg_retrieval = round(
            sum(c.score for c in chunks) / max(len(chunks), 1), 3
        )
        results.append({
            "question":          q,
            "category":          cat,
            "answer":            answer,
            "n_sources":         len(chunks),
            "avg_retrieval_score": avg_retrieval,
            "latency_s":         round(time.time() - t0, 2),
            **scores,
        })
        if progress_cb:
            progress_cb(i + 1, len(qs))
    return results


# ── Streamlit component ────────────────────────────────────────────────────────

def render_evaluation_tab(assistant) -> None:
    """Render the full Evaluation tab UI inside phase7_streamlit_app.py."""
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:800;color:#C4B5FD;"
        "margin-bottom:4px'>🔬 RAG Evaluation Dashboard</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:.83rem;color:#4A3A7A;margin-bottom:18px'>"
        "Automatically scores retrieval quality and answer faithfulness "
        "using GPT-4o-mini as an LLM judge across "
        f"{len(EVAL_QUESTIONS)} benchmark questions.</div>",
        unsafe_allow_html=True,
    )

    col_run, col_custom = st.columns([1, 2])
    with col_run:
        run_btn = st.button(
            "▶  Run Evaluation",
            type="primary",
            use_container_width=True,
            key="eval_run",
        )
    with col_custom:
        custom_q = st.text_input(
            "Add a custom question (optional):",
            placeholder="e.g. What electives are available for CSAI?",
            key="eval_custom_q",
            label_visibility="collapsed",
        )

    if run_btn:
        qs = list(EVAL_QUESTIONS)
        if custom_q.strip():
            qs.append((custom_q.strip(), "custom"))

        progress_bar = st.progress(0)
        status_txt   = st.empty()
        results: list[dict] = []

        def _cb(done: int, total: int) -> None:
            progress_bar.progress(done / total)
            status_txt.caption(f"Evaluated {done}/{total} questions…")

        with st.spinner("Running evaluation — this takes ~30 seconds…"):
            try:
                results = run_evaluation(assistant._rag, questions=qs, progress_cb=_cb)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                return

        progress_bar.empty()
        status_txt.empty()
        st.session_state["eval_results"] = results

    # ── Results display ────────────────────────────────────────────────────────
    results = st.session_state.get("eval_results")
    if not results:
        st.markdown(
            "<div style='text-align:center;padding:52px 0;color:#3D3060'>"
            "<div style='font-size:2.4rem;margin-bottom:10px;opacity:.5'>🔬</div>"
            "<div style='font-size:.9rem;color:#5B4D8A'>"
            "Click <strong>Run Evaluation</strong> to benchmark the RAG pipeline.</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Summary metrics
    def _avg(key: str) -> float:
        vals = [r[key] for r in results if r[key] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    faith_avg = _avg("faithfulness")
    relev_avg = _avg("relevancy")
    ctx_avg   = _avg("context_precision")
    overall   = (faith_avg + relev_avg + ctx_avg) / 3

    def _score_color(v: float) -> str:
        if v >= 4.0: return "#10B981"
        if v >= 3.0: return "#F59E0B"
        return "#EF4444"

    metric_html = "".join(
        f"<div style='text-align:center;background:rgba(255,255,255,.03);"
        f"border:1px solid rgba(139,92,246,.18);border-radius:16px;padding:18px 12px'>"
        f"<div style='font-size:1.9rem;font-weight:800;color:{_score_color(v)}'>{v:.1f}</div>"
        f"<div style='font-size:.64rem;color:#5B4D8A;text-transform:uppercase;"
        f"letter-spacing:.1em;margin-top:4px'>{lbl}</div></div>"
        for v, lbl in [
            (faith_avg, "Faithfulness"),
            (relev_avg, "Answer Relevancy"),
            (ctx_avg,   "Context Precision"),
            (overall,   "Overall / 5"),
        ]
    )
    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:10px;"
        f"margin-bottom:20px'>{metric_html}</div>",
        unsafe_allow_html=True,
    )

    # Per-question table
    import pandas as pd
    df = pd.DataFrame([{
        "Question":          r["question"][:65] + ("…" if len(r["question"]) > 65 else ""),
        "Category":          r["category"],
        "Faithfulness":      r["faithfulness"],
        "Relevancy":         r["relevancy"],
        "Context Precision": r["context_precision"],
        "Sources":           r["n_sources"],
        "Avg Score":         r["avg_retrieval_score"],
        "Latency (s)":       r["latency_s"],
    } for r in results])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Expandable detail per question
    st.markdown(
        "<div style='font-size:.72rem;font-weight:700;color:#9D77F5;"
        "text-transform:uppercase;letter-spacing:.1em;margin:16px 0 8px'>"
        "Question Details</div>",
        unsafe_allow_html=True,
    )
    for r in results:
        label = (
            f"[{r['category']}]  "
            f"{r['question'][:70]}{'…' if len(r['question'])>70 else ''}  "
            f"— F:{r['faithfulness']} R:{r['relevancy']} C:{r['context_precision']}"
        )
        with st.expander(label, expanded=False):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Faithfulness",      f"{r['faithfulness']}/5")
            mc2.metric("Answer Relevancy",  f"{r['relevancy']}/5")
            mc3.metric("Context Precision", f"{r['context_precision']}/5")
            st.markdown(f"**Answer preview:** {r['answer'][:500]}…")
            st.caption(
                f"{r['n_sources']} sources retrieved  ·  "
                f"avg retrieval score {r['avg_retrieval_score']}  ·  "
                f"{r['latency_s']}s"
            )
