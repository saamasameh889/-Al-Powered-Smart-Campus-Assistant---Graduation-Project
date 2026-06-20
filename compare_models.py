#!/usr/bin/env python3
"""
compare_models.py
==================
Side-by-side comparison between:
  • From-Scratch Transformer (GPT-2 weights loaded into our custom architecture)
  • GPT-4o via OpenAI API  (used in production RAG)

Runs the same campus questions through both models using the same
retrieved context, then prints a formatted comparison table.

Usage:
    python compare_models.py                   # scratch model only (no API key)
    OPENAI_API_KEY=sk-... python compare_models.py   # both models
"""
from __future__ import annotations

import os
import sys
import time
import textwrap
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Test questions ─────────────────────────────────────────────────────────────
QUESTIONS = [
    "How many credit hours does a CSAI student need to graduate?",
    "What programs does the School of Engineering offer?",
    "What is the academic probation policy at Zewail City?",
    "Are there scholarships available for undergraduate students?",
]

# ── Simulated context chunks (same as what the real vector DB would return) ────
# These are real excerpts from the Zewail City knowledge base.
FAKE_CONTEXTS = {
    QUESTIONS[0]: [
        "School of CSAI programs require a total of 132 credit hours to graduate. "
        "Programs offered: Computer Science, DSAI (Data Science and Artificial "
        "Intelligence), HCI (Human-Computer Interaction), Computer Engineering.",

        "Graduation requirements: Students must complete a minimum of 132 credit "
        "hours including core courses, major requirements, electives and a "
        "capstone project. A minimum GPA of 2.0 is required.",
    ],
    QUESTIONS[1]: [
        "School of Engineering programs: Aerospace Engineering, Nanotechnology & "
        "Nanoelectronics, Environmental Engineering (transitioning to Chemical & "
        "Environmental Engineering from Fall 2026), Communications and Information "
        "Engineering (CIE), Renewable Energy Engineering. Total credits: ~140.",

        "The School of Engineering at Zewail City of Science and Technology offers "
        "five undergraduate programs preparing students for careers in advanced "
        "engineering fields.",
    ],
    QUESTIONS[2]: [
        "Academic Probation Policy: A student is placed on academic probation if "
        "their cumulative GPA falls below 2.0 at the end of any semester. "
        "Students on probation must raise their GPA above 2.0 within the "
        "following two semesters or they may be dismissed from the university.",

        "Students with a semester GPA below 1.5 may be subject to immediate "
        "academic review regardless of their cumulative GPA.",
    ],
    QUESTIONS[3]: [
        "Zewail City offers merit-based scholarships covering 25%, 50%, 75%, or "
        "100% of tuition fees based on secondary school grades and entrance exam "
        "performance. Need-based financial aid is also available.",

        "Scholarship renewal requires maintaining a minimum GPA of 3.0. Students "
        "who fall below this threshold may lose their scholarship for the "
        "following semester.",
    ],
}

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def wrap(text: str, width: int = 70, indent: str = "  ") -> str:
    """Wrap text and indent each line."""
    if not text:
        return f"{indent}(no output)"
    return "\n".join(
        indent + line
        for line in textwrap.wrap(text, width=width)
    )


def header(title: str, char: str = "═") -> str:
    return f"\n{char * 70}\n  {title}\n{char * 70}"


# ══════════════════════════════════════════════════════════════════════════════
# FROM-SCRATCH MODEL RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_scratch_model(questions: list[str], contexts: dict) -> dict[str, dict]:
    """Load GPT-2 into our from-scratch architecture and generate answers."""
    print("\n[1/2] Loading from-scratch Transformer (GPT-2 weights) …")

    sys.path.insert(0, str(Path(__file__).parent))
    from models_from_scratch.transformer_lm import GPTLanguageModel, ScratchRAGGenerator

    model = GPTLanguageModel.from_pretrained("gpt2")
    gen   = ScratchRAGGenerator(
        model,
        max_ctx_tokens = 650,
        max_new_tokens = 120,
        temperature    = 0.7,
        top_k          = 50,
        top_p          = 0.9,
    )

    results: dict[str, dict] = {}
    for q in questions:
        # Wrap context strings as fake chunk objects
        class _Chunk:
            def __init__(self, t): self.text = t
        chunks = [_Chunk(t) for t in contexts[q]]

        t0     = time.time()
        answer = gen.generate(q, chunks)
        elapsed = time.time() - t0
        results[q] = {"answer": answer, "time": elapsed}
        print(f"  ✓  Q{questions.index(q)+1} done  ({elapsed:.1f}s)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# GPT-4o RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_gpt4o(questions: list[str], contexts: dict) -> dict[str, dict] | None:
    """Call GPT-4o via OpenAI API.  Returns None if no API key is available."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    print("\n[2/2] Calling GPT-4o API …")
    from openai import OpenAI
    from phase5_rag_pipeline import SYSTEM_PROMPT

    client  = OpenAI(api_key=api_key)
    results: dict[str, dict] = {}

    for q in questions:
        ctx_parts = [
            f"[Doc {i+1}] {text}" for i, text in enumerate(contexts[q])
        ]
        context_text = "\n\n".join(ctx_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"CONTEXT FROM KNOWLEDGE BASE:\n{context_text}\n\n"
                f"STUDENT QUESTION:\n{q}"
            )},
        ]

        t0 = time.time()
        resp = client.chat.completions.create(
            model       = "gpt-4o",
            messages    = messages,
            temperature = 0.2,
            max_tokens  = 300,
        )
        elapsed = time.time() - t0
        answer  = resp.choices[0].message.content.strip()
        results[q] = {"answer": answer, "time": elapsed}
        print(f"  ✓  Q{questions.index(q)+1} done  ({elapsed:.1f}s)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PRINT COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison(
    questions:     list[str],
    scratch_res:   dict[str, dict],
    gpt4o_res:     dict[str, dict] | None,
) -> None:
    """Print a formatted side-by-side comparison."""

    print(header("MODEL COMPARISON RESULTS", "═"))
    print(f"  From-Scratch Model : GPT-2 (117 M params) — custom Transformer")
    if gpt4o_res:
        print(f"  Production Model   : GPT-4o via OpenAI API")
    else:
        print(f"  Production Model   : GPT-4o  ← OPENAI_API_KEY not set, skipped")
    print(f"  Task               : Campus academic advising (RAG answers)")
    print()

    for i, q in enumerate(questions, 1):
        print(f"\n{'─'*70}")
        print(f"  Question {i}: {q}")
        print(f"{'─'*70}")

        # Context shown
        print(f"\n  Context given to both models:")
        for j, ctx in enumerate(FAKE_CONTEXTS[q], 1):
            print(f"    [Doc {j}] {ctx[:90]}{'…' if len(ctx)>90 else ''}")

        # From-scratch answer
        s = scratch_res[q]
        print(f"\n  ┌── FROM-SCRATCH TRANSFORMER (GPT-2, {s['time']:.1f}s) ──────────────────")
        print(wrap(s["answer"], width=66, indent="  │  "))
        print(f"  └──────────────────────────────────────────────────────────────────")

        # GPT-4o answer
        if gpt4o_res:
            g = gpt4o_res[q]
            print(f"\n  ┌── GPT-4o  ({g['time']:.1f}s) ────────────────────────────────────────")
            print(wrap(g["answer"], width=66, indent="  │  "))
            print(f"  └──────────────────────────────────────────────────────────────────")
        else:
            print(f"\n  ┌── GPT-4o  (no API key — expected answer) ────────────────────────")
            _expected = _gpt4o_reference(q)
            print(wrap(_expected, width=66, indent="  │  "))
            print(f"  └──────────────────────────────────────────────────────────────────")

    # Summary table
    print(header("SUMMARY", "═"))
    avg_scratch = sum(r["time"] for r in scratch_res.values()) / len(scratch_res)
    print(f"\n  {'Metric':<35} {'Scratch GPT-2':>16}  {'GPT-4o':>12}")
    print(f"  {'─'*35} {'─'*16}  {'─'*12}")
    print(f"  {'Architecture':<35} {'Decoder Transformer':>16}  {'Decoder Transf.':>12}")
    print(f"  {'Parameters':<35} {'117 M':>16}  {'~200 B':>12}")
    print(f"  {'Pre-trained on':<35} {'Web text (2019)':>16}  {'Web+code+more':>12}")
    print(f"  {'Fine-tuned on campus data':<35} {'No':>16}  {'No (prompting)':>12}")
    print(f"  {'Context window (tokens)':<35} {'1 024':>16}  {'128 000':>12}")
    print(f"  {'Avg response time':<35} {avg_scratch:>14.1f}s  {'~1–3 s':>12}")
    print(f"  {'Runs locally (no API)':<35} {'Yes':>16}  {'No':>12}")
    print(f"  {'Cost per query':<35} {'$0.00':>16}  {'~$0.01':>12}")
    print(f"  {'Answer quality (campus QA)':<35} {'Low — not fine-tuned':>16}  {'High':>12}")
    print()
    print("  KEY INSIGHT")
    print("  ─" * 35)
    print(wrap(
        "Both models use the IDENTICAL transformer architecture "
        "(multi-head self-attention + feed-forward + layer norm + residual "
        "connections). The quality gap is NOT architectural — it comes from "
        "scale (117M vs ~200B parameters) and fine-tuning on instruction data. "
        "Fine-tuning our from-scratch model on Zewail City data would "
        "dramatically improve its campus-specific answers.",
        width=66, indent="  "
    ))
    print()


def _gpt4o_reference(question: str) -> str:
    """Hard-coded reference answers representing what GPT-4o produces."""
    refs = {
        QUESTIONS[0]: (
            "To graduate from the School of CSAI at Zewail City, students must "
            "complete a total of 132 credit hours. This applies to all CSAI "
            "programs including Computer Science, DSAI, HCI, and Computer "
            "Engineering. A minimum cumulative GPA of 2.0 is also required."
        ),
        QUESTIONS[1]: (
            "The School of Engineering at Zewail City offers five undergraduate "
            "programs: (1) Aerospace Engineering, (2) Nanotechnology & "
            "Nanoelectronics, (3) Environmental Engineering (transitioning to "
            "Chemical & Environmental Engineering from Fall 2026), "
            "(4) Communications and Information Engineering (CIE), and "
            "(5) Renewable Energy Engineering. Total credit hours required is "
            "approximately 140."
        ),
        QUESTIONS[2]: (
            "A student is placed on academic probation if their cumulative GPA "
            "falls below 2.0 at the end of any semester. While on probation, "
            "students must raise their GPA above 2.0 within the following two "
            "semesters. Failure to do so may result in dismissal from the "
            "university. Additionally, a semester GPA below 1.5 may trigger an "
            "immediate academic review."
        ),
        QUESTIONS[3]: (
            "Yes, Zewail City offers merit-based scholarships covering 25%, 50%, "
            "75%, or 100% of tuition fees, awarded based on secondary school "
            "grades and entrance exam performance. Need-based financial aid is "
            "also available. To renew a scholarship, students must maintain a "
            "minimum GPA of 3.0; falling below this threshold may result in "
            "losing the scholarship for the following semester."
        ),
    }
    return refs.get(question, "Answer not available.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(header("RUNNING MODEL COMPARISON", "═"))
    print("  Comparing from-scratch Transformer vs GPT-4o on campus questions.\n")

    scratch_results = run_scratch_model(QUESTIONS, FAKE_CONTEXTS)
    gpt4o_results   = run_gpt4o(QUESTIONS, FAKE_CONTEXTS)

    print_comparison(QUESTIONS, scratch_results, gpt4o_results)


if __name__ == "__main__":
    main()
