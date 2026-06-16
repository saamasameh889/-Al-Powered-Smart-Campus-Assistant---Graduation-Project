#!/usr/bin/env python3
"""
phase6_conversational_memory.py  —  Zewail Campus Digital Assistant
═══════════════════════════════════════════════════════════════════════════════
Phase 6: Conversational memory layer + Academic Advisor AI integration.

Exports:
  ConversationSession       — holds per-session history + student profile
  ConversationalAssistant   — wraps CampusRAG + AdvisorEngine

Memory design:
  Short-term  : full conversation history (last N turns) in every GPT call.
  Long-term   : StudentProfile extracted from conversation text; persisted per
                session in data/sessions/<session_id>.json.
  Academic AI : AdvisorEngine (phase8) intercepts planning / graduation /
                prerequisite queries and generates structured advisor responses.
                General queries fall through to the standard RAG pipeline.

Standalone demo:
  python phase6_conversational_memory.py
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT  = Path(__file__).parent
SESSIONS_DIR  = PROJECT_ROOT / "data" / "sessions"

MAX_HISTORY_TURNS = 10


# ══════════════════════════════════════════════════════════════════════════════
#  ConversationSession
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConversationSession:
    """
    Everything the assistant knows about one ongoing conversation.

    user_profile stores a StudentProfile.to_dict() snapshot so it survives
    JSON serialisation and page reloads.
    """
    session_id:           str           = field(default_factory=lambda: str(uuid.uuid4()))
    created_at:           str           = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    conversation_history: list[dict]    = field(default_factory=list)
    user_profile:         dict          = field(default_factory=dict)
    topic_counts:         dict[str,int] = field(default_factory=dict)
    query_count:          int           = 0
    total_response_time:  float         = 0.0

    # ── Serialisation ───────────────────────────────────────────────────────────

    def save(self, directory: Path = SESSIONS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.session_id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, session_id: str, directory: Path = SESSIONS_DIR) -> "ConversationSession":
        path = directory / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    # ── History helpers ─────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self.conversation_history.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self.conversation_history.append({"role": "assistant", "content": content})

    def recent_history(self, n_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
        return self.conversation_history[-(n_turns * 2):]

    # ── Profile helpers (delegates to StudentProfile) ──────────────────────────

    def get_profile(self):
        """Return the current StudentProfile object."""
        from phase8_advisor_engine import StudentProfile
        return StudentProfile.from_dict(self.user_profile)

    def set_profile(self, profile) -> None:
        """Store a StudentProfile back into the session dict."""
        self.user_profile = profile.to_dict()

    def profile_summary(self) -> str:
        """Compact profile string for injection into RAG prompts."""
        p = self.get_profile()
        if not p.has_academic_context():
            return ""
        return "[Student Profile]\n" + p.summary_for_prompt()


# ══════════════════════════════════════════════════════════════════════════════
#  ConversationalAssistant
# ══════════════════════════════════════════════════════════════════════════════

class ConversationalAssistant:
    """
    Memory-aware assistant with integrated Academic Advisor AI.

    Routing logic (invisible to the student — same chat interface):
      1. Extract / update StudentProfile from the question text.
      2. Route intent via AdvisorEngine.IntentRouter:
           planning / graduation / prerequisite / risk
             → AdvisorEngine generates structured advisor response.
           profile share (no question)
             → Quick profile acknowledgement.
           general university question
             → Standard RAG pipeline (existing behaviour).
      3. Append to conversation history and save session.

    Usage:
        session   = ConversationSession()
        assistant = ConversationalAssistant()
        answer, sources = assistant.ask("What should I take next semester?", session)
    """

    def __init__(self) -> None:
        from phase5_rag_pipeline import CampusRAG
        from phase8_advisor_engine import AdvisorEngine

        self._rag     = CampusRAG()
        self._advisor = AdvisorEngine(self._rag)

    def ask(
        self,
        question: str,
        session:  ConversationSession,
        top_k:    int = 6,
    ) -> tuple[str, list]:
        """
        Answer a question using RAG + conversational memory + advisor engine.

        Returns:
            (answer_text, retrieved_chunks)
        """
        t0 = time.time()

        # ── 1. Update student profile from question text ────────────────────────
        from phase8_advisor_engine import update_profile

        current_profile = session.get_profile()
        updated_profile = update_profile(question, current_profile)
        session.set_profile(updated_profile)

        # ── 2. Try the Academic Advisor engine first ────────────────────────────
        history = list(session.recent_history())
        answer, chunks = self._advisor.advise(question, updated_profile, history)

        if answer is None:
            # ── 3a. Standard RAG path (general / campus questions) ──────────────
            retrieval_query = self._build_retrieval_query(question, session)

            # Intent classification → adjusts retrieval depth
            intent      = self._rag.classify_intent(question)
            adjusted_k  = top_k + 2 if intent in ("graduation", "prerequisites") else top_k
            chunks, query_note = self._rag.retrieve(retrieval_query, top_k=adjusted_k)

            # Rerank retrieved chunks
            chunks = self._rag.rerank(question, chunks, top_n=top_k)

            # Build history with profile context prepended
            profile_ctx = session.profile_summary()
            gen_history = list(session.recent_history())
            if profile_ctx:
                gen_history = [
                    {"role": "user",      "content": f"[Context] {profile_ctx}"},
                    {"role": "assistant", "content": "Understood. I will use this context to personalise my answers."},
                ] + gen_history

            # Use agentic tool-calling for calculation-heavy intents
            if intent in ("prerequisites", "graduation"):
                answer, chunks = self._rag.answer_with_tools(
                    question, history=gen_history, top_k=top_k
                )
            else:
                answer = self._rag.generate(
                    question, chunks, history=gen_history, query_note=query_note
                )

        # ── 4. Record in session ───────────────────────────────────────────────
        self._finalize(session, question, answer, chunks, t0)
        return answer, chunks

    def ask_stream(
        self,
        question: str,
        session:  "ConversationSession",
        top_k:    int = 6,
    ):
        """
        Streaming variant of ask().

        Returns: (chunks, content, is_streamed)
          - is_streamed=False → content is a complete answer string
          - is_streamed=True  → content is a generator that yields token strings;
                                session is finalised once the generator is exhausted
        """
        t0 = time.time()

        # ── 1. Update student profile ──────────────────────────────────────────
        from phase8_advisor_engine import update_profile
        current_profile = session.get_profile()
        updated_profile = update_profile(question, current_profile)
        session.set_profile(updated_profile)

        # ── 2. Try the Academic Advisor engine ─────────────────────────────────
        history = list(session.recent_history())
        answer, chunks = self._advisor.advise(question, updated_profile, history)

        if answer is not None:
            self._finalize(session, question, answer, chunks, t0)
            return chunks, answer, False

        # ── 3. RAG path with intent + reranking ───────────────────────────────
        retrieval_query     = self._build_retrieval_query(question, session)
        intent              = self._rag.classify_intent(question)
        adjusted_k          = top_k + 2 if intent in ("graduation", "prerequisites") else top_k
        chunks, query_note  = self._rag.retrieve(retrieval_query, top_k=adjusted_k)
        chunks              = self._rag.rerank(question, chunks, top_n=top_k)

        profile_ctx = session.profile_summary()
        gen_history = list(session.recent_history())
        if profile_ctx:
            gen_history = [
                {"role": "user",      "content": f"[Context] {profile_ctx}"},
                {"role": "assistant", "content": "Understood. I will use this context to personalise my answers."},
            ] + gen_history

        # Agentic path (no streaming — already fast, structured output)
        if intent in ("prerequisites", "graduation"):
            answer, chunks = self._rag.answer_with_tools(
                question, history=gen_history, top_k=top_k
            )
            self._finalize(session, question, answer, chunks, t0)
            return chunks, answer, False

        # Streaming RAG path
        stream_gen = self._rag.generate_stream(
            question, chunks, history=gen_history, query_note=query_note
        )

        def _streaming():
            full_text = ""
            for tok in stream_gen:
                full_text += tok
                yield tok
            self._finalize(session, question, full_text, chunks, t0)

        return chunks, _streaming(), True

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _finalize(
        self,
        session:  ConversationSession,
        question: str,
        answer:   str,
        chunks:   list,
        t0:       float,
    ) -> None:
        """Persist the completed turn into session state and save to disk."""
        session.add_user(question)
        session.add_assistant(answer)
        session.query_count         += 1
        session.total_response_time += time.time() - t0
        for c in chunks[:3]:
            cat = c.category
            session.topic_counts[cat] = session.topic_counts.get(cat, 0) + 1
        try:
            session.save()
        except Exception:
            pass

    def _build_retrieval_query(self, question: str, session: ConversationSession) -> str:
        """
        Enrich the retrieval query for follow-up questions:
        if the user asks a short follow-up with reference words ("it", "that", …)
        prepend the previous question so retrieval has enough context.
        """
        if not session.conversation_history:
            return question

        words = question.lower().split()
        _FOLLOWUP = {"it","that","them","they","this","those","its","their",
                     "more","else","also","same","such","which"}
        is_follow_up = len(words) <= 7 and bool(_FOLLOWUP & set(words))

        if is_follow_up:
            last_user = next(
                (m["content"] for m in reversed(session.conversation_history)
                 if m["role"] == "user"), ""
            )
            if last_user and last_user != question:
                return f"{last_user} {question}"

        return question


# ══════════════════════════════════════════════════════════════════════════════
#  Standalone demo
# ══════════════════════════════════════════════════════════════════════════════

DEMO_TURNS = [
    # Turn 1 — profile share (tests profile engine)
    "Hi, I'm a CSAI student studying Computer Science. "
    "I'm in semester 5 and my GPA is 2.8. "
    "I completed CSAI 101, CSAI 102, CSAI 201, CSAI 251, MATH 201, MATH 202, PHYS 201. "
    "I failed CSAI 261 (Signals and Systems).",

    # Turn 2 — academic planning (tests advisor engine)
    "Given my profile, what courses should I take next semester?",

    # Turn 3 — general campus question (tests RAG passthrough)
    "What scholarships are available for undergraduate students at Zewail City?",

    # Turn 4 — graduation (tests graduation engine)
    "When will I graduate based on my current progress?",
]


def demo() -> None:
    print("Phase 6 — Conversational Memory + Academic Advisor Demo")
    print("=" * 65)

    try:
        assistant = ConversationalAssistant()
    except Exception as exc:
        print(f"ERROR: {exc}")
        print("Make sure phase 4/5 are complete and OPENAI_API_KEY is set.")
        return

    session = ConversationSession()
    print(f"Session: {session.session_id}\n")

    for i, question in enumerate(DEMO_TURNS, 1):
        print(f"━━━ Turn {i} ━━━")
        print(f"User: {question[:120]}{'…' if len(question)>120 else ''}\n")
        t0 = time.time()
        answer, chunks = assistant.ask(question, session)
        elapsed = time.time() - t0
        print(f"Assistant ({elapsed:.1f}s):")
        print(f"  {answer[:600]}{'…' if len(answer)>600 else ''}\n")
        if chunks:
            print(f"  Sources ({len(chunks)}):")
            for c in chunks[:2]:
                print(f"    [{c.score:.3f}] [{c.category}] {c.source}")
        profile = session.get_profile()
        if profile.has_academic_context():
            print(f"  Profile: school={profile.school} major={profile.major} "
                  f"sem={profile.semester} gpa={profile.gpa} "
                  f"done={len(profile.completed_courses)} failed={len(profile.failed_courses)}")
        print()

    print(f"Session saved: {SESSIONS_DIR / session.session_id}.json")
    print(f"Total queries: {session.query_count} | "
          f"Avg time: {session.total_response_time/session.query_count:.1f}s")


if __name__ == "__main__":
    demo()
