#!/usr/bin/env python3
"""
phase5_rag_pipeline.py  —  Zewail Campus Digital Assistant
═══════════════════════════════════════════════════════════════════════════════
Phase 5: Core RAG pipeline — semantic retrieval + GPT-4o generation.

Exports:
  CampusRAG         — main class (retrieve + generate + answer)
  RetrievedChunk    — typed result from retrieve()

Standalone demo (5 sample queries):
  python phase5_rag_pipeline.py
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Course code pattern: 2-6 uppercase letters + optional space + 3-4 digits
_COURSE_CODE_RE = re.compile(r'\b([A-Z]{2,6})\s*(\d{3,4}[A-Z]?)\b')
# Arabic name prefixes that distort embedding distance
_ARABIC_PREFIX_RE = re.compile(
    r'\b(El|Al|Abd\s*El|Abd\s*Al|Abd)\s+(?=[A-Za-z])', re.I
)

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT    = Path(__file__).parent
CHROMA_DIR      = PROJECT_ROOT / "db" / "chroma_db"
COLLECTION_NAME = "zewail_campus"
EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o"

_GROQ_BASE  = "https://api.groq.com/openai/v1"
_GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Grade-point map (used by calculate_gpa tool) ──────────────────────────────
_GRADE_POINTS: dict[str, float] = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "F":  0.0,
}
_GRAD_CREDITS: dict[str, int] = {"CSAI": 132, "BUS": 114, "SCI": 132, "ENGR": 140}

# ── Agentic tool definitions (OpenAI function-calling format) ──────────────────
CAMPUS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_curriculum",
            "description": "Search the Zewail City course catalog by keyword or topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "Course name, topic, or course code"},
                    "program": {"type": "string", "description": "Optional program filter: CSAI, BUS, SCI, ENGR"},
                },
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_gpa",
            "description": "Calculate cumulative GPA from a list of letter grades and credit hours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "grades": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "grade":   {"type": "string", "description": "Letter grade: A, B+, C-, etc."},
                                "credits": {"type": "number", "description": "Credit hours for this course"},
                            },
                            "required": ["grade", "credits"],
                        },
                    },
                },
                "required": ["grades"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_graduation_requirements",
            "description": "Return total credit hours required to graduate from a Zewail City school.",
            "parameters": {
                "type": "object",
                "properties": {
                    "school": {"type": "string", "description": "School code: CSAI, BUS, SCI, or ENGR"},
                },
                "required": ["school"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_prerequisites",
            "description": "Retrieve prerequisite information for a specific course code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_code": {"type": "string", "description": "Course code such as 'CSAI 253' or 'MATH 201'"},
                },
                "required": ["course_code"],
            },
        },
    },
]

_KEYWORD_STOPWORDS = frozenset({
    "what", "which", "when", "where", "who", "how", "does", "do", "is", "are",
    "the", "a", "an", "of", "in", "at", "to", "for", "and", "or", "but", "not",
    "can", "could", "would", "should", "will", "have", "has", "had", "been",
    "i", "my", "me", "we", "you", "your", "this", "that", "these", "those",
    "about", "with", "from", "into", "there", "their", "being", "want", "need",
    "tell", "give", "show", "list", "please", "help", "know", "get", "take",
})


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id:    str
    text:        str
    source:      str
    source_type: str
    category:    str
    page:        str
    score:       float          # cosine similarity (0–1, higher = more relevant)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the official Academic Advisor for Zewail City of Science and Technology (UST).
Help students with: academic policies, degree requirements, course prerequisites,
graduation requirements, admissions, scholarships, faculty information, campus life,
course materials, study resources, and student services.

IMPORTANT FACTS:
- Zewail City (UST) has FOUR (4) undergraduate schools: Engineering (ENGR), CSAI, SCI, and BUS.
- School of Engineering programs: Aerospace Engineering, Nanotechnology & Nanoelectronics,
  Environmental Engineering (-> Chemical & Environmental from Fall 2026), Communications and
  Information Engineering (CIE), Renewable Energy Engineering.
- School of CSAI programs: Computer Science, DSAI, HCI, Computer Engineering (132 cr total).
- School of Science programs: Biomedical Sciences, Nanoscience, Physics of the Universe.
- School of Business programs: Finance, Business Analytics, Actuarial Analysis & Risk Mgmt,
  Operations Management, Entrepreneurship & Innovation Management (114 cr total).

KNOWLEDGE BASE -- STUDENT LINKTREE:
You have access to the Zewail City Student LinkTree, a student-maintained resource hub
containing: course materials, recorded lectures, slides, textbooks, past exams, office
hours, staff information, WhatsApp groups, clubs, coding platforms, VPN/LTS tools, and
student drive collections. Covers all major courses: CSAI, DSAI, Math, Engineering,
General Education, and more.

RULES:
1. REASON THEN ANSWER: Before writing your answer, silently identify: (a) which school/program
   the question is about, (b) what specific facts are needed, (c) which context sources contain
   those facts. Then write a clear, accurate answer based on those facts.
2. Answer using the provided context documents. Extract and synthesise information
   across multiple sources -- do NOT ignore relevant facts in table or list format.
3. Never invent course codes, credit hours, dates, names, or policy details not in the context.
4. If the context does not contain the answer but the topic relates to student resources,
   course materials, office hours, staff info, or campus services -- DO NOT say "I don't know".
   Instead provide the MOST SPECIFIC sub-document link from the table below:

   TOPIC                                    | LINK
   ─────────────────────────────────────────────────────────────────────────────────
   Office hours / staff info / faculty      | https://docs.google.com/document/d/1eFvIFMbwjOHDsKnyE2vF17no_KGrzf3AIdtJok14BGI/edit
   CS/CSAI/DSAI/SW/HCI/IT course materials  | https://docs.google.com/document/d/17uIMIfaCAiazIK53pHY33GH_JldF_24-qLr_MtyFE-c/edit
   Math / MATH course materials             | https://docs.google.com/document/d/1DpmNY24lBGnJKbUWiTSWcCEMXVctAHJUBvRER8oubYU/edit
   Engineering / CIE / SPC / REE materials  | https://docs.google.com/document/d/1kYSafBd9ii4Whck6D1hNm9p6XXJGQOpf995cJ6pleiE/edit
   English / Humanities / SCH / ENGL        | https://docs.google.com/document/d/1PD7PZ4pPVRHcHVelXQxLpn5M1HJrxwsmRrBjBDNqL8s/edit
   Course books / textbooks / references    | https://docs.google.com/document/d/1UDKznWf3Yro2Rw8H8sHHMmHwTxu78Lg8MhX1EPoFeck/edit
   Coding practice / HackerRank / LeetCode  | https://docs.google.com/document/d/1ACSuOBTeFa6vcJuPP3RJRTrhCZRbA2L06Fq3M72KCF8/edit
   VPN / self-service portal / LTS / tools  | https://docs.google.com/document/d/1E0TPoMprNY1WwO7YPfZG2HwSa-OZLPiqwExbq8fRm4c/edit
   WhatsApp groups / clubs / socials        | https://docs.google.com/document/d/16NQC9RIQ203dMnZcasaY6laKYU3YRx_krbI_I_LyJZ8/edit
   Drive folders / Day One / collections    | https://docs.google.com/document/d/1j8qSKShjGkVAr-CbeHhOffphRLmQZgB4RXOAdWzjJ34/edit
   Anything else (general campus info)      | https://docs.google.com/document/d/1x61s5Y3Xz36GB7Tz7B0R2gfUF6VXXMs8NjbH5z9ka6E/edit

   Format: "I don't have the specific [detail] in my knowledge base, but you can find it
   here: [URL]"
5. If the context genuinely does not contain the answer and it is NOT a LinkTree-type question,
   say: "I don't have that specific detail in my knowledge base. Please contact the
   Academic Advising Office or visit https://www.zewailcity.edu.eg/contact"
6. Do NOT include any source citations, footnotes, or "[Source X]" references in
   your answer. Sources are shown to the student separately -- keep the answer clean.
7. Be precise and helpful. For multi-step questions walk through each step.
8. Use the student's profile (program, semester, GPA) to personalise answers when given.
9. Credit hours / graduation requirements: look for explicit numbers like "132 credit hours",
   "114 credit hours", "minimum of X credits" and state them clearly with the correct program.
10. School-specific queries: when the question mentions Engineering/ENGR, CSAI, SCI, or BUS,
    focus on that school's data. Do NOT mix graduation requirements across schools.
11. Course code lookups (e.g. "what is CSAI 201?"): look for the code in table rows
    "CODE | Course Title | Cr | L | P | Prerequisite" and extract the Course Title.
    Also check prerequisite lists: "CSAI 201, Data Structures" means CSAI 201 = Data Structures.
12. Course material queries (e.g. "where can I find slides for MATH 101?"): look in the
    LinkTree course sections for direct links or drive folders. Provide the link if found.
    If not found in context, refer to the LinkTree URL from rule 4.
13. Faculty / director queries: look for faculty listings with names, titles, programs,
    and emails. Match partial names and list all found faculty for the requested school.
14. Name queries: context may use different transliterations of Arabic names.
    If a name sounds similar to the one asked about, provide their information and note
    the exact spelling as it appears in the records.
15. Follow-up questions: use the conversation history to understand what school/topic was
    being discussed. "So they are 3 or 2?" after a question about Engineering programs means
    "how many Engineering programs?" -- answer based on that context.
"""


# ── CampusRAG class ────────────────────────────────────────────────────────────

class CampusRAG:
    """
    Retrieve-and-generate pipeline for Zewail City academic advising.

    Usage:
        rag = CampusRAG()
        answer, sources = rag.answer("What courses are required for CSAI?")
    """

    def __init__(
        self,
        chroma_dir:      Optional[str] = None,
        collection_name: str = COLLECTION_NAME,
        embed_model:     str = EMBED_MODEL,
        chat_model:      str = CHAT_MODEL,
        openai_api_key:  Optional[str] = None,
    ) -> None:
        import chromadb
        from openai import OpenAI

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set. Create a .env file or set the env var.")

        _OPENROUTER_BASE = "https://openrouter.ai/api/v1"

        if api_key.startswith("gsk_"):
            # Groq for chat (free tier); embeddings use a separate OPENAI_EMBED_KEY if set,
            # otherwise fall back to keyword-based retrieval automatically at query time.
            self._oai_chat  = OpenAI(api_key=api_key, base_url=_GROQ_BASE)
            embed_key       = os.environ.get("OPENAI_EMBED_KEY")
            self._oai_embed = OpenAI(api_key=embed_key) if embed_key else None
            self._oai       = self._oai_chat
            self._chat_model = _GROQ_MODEL if chat_model == CHAT_MODEL else chat_model

        elif api_key.startswith("sk-or-v1-"):
            # OpenRouter for chat; OpenAI for embeddings (OpenRouter has no embedding API)
            self._oai_chat  = OpenAI(api_key=api_key, base_url=_OPENROUTER_BASE)
            embed_key = os.environ.get("OPENAI_EMBED_KEY") or api_key
            self._oai_embed = OpenAI(api_key=embed_key)
            self._oai       = self._oai_chat          # fallback alias
            if chat_model == CHAT_MODEL:
                self._chat_model = "openai/gpt-4o"    # OpenRouter model name

        else:
            self._oai       = OpenAI(api_key=api_key)
            self._oai_chat  = self._oai
            self._oai_embed = self._oai
            self._chat_model = chat_model

        self._embed_model = embed_model

        persist = str(chroma_dir or CHROMA_DIR)
        db      = chromadb.PersistentClient(path=persist)
        self._col = db.get_collection(collection_name)

    # ── Retrieval ───────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        if self._oai_embed is None:
            raise RuntimeError("No embedding client — keyword fallback will be used.")
        return self._oai_embed.embeddings.create(
            model=self._embed_model, input=[text]
        ).data[0].embedding

    def _keyword_retrieve(self, query: str, n: int) -> list[RetrievedChunk]:
        """
        Keyword-based retrieval used when an embedding client is unavailable.
        Searches ChromaDB's stored document text directly (no embedding call needed).
        Preserves the same source-diversity cap (max 2 chunks per source) as the
        semantic path so context quality stays as consistent as possible.
        """
        # Course codes get highest priority as search terms
        course_codes = [
            f"{m.group(1)} {m.group(2)}"
            for m in _COURSE_CODE_RE.finditer(query.upper())
        ]
        # Content keywords: >=4 chars, not stopwords
        content_words = [
            w.lower() for w in re.findall(r'\b[a-zA-Z]{4,}\b', query)
            if w.lower() not in _KEYWORD_STOPWORDS
        ]
        # Deduplicate while preserving order (course codes first)
        seen: set[str] = set()
        search_terms: list[str] = []
        for t in course_codes + content_words:
            if t not in seen:
                seen.add(t)
                search_terms.append(t)

        # Score each unique chunk by how many search terms appear in it
        scored: dict[str, tuple[int, RetrievedChunk]] = {}
        for term in search_terms:
            try:
                res = self._col.get(
                    where_document={"$contains": term},
                    include=["documents", "metadatas"],
                )
                for cid, doc, meta in zip(
                    res.get("ids", []),
                    res.get("documents", []),
                    res.get("metadatas", []),
                ):
                    if cid in scored:
                        hits, chunk = scored[cid]
                        scored[cid] = (hits + 1, chunk)
                    else:
                        scored[cid] = (1, RetrievedChunk(
                            chunk_id    = cid,
                            text        = doc,
                            source      = meta.get("source", ""),
                            source_type = meta.get("source_type", ""),
                            category    = meta.get("category", ""),
                            page        = meta.get("page", ""),
                            score       = 0.0,
                        ))
            except Exception:
                continue

        if not scored:
            return []

        ranked = sorted(scored.values(), key=lambda x: -x[0])
        max_hits = ranked[0][0]
        for hits, chunk in ranked:
            chunk.score = round(hits / max_hits, 4)

        # Same source-diversity cap as the semantic path
        source_counts: dict[str, int] = {}
        diverse: list[RetrievedChunk] = []
        for _hits, chunk in ranked:
            cnt = source_counts.get(chunk.source, 0)
            if cnt < 2:
                diverse.append(chunk)
                source_counts[chunk.source] = cnt + 1
            if len(diverse) >= n:
                break
        return diverse

    def _query_chroma(
        self,
        emb: list[float],
        n: int,
        where_doc: dict | None = None,
    ) -> list[RetrievedChunk]:
        kwargs: dict = dict(
            query_embeddings=[emb],
            n_results=max(1, min(n, self._col.count())),
            include=["documents", "metadatas", "distances"],
        )
        if where_doc:
            kwargs["where_document"] = where_doc
        results = self._col.query(**kwargs)
        chunks = []
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(RetrievedChunk(
                chunk_id    = cid,
                text        = doc,
                source      = meta.get("source", ""),
                source_type = meta.get("source_type", ""),
                category    = meta.get("category", ""),
                page        = meta.get("page", ""),
                score       = round(max(0.0, 1.0 - dist), 4),
            ))
        return chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
    ) -> tuple[list[RetrievedChunk], str]:
        """
        Smart retrieval returning (chunks, query_note).

        Enhancements:
        1. Arabic name-prefix normalisation — strips El/Al/Abd before embedding
           so "Dr el Reabey" finds "Dr Rabeay" instead of unrelated El-X names.
        2. Course-code anchor — when the query names a course code (e.g. CSAI 201)
           the best 2 matching chunks are *guaranteed* to appear in the result,
           bypassing the per-source diversity cap that would otherwise block them.
        3. Source diversity — max 2 chunks per source file so near-duplicate table
           pages from the same PDF cannot flood all result slots.

        Returns a (chunks, note) tuple where note is injected into the GPT context
        when query normalisation changed the search terms.
        """
        query_note = ""

        # ── 1. Normalise query before embedding ─────────────────────────────
        normalised = _ARABIC_PREFIX_RE.sub("", query).strip() or query
        if normalised != query:
            query_note = (
                f"[Search note: the query '{query}' was normalised to '{normalised}' "
                f"by removing Arabic name prefixes. Names in the retrieved documents "
                f"may use different transliterations of the same person's name — treat "
                f"phonetically similar names as referring to the same individual.]"
            )

        # ── 2. Try semantic search; fall back to keyword search if unavailable ──
        try:
            emb = self._embed(normalised)
            use_semantic = True
        except Exception:
            use_semantic = False

        if not use_semantic:
            return self._keyword_retrieve(normalised, top_k), query_note

        # ── 3. Semantic search over full collection ──────────────────────────
        n_candidates = min(top_k * 4, self._col.count())
        candidates: list[RetrievedChunk] = self._query_chroma(emb, n_candidates)

        # ── 4. Course-code anchor (guaranteed slots) ─────────────────────────
        guaranteed: list[RetrievedChunk] = []
        guaranteed_ids: set[str] = set()
        code_match = _COURSE_CODE_RE.search(query.upper())
        if code_match:
            code_str = f"{code_match.group(1)} {code_match.group(2)}"
            try:
                anchored = self._query_chroma(
                    emb,
                    min(top_k * 2, self._col.count()),
                    where_doc={"$contains": code_str},
                )
                anchored.sort(key=lambda c: -c.score)
                # Guarantee the best 2 anchored chunks regardless of diversity
                guaranteed = anchored[:2]
                guaranteed_ids = {c.chunk_id for c in guaranteed}
            except Exception:
                pass

        # ── 5. Source-diversity filter on remaining slots ────────────────────
        # Pre-seed source counts from guaranteed slots so fill respects them
        source_counts: dict[str, int] = {}
        for g in guaranteed:
            source_counts[g.source] = source_counts.get(g.source, 0) + 1

        remaining_slots = top_k - len(guaranteed)
        diverse: list[RetrievedChunk] = list(guaranteed)

        candidates.sort(key=lambda c: -c.score)
        for c in candidates:
            if c.chunk_id in guaranteed_ids:
                continue
            cnt = source_counts.get(c.source, 0)
            if cnt < 2:
                diverse.append(c)
                source_counts[c.source] = cnt + 1
            if len(diverse) - len(guaranteed) == remaining_slots:
                break

        return diverse, query_note

    # ── Generation ──────────────────────────────────────────────────────────────

    def generate(
        self,
        query:        str,
        chunks:       list[RetrievedChunk],
        history:      Optional[list[dict]] = None,
        temperature:  float = 0.2,
        query_note:   str = "",
    ) -> str:
        """
        Call GPT-4o with the retrieved context and conversation history.
        Returns the assistant's answer as a string.
        """
        if not chunks:
            context_text = "No relevant documents found in the knowledge base."
        else:
            parts = []
            if query_note:
                parts.append(query_note)
            for i, c in enumerate(chunks, 1):
                src_label = c.source
                if c.page:
                    src_label += f" (page {c.page})"
                parts.append(
                    f"[Source {i} | {c.category} | {src_label}]\n{c.text}"
                )
            context_text = "\n\n---\n\n".join(parts)

        try:
            from feedback_logger import build_patched_prompt as _bpp
            _effective_prompt = _bpp(SYSTEM_PROMPT)
        except Exception:
            _effective_prompt = SYSTEM_PROMPT
        messages: list[dict] = [{"role": "system", "content": _effective_prompt}]

        # Inject conversation history (keep last 6 turns to stay within context)
        if history:
            messages.extend(history[-12:])

        messages.append({
            "role": "user",
            "content": (
                f"CONTEXT FROM KNOWLEDGE BASE:\n{context_text}\n\n"
                f"STUDENT QUESTION:\n{query}"
            ),
        })

        resp = self._oai_chat.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip()

    # ── Combined entry-point ────────────────────────────────────────────────────

    def answer(
        self,
        query:    str,
        history:  Optional[list[dict]] = None,
        top_k:    int   = 6,
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        Full pipeline: retrieve relevant chunks → generate answer.

        Returns:
            (answer_text, retrieved_chunks)
        """
        chunks, note = self.retrieve(query, top_k=top_k)
        answer = self.generate(query, chunks, history=history, query_note=note)
        return answer, chunks


    # ── Intent Classification ───────────────────────────────────────────────────

    def classify_intent(self, query: str) -> str:
        """
        Classify the query into one of:
          courses | faculty | prerequisites | graduation | scholarships | general

        Uses gpt-4o-mini (fast, cheap). Falls back to 'general' on any error.
        """
        try:
            resp = self._oai_chat.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        "Classify this university academic query into exactly ONE category:\n"
                        "courses | faculty | prerequisites | graduation | scholarships | general\n\n"
                        f"Query: {query}\n\n"
                        "Reply with the single category word only."
                    ),
                }],
                max_tokens=10,
                temperature=0,
            )
            intent = resp.choices[0].message.content.strip().lower().split()[0]
            _VALID = {"courses", "faculty", "prerequisites", "graduation", "scholarships", "general"}
            return intent if intent in _VALID else "general"
        except Exception:
            return "general"

    # ── LLM-based Reranking ─────────────────────────────────────────────────────

    def rerank(
        self,
        query:  str,
        chunks: list[RetrievedChunk],
        top_n:  int = 5,
    ) -> list[RetrievedChunk]:
        """
        Rerank retrieved chunks with a single gpt-4o-mini call.
        Returns up to top_n chunks ordered by relevance to query.
        Falls back to original cosine-score order on any error.
        """
        if len(chunks) <= 2:
            return chunks
        try:
            doc_block = "\n".join(
                f"[{i}] {c.text[:280].replace(chr(10), ' ')}"
                for i, c in enumerate(chunks)
            )
            resp = self._oai_chat.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Documents:\n{doc_block}\n\n"
                        f"Output the {min(top_n, len(chunks))} most relevant document indices "
                        f"in descending relevance order. Comma-separated integers only."
                    ),
                }],
                max_tokens=40,
                temperature=0,
            )
            raw     = resp.choices[0].message.content.strip()
            indices = [int(x) for x in re.findall(r'\d+', raw) if int(x) < len(chunks)]
            seen: set[int] = set()
            reranked: list[RetrievedChunk] = []
            for i in indices[:top_n]:
                if i not in seen:
                    reranked.append(chunks[i])
                    seen.add(i)
            for i, c in enumerate(chunks):
                if i not in seen:
                    reranked.append(c)
            return reranked
        except Exception:
            return chunks

    # ── Streaming Generation ────────────────────────────────────────────────────

    def generate_stream(
        self,
        query:       str,
        chunks:      list[RetrievedChunk],
        history:     Optional[list[dict]] = None,
        temperature: float = 0.2,
        query_note:  str = "",
    ):
        """
        Streaming version of generate(). Yields token strings progressively.
        Identical prompt construction to generate() — safe to swap in.
        """
        if not chunks:
            context_text = "No relevant documents found in the knowledge base."
        else:
            parts = []
            if query_note:
                parts.append(query_note)
            for i, c in enumerate(chunks, 1):
                src_label = c.source + (f" (page {c.page})" if c.page else "")
                parts.append(f"[Source {i} | {c.category} | {src_label}]\n{c.text}")
            context_text = "\n\n---\n\n".join(parts)

        try:
            from feedback_logger import build_patched_prompt as _bpp
            _effective_prompt = _bpp(SYSTEM_PROMPT)
        except Exception:
            _effective_prompt = SYSTEM_PROMPT
        messages: list[dict] = [{"role": "system", "content": _effective_prompt}]
        if history:
            messages.extend(history[-12:])
        messages.append({
            "role": "user",
            "content": (
                f"CONTEXT FROM KNOWLEDGE BASE:\n{context_text}\n\n"
                f"STUDENT QUESTION:\n{query}"
            ),
        })

        stream = self._oai_chat.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=1200,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Follow-up Suggestions ───────────────────────────────────────────────────

    def suggest_followups(self, query: str, answer: str, n: int = 3) -> list[str]:
        """Generate n short follow-up questions using gpt-4o-mini."""
        try:
            resp = self._oai_chat.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": (
                        f"A Zewail City student asked: '{query}'\n"
                        f"Advisor answered: '{answer[:350]}'\n\n"
                        f"Write {n} short natural follow-up questions the student might ask next. "
                        f"One per line, no numbering or bullets, max 12 words each."
                    ),
                }],
                max_tokens=120,
                temperature=0.8,
            )
            lines = [
                l.strip("- •·1234567890. )")
                for l in resp.choices[0].message.content.strip().split("\n")
                if l.strip("- •·1234567890. )")
            ]
            return lines[:n]
        except Exception:
            return []

    # ── Agentic Tool Execution ──────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict) -> str:
        """Run one CAMPUS_TOOLS function call and return the result string."""
        if name == "search_curriculum":
            keyword = args.get("keyword", "")
            results = self._keyword_retrieve(keyword, n=4)
            if not results:
                return f"No courses found matching '{keyword}'."
            return "\n".join(
                f"- [{c.category}] {c.source}: {c.text[:200].replace(chr(10),' ')}"
                for c in results
            )

        elif name == "calculate_gpa":
            grades = args.get("grades", [])
            total_pts = total_cr = 0.0
            for g in grades:
                letter = str(g.get("grade", "")).strip().upper()
                pts    = _GRADE_POINTS.get(letter)
                cr     = float(g.get("credits", 3))
                if pts is not None:
                    total_pts += pts * cr
                    total_cr  += cr
            if total_cr == 0:
                return "Could not calculate GPA — no valid grades provided."
            return (
                f"Calculated GPA: {total_pts / total_cr:.2f} "
                f"over {total_cr:.0f} credit hours."
            )

        elif name == "get_graduation_requirements":
            school = str(args.get("school", "")).upper().strip()
            total  = _GRAD_CREDITS.get(school)
            if total is None:
                return (
                    f"Unknown school '{school}'. "
                    f"Valid: CSAI (132 cr), BUS (114 cr), SCI (132 cr), ENGR (~140 cr)."
                )
            return f"{school} requires {total} total credit hours to graduate from Zewail City."

        elif name == "get_prerequisites":
            code    = str(args.get("course_code", "")).upper().strip()
            results = self._keyword_retrieve(f"{code} prerequisite", n=3)
            if not results:
                return f"No prerequisite information found for {code}."
            return "\n".join(
                f"- {c.text[:300].replace(chr(10),' ')}" for c in results
            )

        return f"Unknown tool: {name}"

    def answer_with_tools(
        self,
        query:      str,
        history:    Optional[list[dict]] = None,
        top_k:      int = 6,
        max_rounds: int = 3,
    ) -> tuple[str, list[RetrievedChunk]]:
        """
        Agentic answer: runs an OpenAI function-calling loop.
        The model may call CAMPUS_TOOLS one or more times before giving a final answer.
        """
        import json as _json

        chunks, note = self.retrieve(query, top_k=top_k)

        if not chunks:
            context_text = "No relevant documents found in the knowledge base."
        else:
            parts = [note] if note else []
            for i, c in enumerate(chunks, 1):
                src_label = c.source + (f" (page {c.page})" if c.page else "")
                parts.append(f"[Source {i} | {c.category} | {src_label}]\n{c.text}")
            context_text = "\n\n---\n\n".join(parts)

        try:
            from feedback_logger import build_patched_prompt as _bpp
            _effective_prompt = _bpp(SYSTEM_PROMPT)
        except Exception:
            _effective_prompt = SYSTEM_PROMPT
        messages: list[dict] = [{"role": "system", "content": _effective_prompt}]
        if history:
            messages.extend(history[-12:])
        messages.append({
            "role": "user",
            "content": (
                f"CONTEXT FROM KNOWLEDGE BASE:\n{context_text}\n\n"
                f"STUDENT QUESTION:\n{query}"
            ),
        })

        for _ in range(max_rounds):
            resp = self._oai_chat.chat.completions.create(
                model=self._chat_model,
                messages=messages,
                tools=CAMPUS_TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1200,
            )
            msg = resp.choices[0].message

            if not getattr(msg, "tool_calls", None):
                return msg.content.strip(), chunks

            messages.append(msg)
            for tc in msg.tool_calls:
                try:
                    tool_args = _json.loads(tc.function.arguments)
                except Exception:
                    tool_args = {}
                result = self._execute_tool(tc.function.name, tool_args)
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })

        # Safety exit: one final non-tool pass
        resp = self._oai_chat.chat.completions.create(
            model=self._chat_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip(), chunks


# ── Standalone demo ────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    "What are the graduation requirements for undergraduate students at Zewail City?",
    "Can you explain the academic probation policy?",
    "What scholarships are available for undergraduate students?",
    "What research institutes are available at Zewail City?",
    "How many credit hours does a student need to complete to graduate?",
]


def demo() -> None:
    print("Phase 5 - RAG Pipeline Demo")
    print("=" * 62)

    try:
        rag = CampusRAG()
    except Exception as exc:
        print(f"ERROR initialising RAG pipeline: {exc}")
        print("Make sure Phase 4 has been run and OPENAI_API_KEY is set.")
        return

    print(f"  Collection size : {rag._col.count()} chunks")
    print(f"  Chat model      : {rag._chat_model}")
    print()

    for i, q in enumerate(SAMPLE_QUERIES, 1):
        print(f"Query {i}/{len(SAMPLE_QUERIES)}: {q}")
        print("-" * 62)
        t0 = time.time()
        ans, chunks = rag.answer(q)
        elapsed = time.time() - t0

        print(f"Answer ({elapsed:.1f}s):\n{ans}")
        print()
        print("Retrieved sources:")
        for c in chunks:
            src = c.source
            if c.page:
                src += f" p.{c.page}"
            print(f"  [{c.score:.3f}] [{c.category}] {src}")
        print("=" * 62)
        print()


if __name__ == "__main__":
    demo()
