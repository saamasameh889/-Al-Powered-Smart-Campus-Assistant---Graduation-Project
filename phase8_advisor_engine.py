#!/usr/bin/env python3
"""
phase8_advisor_engine.py — Zewail Campus Digital Assistant
═══════════════════════════════════════════════════════════════════════════════
Academic Advisor AI Engine — all engines in one module.

Engines implemented:
  1.  Student Profile Engine  — persistent academic state per session
  2.  Academic Memory Engine  — conversation-aware profile updates
  3.  Curriculum Engine       — structured course catalog (courses.json)
  4.  Prerequisite Engine     — NetworkX directed prerequisite graph
  5.  Eligibility Engine      — eligible / blocked course computation
  6.  Academic Planning Engine— SAFE / BALANCED / FAST semester plans
  7.  Graduation Engine       — remaining credits + expected graduation
  8.  Decision Support Engine — every recommendation is justified
  9.  Academic Risk Engine    — LOW / MEDIUM / HIGH risk assessment
  10. RAG Knowledge Engine    — grounded in official Zewail documents
  11. Campus Assistant        — delegates general queries to RAG

Public API (used by phase6):
    from phase8_advisor_engine import (
        StudentProfile,
        update_profile,
        AdvisorEngine,
    )
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
COURSES_FILE = PROJECT_ROOT / "data" / "curriculum" / "courses.json"

# ── Total credit requirements per school ───────────────────────────────────────
_TOTAL_CREDITS = {
    "CSAI": 132,
    "SCI":  132,
    "BUS":  114,
    "ENGR": 140,  # approximate; varies by program
}

# Typical semester credit loads
_LOAD_SAFE     = (12, 15)   # credits (min, typical)
_LOAD_BALANCED = (15, 18)
_LOAD_FAST     = (18, 21)

# GPA thresholds
_GPA_PROBATION  = 2.0
_GPA_WARNING    = 2.5
_GPA_GOOD       = 3.0


# ══════════════════════════════════════════════════════════════════════════════
#  Regex for profile extraction
# ══════════════════════════════════════════════════════════════════════════════

_PROGRAM_RE = re.compile(
    r'\b(csai|sci|bus|engr|computer science|data science|dsai|hci|'
    r'computer engineering|biomedical|nanoscience|physics of|physics|finance|'
    r'business analytics|actuarial|operations management|entrepreneurship|cie|'
    r'aerospace|nanotechnology|renewable energy|environmental engineering)\b',
    re.I,
)
_SEMESTER_RE = re.compile(
    r'semester\s*(\d+)|year\s*(\d+)|(\d+)(?:st|nd|rd|th)\s+(?:semester|year)',
    re.I,
)
_GPA_RE = re.compile(
    r'\b([0-3]\.\d{1,2}|4\.0)\s*(?:gpa|cgpa)'
    r'|\bgpa\s*(?:is|of|:)?\s*([0-3]\.\d{1,2}|4\.0)\b',
    re.I,
)
_COMPLETED_RE = re.compile(
    r'(?:completed?|passed?|finished?|took|done with|have taken|'
    r'already took|i passed|i completed|i finished|i have done)\s+'
    r'(.+?)(?=\.|$|\band\b\s+(?:i\s+)?(?:failed|currently|now))',
    re.I,
)
_FAILED_RE = re.compile(
    r'(?:failed?|did not pass|did not complete|need to retake|retaking|'
    r'withdrew from|dropped?)\s+'
    r'(.+?)(?=\.|$)',
    re.I,
)
_CURRENT_RE = re.compile(
    r'(?:currently taking|taking now|enrolled in|registered for|'
    r'this semester i(?:\'m| am) taking|this semester i have)\s+'
    r'(.+?)(?=\.|$)',
    re.I,
)
_CREDITS_RE  = re.compile(r'(?:completed|finished|have|earned)\s+(\d+)\s+credits?', re.I)
_CODE_RE     = re.compile(r'\b([A-Z]{2,6})\s+(\d{3,4}[A-Z]?)\b')

# Canonical major + school from keywords
_MAJOR_MAP: dict[str, tuple[str, str]] = {
    "COMPUTER SCIENCE":              ("CS",                                    "CSAI"),
    "DATA SCIENCE":                  ("DSAI",                                  "CSAI"),
    "DSAI":                          ("DSAI",                                  "CSAI"),
    "HCI":                           ("HCI",                                   "CSAI"),
    "COMPUTER ENGINEERING":          ("CE",                                    "CSAI"),
    "CSAI":                          ("CSAI",                                  "CSAI"),
    "BIOMEDICAL":                    ("Biomedical Sciences",                   "SCI"),
    "NANOSCIENCE":                   ("Nanoscience",                           "SCI"),
    "PHYSICS OF":                    ("Physics of the Universe",               "SCI"),
    "PHYSICS":                       ("Physics of the Universe",               "SCI"),
    "SCI":                           ("SCI",                                   "SCI"),
    "FINANCE":                       ("Finance",                               "BUS"),
    "BUSINESS ANALYTICS":            ("Business Analytics",                    "BUS"),
    "ACTUARIAL":                     ("Actuarial Analysis & Risk Management",  "BUS"),
    "OPERATIONS MANAGEMENT":         ("Operations Management",                 "BUS"),
    "ENTREPRENEURSHIP":              ("Entrepreneurship & Innovation Mgmt",    "BUS"),
    "BUS":                           ("BUS",                                   "BUS"),
    "AEROSPACE":                     ("Aerospace Engineering",                 "ENGR"),
    "NANOTECHNOLOGY":                ("Nanotechnology & Nanoelectronics",      "ENGR"),
    "CIE":                           ("Communications & Information Eng.",     "ENGR"),
    "RENEWABLE ENERGY":              ("Renewable Energy Engineering",          "ENGR"),
    "ENVIRONMENTAL ENGINEERING":     ("Environmental Engineering",             "ENGR"),
    "ENGR":                          ("ENGR",                                  "ENGR"),
}

# Intent keywords
_PLANNING_KW = frozenset([
    "what should i take", "courses to take", "next semester", "plan for next",
    "semester plan", "what courses should", "what can i take", "recommend course",
    "advise me on courses", "course recommendation", "register for",
    "what to register", "which courses am i eligible", "what am i eligible",
    "plan my semester", "suggest courses", "course load", "course selection",
    "what courses do i take", "what should i study", "schedule for next",
    "enrollment plan", "what courses to enroll", "courses for semester",
])
_GRADUATION_KW = frozenset([
    "graduation", "graduate", "when will i finish", "when will i graduate",
    "how many semesters", "semesters left", "remaining credits",
    "how many credits left", "credits remaining", "on track to graduate",
    "graduation timeline", "graduation plan", "graduation roadmap",
    "time to graduate", "expected graduation", "when do i graduate",
    "how long until i graduate", "years to graduate",
])
_PREREQ_KW = frozenset([
    "prerequisite", "prereq", "required before", "required for",
    "can i take", "am i allowed to take", "before taking",
    "what do i need before", "need for", "unlocked by", "unlock",
    "requirements for", "what must i complete before",
])
_RISK_KW = frozenset([
    "academic risk", "gpa risk", "probation", "academic warning",
    "heavy load", "too many courses", "overload", "academic standing",
    "will i fail", "danger of failing",
])
# Phrasing that signals the student is asking about THEIR OWN risk/standing,
# as opposed to a general policy/definition question like "what is the
# probation policy?". Only the former should route to the personal risk engine.
_SELF_RISK_RE = re.compile(
    r'\b(am i|will i|my risk|my gpa|my academic standing|my workload|'
    r'do i|could i|can i|should i|i be put|i fail|i go on|i get put|'
    r'for me|i\'m taking|i am taking)\b',
    re.I,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Engine 1: Student Profile
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StudentProfile:
    """
    Persistent academic profile updated incrementally from conversation.
    Serialised as a plain dict in ConversationSession.user_profile.
    """
    major:             str   = ""
    school:            str   = ""
    semester:          int   = 0
    gpa:               float = 0.0
    completed_courses: list  = field(default_factory=list)   # codes or names
    failed_courses:    list  = field(default_factory=list)
    current_courses:   list  = field(default_factory=list)
    completed_credits: int   = 0
    preferences:       dict  = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce numeric fields — guards against old session files that stored
        gpa as '2.8' (str) and semester as 'Semester 5' (str)."""
        # semester: extract first integer found, e.g. "Semester 5" → 5
        if not isinstance(self.semester, int):
            try:
                m = re.search(r'\d+', str(self.semester))
                self.semester = int(m.group()) if m else 0
            except Exception:
                self.semester = 0

        # gpa: coerce to float
        if not isinstance(self.gpa, (int, float)):
            try:
                self.gpa = float(self.gpa) if self.gpa else 0.0
            except Exception:
                self.gpa = 0.0

        # completed_credits: coerce to int
        if not isinstance(self.completed_credits, int):
            try:
                self.completed_credits = int(self.completed_credits) if self.completed_credits else 0
            except Exception:
                self.completed_credits = 0

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StudentProfile":
        """Load from dict, handling both old string format and new list format."""
        valid = set(cls.__dataclass_fields__)
        kwargs: dict = {}
        for k, v in d.items():
            if k not in valid:
                continue
            # Back-compat: old sessions stored courses as comma-separated strings
            if k in ("completed_courses", "failed_courses", "current_courses"):
                if isinstance(v, str):
                    v = [c.strip() for c in v.split(",") if c.strip()] if v else []
            kwargs[k] = v
        return cls(**kwargs)  # __post_init__ handles all remaining type coercions

    # ── Status helpers ─────────────────────────────────────────────────────────

    def has_academic_context(self) -> bool:
        """True when we know enough to give personalised advice."""
        return bool(
            self.school or self.major or
            self.completed_courses or self.failed_courses or
            (self.gpa > 0 and self.semester > 0)
        )

    def is_ready_for_planning(self) -> bool:
        """True when we have enough for a full semester plan."""
        return bool(
            (self.school or self.major) and
            (self.completed_courses or self.completed_credits > 0 or self.semester > 0)
        )

    def total_credits_needed(self) -> int:
        return _TOTAL_CREDITS.get(self.school, 132)

    def estimated_remaining_credits(self) -> int:
        needed = self.total_credits_needed()
        if self.completed_credits:
            return max(0, needed - self.completed_credits)
        # Estimate from semester number (avg 15 cr/sem)
        if self.semester:
            done = (self.semester - 1) * 15
            return max(0, needed - done)
        return needed

    # ── Formatted summaries ────────────────────────────────────────────────────

    def summary_for_prompt(self) -> str:
        """Multi-line string injected into advisor prompts."""
        parts: list[str] = []
        if self.major:
            parts.append(f"Major: {self.major}")
        if self.school:
            parts.append(f"School: {self.school}")
        if self.semester:
            parts.append(f"Current Semester: {self.semester}")
        if self.gpa:
            flag = ""
            if   self.gpa < _GPA_PROBATION: flag = " ⚠ PROBATION RISK"
            elif self.gpa < _GPA_WARNING:   flag = " ⚠ LOW GPA"
            parts.append(f"GPA: {self.gpa}{flag}")
        if self.completed_credits:
            remaining = self.estimated_remaining_credits()
            parts.append(
                f"Credits completed: {self.completed_credits} "
                f"(~{remaining} remaining to graduate)"
            )
        if self.completed_courses:
            parts.append(
                f"Completed courses ({len(self.completed_courses)}): "
                + ", ".join(self.completed_courses)
            )
        if self.failed_courses:
            parts.append(
                f"FAILED / Must retake: "
                + ", ".join(self.failed_courses)
            )
        if self.current_courses:
            parts.append(
                f"Currently enrolled: "
                + ", ".join(self.current_courses)
            )
        if self.preferences:
            for k, v in self.preferences.items():
                parts.append(f"Preference — {k}: {v}")
        return "\n".join(parts) if parts else "No profile shared yet."

    def sidebar_items(self) -> list[tuple[str, str]]:
        """(label, value) pairs for the Streamlit sidebar."""
        items: list[tuple[str, str]] = []
        if self.major:
            items.append(("Major", self.major))
        if self.school:
            items.append(("School", self.school))
        if self.semester:
            items.append(("Semester", str(self.semester)))
        if self.gpa:
            gpa_str = str(self.gpa)
            if self.gpa < _GPA_PROBATION:
                gpa_str += " ⚠"
            items.append(("GPA", gpa_str))
        if self.completed_credits:
            items.append(("Credits Done", str(self.completed_credits)))
        if self.completed_courses:
            items.append(("Completed", f"{len(self.completed_courses)} courses"))
        if self.failed_courses:
            items.append(("Must Retake", ", ".join(self.failed_courses[:3])
                          + ("…" if len(self.failed_courses) > 3 else "")))
        if self.current_courses:
            items.append(("Enrolled Now", ", ".join(self.current_courses[:2])
                          + ("…" if len(self.current_courses) > 2 else "")))
        return items


# ══════════════════════════════════════════════════════════════════════════════
#  Engine 2: Academic Memory — profile extraction from conversation text
# ══════════════════════════════════════════════════════════════════════════════

def _extract_course_refs(text: str) -> list[str]:
    """
    Pull course identifiers from a text fragment.
    Prefers explicit codes (CSAI 201); falls back to noun phrases.
    """
    codes = [f"{m.group(1)} {m.group(2)}" for m in _CODE_RE.finditer(text.upper())]
    if codes:
        return codes
    results: list[str] = []
    for part in re.split(r',|;|\band\b|\bor\b|\bplus\b', text, flags=re.I):
        part = part.strip().strip('."\'()')
        if 3 < len(part) < 80:
            results.append(part)
    return results


def update_profile(text: str, current: StudentProfile) -> StudentProfile:
    """
    Engine 2: update student profile from a new message.
    Returns a new StudentProfile (never mutates the input).
    """
    p = StudentProfile.from_dict(current.to_dict())  # copy

    # ── Major / school ─────────────────────────────────────────────────────────
    m = _PROGRAM_RE.search(text)
    if m:
        raw = m.group(0).upper()
        # Find the best matching key (longest prefix match)
        best_key: Optional[str] = None
        for k in _MAJOR_MAP:
            if raw.startswith(k) or k in raw:
                if best_key is None or len(k) > len(best_key):
                    best_key = k
        if best_key:
            major, school = _MAJOR_MAP[best_key]
            if not p.major:
                p.major = major
            if not p.school:
                p.school = school

    # ── Semester ──────────────────────────────────────────────────────────────
    m = _SEMESTER_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2) or m.group(3)
        if raw:
            p.semester = int(raw)   # allow override (student may correct themselves)

    # ── GPA ───────────────────────────────────────────────────────────────────
    m = _GPA_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        if raw:
            p.gpa = float(raw)

    # ── Completed credits ─────────────────────────────────────────────────────
    m = _CREDITS_RE.search(text)
    if m:
        p.completed_credits = int(m.group(1))

    # ── Completed courses ─────────────────────────────────────────────────────
    for m in _COMPLETED_RE.finditer(text):
        for ref in _extract_course_refs(m.group(1)):
            if ref and ref.upper() not in [c.upper() for c in p.completed_courses]:
                p.completed_courses.append(ref.upper() if _CODE_RE.match(ref) else ref)

    # ── Failed courses ────────────────────────────────────────────────────────
    for m in _FAILED_RE.finditer(text):
        for ref in _extract_course_refs(m.group(1)):
            if ref and ref.upper() not in [c.upper() for c in p.failed_courses]:
                p.failed_courses.append(ref.upper() if _CODE_RE.match(ref) else ref)

    # ── Current courses ───────────────────────────────────────────────────────
    for m in _CURRENT_RE.finditer(text):
        for ref in _extract_course_refs(m.group(1)):
            if ref and ref.upper() not in [c.upper() for c in p.current_courses]:
                p.current_courses.append(ref.upper() if _CODE_RE.match(ref) else ref)

    return p


# ══════════════════════════════════════════════════════════════════════════════
#  Engine 3 + 4: Curriculum + Prerequisite Graph
# ══════════════════════════════════════════════════════════════════════════════

class CurriculumGraph:
    """
    Loads data/curriculum/courses.json and builds a prerequisite graph.

    Graph model:
      Nodes: course codes ("CSAI 201")
      Edges: A → B  means  "A must be completed before B"

    Uses NetworkX when available; falls back to dict-based checks.
    Works gracefully when courses.json doesn't exist yet.
    """

    def __init__(self, courses_file: Path = COURSES_FILE) -> None:
        self._courses:      dict[str, dict] = {}
        self._name_to_code: dict[str, str]  = {}
        self._G = None  # NetworkX DiGraph or None
        self._load(courses_file)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._courses = raw.get("courses", {})

            # Name → code lookup (case-insensitive)
            for code, info in self._courses.items():
                self._name_to_code[info["name"].lower().strip()] = code

            # NetworkX graph
            try:
                import networkx as nx
                G = nx.DiGraph()
                for code, info in self._courses.items():
                    G.add_node(code, name=info["name"], credits=info["credits"])
                    for pre in info.get("prerequisites", []):
                        if pre in self._courses:
                            G.add_edge(pre, code)
                self._G = G
            except ImportError:
                self._G = None
        except Exception:
            pass

    # ── Public helpers ─────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return bool(self._courses)

    @property
    def course_count(self) -> int:
        return len(self._courses)

    def resolve_code(self, ref: str) -> Optional[str]:
        """Convert a name-or-code string to a canonical code, or None."""
        upper = re.sub(r'\s+', ' ', ref.upper().strip())
        if upper in self._courses:
            return upper
        return self._name_to_code.get(ref.lower().strip())

    def get_info(self, code: str) -> Optional[dict]:
        return self._courses.get(code)

    def get_prereqs(self, code: str) -> list[str]:
        return self._courses.get(code, {}).get("prerequisites", [])

    def prereqs_met(self, code: str, done: set[str]) -> bool:
        return set(self.get_prereqs(code)).issubset(done)

    # ── Engine 5: Eligibility ──────────────────────────────────────────────────

    def analyze_eligibility(
        self,
        completed: list[str],
        failed:    list[str],
        current:   list[str],
    ) -> dict:
        """
        Engine 5: compute eligible / blocked course lists.

        A course is ELIGIBLE if:
          - Not already completed (or currently taking)
          - Not in the failed list (must retake — eligible separately)
          - All prerequisites are in the effective-done set

        Returns dict with keys:
          eligible        : [(code, name, credits)]
          blocked         : [(code, name, credits, [missing_prereqs])]
          retake_eligible : [(code, name, credits)]   ← failed but prereqs met
          completed_codes : set[str]
          failed_codes    : set[str]
        """
        done_codes    = {c for ref in completed if (c := self.resolve_code(ref))}
        failed_codes  = {c for ref in failed    if (c := self.resolve_code(ref))}
        current_codes = {c for ref in current   if (c := self.resolve_code(ref))}

        # Effective completion = completed minus failed
        effective_done = done_codes - failed_codes

        eligible:        list = []
        blocked:         list = []
        retake_eligible: list = []

        for code, info in self._courses.items():
            if code in effective_done or code in current_codes:
                continue

            prereqs  = set(info.get("prerequisites", []))
            missing  = prereqs - effective_done

            entry = (code, info["name"], info["credits"])

            if code in failed_codes:
                # Retake: just check prereqs (should already be met)
                if not missing:
                    retake_eligible.append(entry)
                # If prereqs not met — very unusual, but include in blocked
                else:
                    blocked.append((*entry, sorted(missing)))
            elif not missing:
                eligible.append(entry)
            else:
                blocked.append((*entry, sorted(missing)))

        # Sort: eligible/retake by code; blocked by fewest missing prereqs
        eligible.sort(key=lambda x: x[0])
        retake_eligible.sort(key=lambda x: x[0])
        blocked.sort(key=lambda x: (len(x[3]), x[0]))

        return {
            "eligible":        eligible,
            "blocked":         blocked,
            "retake_eligible": retake_eligible,
            "completed_codes": effective_done,
            "failed_codes":    failed_codes,
        }

    # ── Engine 4: Unlock path via NetworkX ────────────────────────────────────

    def unlock_path(self, target: str) -> list[str]:
        """
        Return the full prerequisite chain for `target` in topological order.
        This shows the student exactly what they need before taking `target`.
        """
        if self._G is not None and target in self._G:
            try:
                import networkx as nx
                ancestors = nx.ancestors(self._G, target)
                sub = self._G.subgraph(ancestors | {target})
                return list(nx.topological_sort(sub))[:-1]  # exclude target itself
            except Exception:
                pass
        return self.get_prereqs(target)

    def courses_unlocked_by(self, done: set[str], new: set[str]) -> list[str]:
        """Courses that become eligible when the student completes `new`."""
        all_done = done | new
        return sorted(
            code for code, info in self._courses.items()
            if code not in all_done
            and set(info.get("prerequisites", [])).issubset(all_done)
            and not set(info.get("prerequisites", [])).issubset(done)
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Intent Router
# ══════════════════════════════════════════════════════════════════════════════

class IntentRouter:
    """
    Classifies student queries without extra API calls.

    Routing:
      planning     → Engine 6 (Planning + Risk + Graduation)
      graduation   → Engine 7 (Graduation focus)
      prerequisite → Engine 4/5 (Prereq check)
      risk         → Engine 9 (Risk only)
      profile      → acknowledge profile info, ask for question
      general      → pass through to standard RAG
    """

    def classify(self, query: str, profile: StudentProfile) -> str:
        q = query.lower()

        # Explicit planning intent
        for kw in _PLANNING_KW:
            if kw in q:
                return "planning"

        # Graduation-focused
        for kw in _GRADUATION_KW:
            if kw in q:
                # If student also gives context → full planning
                return "graduation" if not any(pk in q for pk in _PLANNING_KW) else "planning"

        # Prerequisite check
        for kw in _PREREQ_KW:
            if kw in q:
                return "prerequisite"

        # Risk / standing — only when the question is about the student's OWN
        # situation. A general policy/definition question (e.g. "what is the
        # academic probation policy?") should fall through to standard RAG,
        # which can answer from the official documents instead of generating
        # a personal risk assessment for an unrelated question.
        for kw in _RISK_KW:
            if kw in q:
                if _SELF_RISK_RE.search(q):
                    return "risk"
                break

        # Pure profile share (no question mark, no question word)
        is_info_share = bool(
            _PROGRAM_RE.search(query) or _SEMESTER_RE.search(query) or
            _GPA_RE.search(query) or _COMPLETED_RE.search(query) or
            _FAILED_RE.search(query)
        )
        has_question = "?" in query or re.match(
            r'\b(what|which|how|when|can|do|does|is|are|will|should|would|could|tell)\b',
            q,
        )
        if is_info_share and not has_question:
            return "profile"

        return "general"


# ══════════════════════════════════════════════════════════════════════════════
#  Advisor System Prompt (Engine 8 — Decision Support)
# ══════════════════════════════════════════════════════════════════════════════

_ADVISOR_SYSTEM = """\
You are an expert Academic Advisor AI for Zewail City of Science and Technology (UST).

Your role: generate personalized, structured academic guidance grounded in official
Zewail City curriculum documents and prerequisite analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. NEVER recommend a course whose prerequisites have not been met.
2. Failed courses are NOT completed — the student must retake them.
3. If a failed course is a prerequisite for others, those others are BLOCKED.
4. Typical credit load: 15–18 cr/semester (5–6 courses). Max: 21 cr.
5. GPA < 2.0 → advise limiting to 12 credits and retaking failed courses.
6. GPA 2.0–2.49 → flag academic warning risk; recommend lighter load.
7. Use exact course codes (CSAI 201, not just "Data Structures").
8. Base ALL facts on the provided curriculum context — never invent details.
9. If information is missing, say so clearly and advise visiting Academic Advising.
10. Every recommendation must include WHY it was chosen.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT A — PLANNING / GRADUATION questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use exactly these Markdown sections:

## Student Summary
2–3 sentences summarizing academic standing.

## Eligibility Analysis
**Can Take (prerequisites met):**
• CSAI 201 — Data Structures (3 cr) — reason

**Must Retake:**
• CSAI 261 — Signals and Systems (3 cr) — failed, must repeat

**Blocked (missing prerequisites):**
• CSAI 371 — Digital Signal Processing (3 cr) — needs: CSAI 261

## Recommended Plans

### Safe Plan (~12–15 credits)
| Course | Credits | Why |
|--------|---------|-----|
| CSAI 201 | 3 | Core requirement, prerequisites met |
**Total: X credits | Risk: LOW**
*Advantages:* …
*Disadvantages:* …

### Balanced Plan (~15–18 credits)
(same table format)
**Total: X credits | Risk: MEDIUM**

### Fast Graduation Plan (~18–21 credits)
(same table format)
**Total: X credits | Risk: HIGH**

## Risk Analysis
- **GPA Risk**: LOW/MEDIUM/HIGH — reason
- **Workload Risk**: LOW/MEDIUM/HIGH — reason
- **Graduation Delay Risk**: LOW/MEDIUM/HIGH — reason

## Graduation Impact
- Remaining credits to graduate: ~X
- Estimated completion: Safe plan → Semester X / Fast plan → Semester Y

## Academic Notes
Any warnings, retake policies, office contacts, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT B — PREREQUISITE questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Answer ONLY what was asked. Be direct and concise. No semester plans.

## Prerequisite Chain for [Course Name]
List the direct prerequisites, then the full chain step-by-step with course codes and names.

## What Completing [Course(s)] Unlocks
List courses that become available, grouped by relevance to the student's major.
Use exact course codes. Focus on the most important/relevant ones (up to 10).

## Current Status
One sentence on whether the student can register now or what they need first.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FORMAT C — RISK ASSESSMENT questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Student Summary
1–2 sentences on academic standing.

## Risk Assessment
- **GPA Risk**: LOW/MEDIUM/HIGH — specific reason
- **Workload Risk**: LOW/MEDIUM/HIGH — specific reason
- **Graduation Delay Risk**: LOW/MEDIUM/HIGH — specific reason

## Recommended Actions
Bullet list of specific steps to reduce identified risks.
"""


# ══════════════════════════════════════════════════════════════════════════════
#  AdvisorEngine — orchestrates all engines
# ══════════════════════════════════════════════════════════════════════════════

class AdvisorEngine:
    """
    Main entry point for the Academic Advisor AI.

    Usage (from phase6):
        engine = AdvisorEngine(rag)
        answer, chunks = engine.advise(question, profile, history)
        # Returns (None, []) when routing to standard RAG
    """

    def __init__(self, rag) -> None:
        self._rag    = rag
        self._graph  = CurriculumGraph()
        self._router = IntentRouter()

        status = (
            f"CurriculumGraph: {self._graph.course_count} courses loaded"
            if self._graph.available
            else "CurriculumGraph: no courses.json (run phase8a first)"
        )
        print(f"[AdvisorEngine] {status}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def advise(
        self,
        question: str,
        profile:  StudentProfile,
        history:  list[dict],
    ) -> tuple[Optional[str], list]:
        """
        Route question to the appropriate engine and return (answer, chunks).
        Returns (None, []) to signal caller to use standard RAG.
        """
        intent = self._router.classify(question, profile)

        if intent == "general":
            return None, []  # caller uses standard RAG

        if intent == "profile":
            return self._ack_profile(profile), []

        if intent in ("planning", "graduation", "prerequisite", "risk"):
            return self._full_advisory(question, profile, intent, history)

        return None, []

    # ── Engine 2: Profile acknowledgement ─────────────────────────────────────

    def _ack_profile(self, profile: StudentProfile) -> str:
        items = profile.sidebar_items()
        if not items:
            return (
                "Thanks for sharing! To give you personalised advice, please tell me:\n"
                "- Your school and major (e.g., CSAI — Computer Science)\n"
                "- Your current semester\n"
                "- Your GPA\n"
                "- Courses you have completed\n"
                "- Any courses you failed or need to retake"
            )
        lines = ["Got it! I have noted your academic profile:\n"]
        for label, val in items:
            lines.append(f"- **{label}**: {val}")
        lines.append(
            "\nFeel free to ask me:\n"
            "- *What courses should I take next semester?*\n"
            "- *When will I graduate?*\n"
            "- *What are the prerequisites for [course]?*\n"
            "- *What is my academic risk?*"
        )
        return "\n".join(lines)

    # ── Full advisory response ─────────────────────────────────────────────────

    def _full_advisory(
        self,
        question: str,
        profile:  StudentProfile,
        intent:   str,
        history:  list[dict],
    ) -> tuple[str, list]:

        # 1. Build retrieval query enriched with profile context
        parts = [question]
        school = profile.school or profile.major or ""
        if school:
            parts.append(f"{school} curriculum requirements graduation")
        if profile.semester:
            parts.append(f"semester {profile.semester} courses")
        retrieval_query = " ".join(parts)

        # 2. Retrieve curriculum context via RAG
        chunks, query_note = self._rag.retrieve(retrieval_query, top_k=8)

        # 3. Engine 5: Eligibility analysis from prerequisite graph
        graph_section = ""
        if self._graph.available:
            analysis     = self._graph.analyze_eligibility(
                completed=profile.completed_courses,
                failed=profile.failed_courses,
                current=profile.current_courses,
            )
            graph_section = self._format_graph_section(analysis, profile)

        # For prerequisite questions: also compute hypothetical eligibility using
        # any course codes mentioned in the question (treats them as if completed).
        if intent == "prerequisite" and self._graph.available:
            _CODE_UPPER = re.compile(r'\b([A-Z]{2,6})\s+(\d{3,4}[A-Z]?)\b')
            hypo_codes = [
                f"{m.group(1)} {m.group(2)}"
                for m in _CODE_UPPER.finditer(question.upper())
            ]
            if hypo_codes:
                hypo_completed = list(profile.completed_courses) + [
                    c for c in hypo_codes if c not in profile.failed_courses
                ]
                hypo_analysis = self._graph.analyze_eligibility(
                    completed=hypo_completed,
                    failed=profile.failed_courses,
                    current=profile.current_courses,
                )
                hypo_section = self._format_graph_section(hypo_analysis, profile)
                graph_section = (
                    f"Current eligibility (actual completed courses):\n{graph_section}\n\n"
                    f"Hypothetical eligibility assuming {', '.join(hypo_codes)} are completed:\n{hypo_section}"
                )

        # 4. Engine 7: Graduation estimate
        grad_section = self._graduation_estimate(profile)

        # 5. Engine 9: Risk flags
        risk_flags = self._risk_flags(profile)

        # 6. Assemble full prompt — instruction varies by intent
        rag_ctx = self._format_rag_context(chunks, query_note)
        intent_label = {
            "planning":     "semester course planning",
            "graduation":   "graduation timeline analysis",
            "prerequisite": "prerequisite check",
            "risk":         "academic risk assessment",
        }.get(intent, "academic advising")

        if intent == "prerequisite":
            task_instruction = (
                "Answer ONLY the prerequisite question the student asked. "
                "Use FORMAT B (prerequisite format) from your instructions. "
                "Do NOT generate semester plans or the full planning template. "
                "Be specific with course codes. Keep response concise."
            )
        elif intent == "risk":
            task_instruction = (
                "Assess academic risk ONLY. "
                "Use FORMAT C (risk assessment format) from your instructions. "
                "Do NOT generate semester plans."
            )
        elif intent == "graduation":
            task_instruction = (
                "Focus on graduation timeline. Use FORMAT A but keep Recommended Plans "
                "to ONE concise table (the most realistic plan). "
                "Emphasise the graduation estimate, remaining credits, and what the student "
                "must complete to finish on time."
            )
        else:  # planning
            task_instruction = (
                "Generate a comprehensive academic advisory response "
                "using FORMAT A (planning format) from your instructions exactly. "
                "Ground every recommendation in the prerequisite analysis and curriculum documents. "
                "Be specific: use course codes, credit counts, and clear reasoning."
            )

        user_message = (
            f"## Student Profile\n{profile.summary_for_prompt()}\n\n"
            f"## Prerequisite Graph Analysis\n{graph_section}\n\n"
            f"## Graduation Estimate\n{grad_section}\n\n"
            f"## Academic Risk Flags\n{risk_flags}\n\n"
            f"## Official Curriculum Documents\n{rag_ctx}\n\n"
            f"## Task: {intent_label}\n"
            f"Student's question: {question}\n\n"
            f"{task_instruction}"
        )

        # 7. Call GPT with advisor system prompt
        messages: list[dict] = [{"role": "system", "content": _ADVISOR_SYSTEM}]
        if history:
            messages.extend(history[-6:])  # last 3 turns for context
        messages.append({"role": "user", "content": user_message})

        resp = self._rag._oai.chat.completions.create(
            model=self._rag._chat_model,
            messages=messages,
            temperature=0.15,
            max_tokens=2200,
        )
        answer = resp.choices[0].message.content.strip()
        return answer, chunks

    # ── Engine 5: Format eligibility section ──────────────────────────────────

    def _format_graph_section(self, analysis: dict, profile: StudentProfile) -> str:
        lines: list[str] = []

        eligible  = analysis["eligible"]
        blocked   = analysis["blocked"]
        retake    = analysis["retake_eligible"]
        done_n    = len(analysis["completed_codes"])
        failed_n  = len(analysis["failed_codes"])

        lines.append(f"Courses confirmed completed: {done_n}")
        lines.append(f"Courses confirmed failed (must retake): {failed_n}")
        lines.append("")

        if retake:
            lines.append(f"Retake-eligible ({len(retake)} — prerequisites met):")
            for code, name, cr in retake[:10]:
                lines.append(f"  ↩ {code} — {name} ({cr} cr)")
            lines.append("")

        if eligible:
            lines.append(f"Eligible to take ({len(eligible)} courses, prerequisites met):")
            for code, name, cr in eligible[:25]:
                lines.append(f"  ✓ {code} — {name} ({cr} cr)")
            if len(eligible) > 25:
                lines.append(f"  … and {len(eligible)-25} more")
            lines.append("")

        if blocked:
            lines.append(f"Blocked courses (top 15, fewest missing prerequisites first):")
            for code, name, cr, missing in blocked[:15]:
                lines.append(f"  ✗ {code} — {name} ({cr} cr) | needs: {', '.join(missing)}")

        if not eligible and not retake:
            lines.append(
                "NOTE: Could not resolve any completed courses to known course codes. "
                "The student may have stated them by name rather than code. "
                "Use the curriculum documents to infer eligible courses from the stated course names."
            )

        return "\n".join(lines)

    # ── Engine 7: Graduation estimate ─────────────────────────────────────────

    def _graduation_estimate(self, profile: StudentProfile) -> str:
        total_needed = profile.total_credits_needed()
        remaining    = profile.estimated_remaining_credits()
        done         = total_needed - remaining

        sems_at_safe     = max(1, -(-remaining // _LOAD_SAFE[1]))      # ceil division
        sems_at_balanced = max(1, -(-remaining // _LOAD_BALANCED[1]))
        sems_at_fast     = max(1, -(-remaining // _LOAD_FAST[1]))

        lines = [
            f"Total credits needed for graduation: {total_needed}",
            f"Estimated credits completed: ~{done}",
            f"Estimated remaining: ~{remaining}",
            f"Semesters to graduation at Safe load (~{_LOAD_SAFE[1]} cr): {sems_at_safe}",
            f"Semesters to graduation at Balanced load (~{_LOAD_BALANCED[1]} cr): {sems_at_balanced}",
            f"Semesters to graduation at Fast load (~{_LOAD_FAST[1]} cr): {sems_at_fast}",
        ]
        if profile.failed_courses:
            lines.append(
                f"Note: {len(profile.failed_courses)} failed course(s) must be retaken "
                "— add ~1 semester if they are not completed in the next cycle."
            )
        return "\n".join(lines)

    # ── Engine 9: Risk flags ───────────────────────────────────────────────────

    def _risk_flags(self, profile: StudentProfile) -> str:
        flags: list[str] = []

        if profile.gpa and profile.gpa < _GPA_PROBATION:
            flags.append(
                f"🔴 HIGH GPA RISK: GPA {profile.gpa} is below the probation threshold "
                f"({_GPA_PROBATION}). Student should take at most 12 credits and prioritize "
                "high-impact courses."
            )
        elif profile.gpa and profile.gpa < _GPA_WARNING:
            flags.append(
                f"🟡 MEDIUM GPA RISK: GPA {profile.gpa} is in the warning zone. "
                "Recommend balanced load (15 credits) and focus on courses likely to improve GPA."
            )

        if profile.failed_courses:
            count = len(profile.failed_courses)
            flags.append(
                f"🟡 RETAKE REQUIRED: {count} failed course(s) — "
                f"{', '.join(profile.failed_courses)}. "
                "These must be completed; any courses that depend on them remain blocked."
            )

        if profile.semester and profile.semester >= 8 and profile.estimated_remaining_credits() > 20:
            flags.append(
                "🔴 GRADUATION DELAY RISK: Student is in semester 8+ with significant "
                "credits remaining. Immediate advising appointment recommended."
            )

        return "\n".join(flags) if flags else "No major risk flags detected."

    # ── RAG context formatter ──────────────────────────────────────────────────

    def _format_rag_context(self, chunks: list, query_note: str) -> str:
        parts: list[str] = []
        if query_note:
            parts.append(query_note)
        for i, c in enumerate(chunks, 1):
            src = c.source + (f" (p.{c.page})" if c.page else "")
            parts.append(f"[Doc {i} | {c.category} | {src}]\n{c.text}")
        return "\n\n---\n\n".join(parts) if parts else "No curriculum documents retrieved."
