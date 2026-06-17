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

PROJECT_ROOT      = Path(__file__).parent
COURSES_FILE      = PROJECT_ROOT / "data" / "curriculum" / "courses.json"
STUDY_PLANS_FILE  = PROJECT_ROOT / "data" / "curriculum" / "study_plans.json"

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

# Courses in semester N + _PREMATURE_THRESHOLD or later are suppressed from
# recommended plans even when prerequisites are technically satisfied.
_PREMATURE_THRESHOLD = 2


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
    r'semester[\s:]*(\d+)|\bsem[\s:]*(\d+)\b|year[\s:]*(\d+)|(\d+)(?:st|nd|rd|th)\s+(?:semester|year)'
    r'|\b(first|second|third|fourth|fifth|sixth|seventh|eighth)\s+(?:semester|year|sem)\b',
    re.I,
)
_ORDINAL_TO_INT = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
}
_GPA_RE = re.compile(
    r'\b([0-3]\.\d{1,2}|4\.0)\s*(?:gpa|cgpa)'
    r'|\bgpa\s*(?:is|of|:)?\s*([0-3]\.\d{1,2}|4\.0)\b',
    re.I,
)
_COMPLETED_RE = re.compile(
    r'(?:completed?|passed?|finished?|took|done with|have taken|'
    r'already took|i passed|i completed|i finished|i have done|'
    r'(?:courses?|subjects?)\s+(?:taken|i took|done)|'
    r'the following courses?|these courses?|my courses?\s+(?:are|were|include))'
    r'[:\s]+'
    r'(.+?)(?=\.|$|,\s*\band\b|\band\b\s+(?:i\s+)?(?:failed|currently|now))',
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

# Words that, when immediately preceding a multi-word programme keyword, indicate
# it's part of a course title rather than a student self-identification phrase.
# e.g. "Programming AND Computer Science" → course name; "studying Computer Science" → major.
_PROG_TITLE_CONJUNCTIONS: frozenset = frozenset({"AND", "OR", "OF"})

# Concentration / track extraction
_CONC_RE = re.compile(
    r'(?:'
    r'(?:concentration|track|speciali[sz]ation)\s*[:\-]?\s*(?:in\s+|on\s+)?'
    r'|my\s+(?:track|concentration)\s+is\s+'
    r'|i(?:\'m|\s+am)\s+in(?:\s+the)?\s+'
    r'|i(?:\'m|\s+am)\s+(?:doing|studying)\s+'
    r')'
    r'(CBG|MCB|DDD|NPHY|NCHEM|BIONANO|NMED|AST|HEP|APD|GCG|HCI|ITNS|ITCC'
    r'|cell\s+bio(?:logy)?(?:\s+and\s+genetics)?'
    r'|molecular\s+cell(?:\s+biology)?'
    r'|drug\s+design(?:\s+and\s+discovery)?'
    r'|nanophysics|nanochemistry'
    r'|bio.nanotechnology'
    r'|nanomedicine'
    r'|astrophysics|high.energy(?:\s+physics)?'
    r'|advanced\s+programming'
    r'|computer\s+graphics'
    r'|human.computer\s+interaction'
    r'|network\s+security|cloud\s+computing)',
    re.I,
)

_CONC_CANONICAL: dict[str, str] = {
    "CBG": "CBG",   "CELL BIO": "CBG",    "CELL BIOLOGY": "CBG",
    "CELL BIOLOGY AND GENETICS": "CBG",
    "MCB": "MCB",   "MOLECULAR CELL": "MCB", "MOLECULAR CELL BIOLOGY": "MCB",
    "DDD": "DDD",   "DRUG DESIGN": "DDD",  "DRUG DESIGN AND DISCOVERY": "DDD",
    "NPHY": "NPHY", "NANOPHYSICS": "NPHY",
    "NCHEM": "NCHEM", "NANOCHEMISTRY": "NCHEM",
    "BIONANO": "BIONANO", "BIO-NANOTECHNOLOGY": "BIONANO", "BIO NANOTECHNOLOGY": "BIONANO",
    "NMED": "NMED", "NANOMEDICINE": "NMED",
    "AST": "AST",   "ASTROPHYSICS": "AST",
    "HEP": "HEP",   "HIGH-ENERGY": "HEP",  "HIGH ENERGY": "HEP",
    "HIGH-ENERGY PHYSICS": "HEP", "HIGH ENERGY PHYSICS": "HEP",
    "APD": "APD",   "ADVANCED PROGRAMMING": "APD",
    "GCG": "GCG",   "COMPUTER GRAPHICS": "GCG",
    "HCI": "HCI",   "HUMAN-COMPUTER INTERACTION": "HCI", "HUMAN COMPUTER INTERACTION": "HCI",
    "ITNS": "ITNS", "NETWORK SECURITY": "ITNS",
    "ITCC": "ITCC", "CLOUD COMPUTING": "ITCC",
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

# Personal self-identification signals (distinct from a generic question about a programme).
# Used by IntentRouter to decide when the student is sharing their OWN situation vs.
# asking a factual question that happens to mention a school name.
_PERSONAL_RE = re.compile(
    r"\b(i\s+am\b|i'm|i\'m\s+in|my\b|i\s+have|i\s+failed|i\s+took|"
    r"i\s+passed|i\s+need\s+to\s+retake|my\s+gpa|my\s+major|"
    r"my\s+semester|my\s+programme|my\s+program|for\s+me|"
    r"i\s+am\s+in|i'm\s+in|i\s+study|i\s+am\s+doing)\b",
    re.I,
)

# Schools that use a generic code as major when no specific programme is stated.
# "I am CSAI" is ambiguous — could be DSAI, SWD, or IT.
# "I am DSAI" is specific — resolves unambiguously.
_GENERIC_SCHOOL_CODES: frozenset = frozenset({"CSAI", "SCI", "BUS", "ENGR"})

# Human-readable track options shown when a student must pick a specific programme.
_TRACK_OPTIONS: dict[str, str] = {
    "CSAI": "DSAI (Data Science & AI), Software Development (SWD), or IT",
    "SCI":  "Biomedical Sciences (BMS), Nanoscience & Nanotechnology (NANO), "
            "or Physics of the Universe (PHY)",
    "BUS":  "Finance & Investment (FIM), Actuarial Analysis & Risk Mgmt (AARM), "
            "Marketing & Entrepreneurship (MEIM), or Operations & Supply Chain (OSCTM)",
    "ENGR": "Aerospace Engineering, Communications & Information Eng. (CIE), "
            "Nanotechnology & Nanoelectronics, Renewable Energy Engineering, "
            "or Environmental Engineering",
}


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
    concentration:     str   = ""   # e.g. "CBG", "MCB", "NPHY", "APD"

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
        if self.concentration:
            parts.append(f"Concentration/Track: {self.concentration}")
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
        if self.concentration:
            items.append(("Track", self.concentration))
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

    # Collapse newlines → spaces so multiline course listings (copy-pasted from
    # PDFs or typed across multiple lines) are captured as a single line.
    text_norm  = re.sub(r'[ \t]*\n[ \t]*', ' ', text)
    text_upper = text_norm.upper()

    # ── Major / school ─────────────────────────────────────────────────────────
    # Walk _MAJOR_MAP in insertion order (most-specific keys first) and take
    # the first key whose word appears in the text.  This means "DSAI" wins
    # over "CSAI" when a student says "I am in CSAI, DSAI major" because
    # "DSAI" is listed earlier in the map.
    # Multi-word keys (e.g. "COMPUTER SCIENCE") are skipped when they are
    # immediately preceded by a conjunction (AND/OR/OF), which signals the
    # match is inside a course title such as "Programming and Computer Science"
    # rather than a self-identification phrase.
    best_key: Optional[str] = None
    for k in _MAJOR_MAP:
        for m in re.finditer(r'\b' + re.escape(k) + r'\b', text_upper):
            if ' ' in k:
                pre_words = text_upper[max(0, m.start() - 15):m.start()].split()
                if pre_words and pre_words[-1] in _PROG_TITLE_CONJUNCTIONS:
                    continue  # this match is inside a course name
            best_key = k
            break
        if best_key:
            break
    if best_key:
        major, school = _MAJOR_MAP[best_key]
        # Override unconditionally when the message also contains personal
        # academic data (GPA or semester), which signals self-identification
        # ("CSAI, sem 4, GPA 3.1"). Without personal data, only update if
        # empty — so "BUS graduation requirements?" doesn't change a CSAI
        # student's profile to BUS.
        has_personal_context = bool(
            _SEMESTER_RE.search(text_norm) or _GPA_RE.search(text_norm)
            or _CREDITS_RE.search(text_norm)
        )
        # Allow override when moving from a generic school code to a specific programme,
        # e.g. "CSAI" → "DSAI" in response to a track clarification question.
        is_more_specific = (
            major != school                        # new major is not the school name
            and p.major.upper() == p.school.upper() # current major is still the generic code
            and bool(p.major)                      # profile already has something
        )
        if not p.major or has_personal_context or is_more_specific:
            p.major = major
        if not p.school or has_personal_context:
            p.school = school

    # ── Concentration / track ─────────────────────────────────────────────────
    m = _CONC_RE.search(text_norm)
    if m:
        raw = re.sub(r'\s+', ' ', m.group(1).upper().strip())
        conc = _CONC_CANONICAL.get(raw)
        if not conc:
            for key, val in _CONC_CANONICAL.items():
                if raw.startswith(key):
                    conc = val
                    break
        if conc:
            p.concentration = conc

    # ── Semester ──────────────────────────────────────────────────────────────
    m = _SEMESTER_RE.search(text_norm)
    if m:
        raw     = m.group(1) or m.group(2) or m.group(3) or m.group(4)
        ordinal = m.group(5)  # "first", "second", … matched by new group
        if raw:
            p.semester = int(raw)   # allow override (student may correct themselves)
        elif ordinal:
            p.semester = _ORDINAL_TO_INT.get(ordinal.lower(), 0)

    # ── GPA ───────────────────────────────────────────────────────────────────
    m = _GPA_RE.search(text_norm)
    if m:
        raw = m.group(1) or m.group(2)
        if raw:
            p.gpa = float(raw)

    # ── Completed credits ─────────────────────────────────────────────────────
    m = _CREDITS_RE.search(text_norm)
    if m:
        p.completed_credits = int(m.group(1))

    # ── Completed courses ─────────────────────────────────────────────────────
    for m in _COMPLETED_RE.finditer(text_norm):
        for ref in _extract_course_refs(m.group(1)):
            if ref and ref.upper() not in [c.upper() for c in p.completed_courses]:
                p.completed_courses.append(ref.upper() if _CODE_RE.match(ref) else ref)

    # ── Failed courses ────────────────────────────────────────────────────────
    for m in _FAILED_RE.finditer(text_norm):
        for ref in _extract_course_refs(m.group(1)):
            if ref and ref.upper() not in [c.upper() for c in p.failed_courses]:
                p.failed_courses.append(ref.upper() if _CODE_RE.match(ref) else ref)

    # ── Current courses ───────────────────────────────────────────────────────
    for m in _CURRENT_RE.finditer(text_norm):
        for ref in _extract_course_refs(m.group(1)):
            if ref and ref.upper() not in [c.upper() for c in p.current_courses]:
                p.current_courses.append(ref.upper() if _CODE_RE.match(ref) else ref)

    return p


# ══════════════════════════════════════════════════════════════════════════════
#  Curriculum progression data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CurriculumProgressionMetrics:
    """How far the student is along the official study plan."""
    status:           str   # "on_track" | "ahead" | "behind" | "unknown"
    expected_done:    list  # codes the plan expects completed before cur_sem
    missing_core:     list  # expected_done − actual_completed (delayed)
    extra_done:       list  # done but not in plan (electives / ahead of plan)
    delayed_count:    int
    total_expected:   int
    total_completed:  int
    expected_credits: int   # total credits the plan expects done by now
    actual_credits:   int   # total credits from resolved completed codes

    def status_line(self) -> str:
        if self.status == "ahead":
            return (
                f"Ahead of plan — {len(self.extra_done)} extra course(s) "
                "completed beyond what the official plan expected by this semester"
            )
        if self.status == "behind":
            sample = ", ".join(self.missing_core[:5])
            trail  = " …" if len(self.missing_core) > 5 else ""
            return (
                f"Behind plan — {self.delayed_count} expected core course(s) "
                f"not yet completed: {sample}{trail}"
            )
        if self.status == "on_track":
            return "On track with the official study plan"
        return "Curriculum progression status unknown (plan data unavailable)"


@dataclass
class CourseRecommendation:
    """A single recommended course with pre-computed rationale."""
    code:     str
    name:     str
    credits:  int
    reason:   str
    plan_sem: str   # e.g. "Y3S1" or "" when not in plan
    priority: str   # "retake" | "current_sem" | "next_sem" | "catch_up" | "elective"


@dataclass
class SemesterPlan:
    """One load-tier recommendation (Safe / Balanced / Fast)."""
    label:         str   # "Safe" | "Balanced" | "Fast"
    min_cr:        int
    max_cr:        int
    risk_level:    str   # "LOW" | "MEDIUM" | "HIGH"
    courses:       list  # list[CourseRecommendation]
    total_credits: int
    notes:         list  # list[str]

    def to_prompt_text(self) -> str:
        lines = [
            f"### {self.label} Plan ({self.min_cr}–{self.max_cr} cr) | Risk: {self.risk_level}",
        ]
        for r in self.courses:
            tag       = f"[{r.priority}]"
            plan_note = f" (plan: {r.plan_sem})" if r.plan_sem else ""
            lines.append(
                f"  {tag} {r.code} — {r.name} ({r.credits} cr){plan_note} — {r.reason}"
            )
        lines.append(f"  Total: {self.total_credits} credits")
        for note in self.notes:
            lines.append(f"  Note: {note}")
        return "\n".join(lines)


@dataclass
class ComputedPlan:
    """All three load-tier plans + progression metrics, computed before the GPT call."""
    safe:        SemesterPlan
    balanced:    SemesterPlan
    fast:        SemesterPlan
    progression: CurriculumProgressionMetrics

    def to_prompt_text(self) -> str:
        p = self.progression
        lines = [
            "COMPUTED SEMESTER PLANS "
            "(generated by Planning Engine — DO NOT override course selection):",
            "",
            self.safe.to_prompt_text(),
            "",
            self.balanced.to_prompt_text(),
            "",
            self.fast.to_prompt_text(),
            "",
            "CURRICULUM PROGRESSION:",
            f"  Status: {p.status_line()}",
            f"  Expected completed before this semester: "
            f"{p.total_expected} course(s) ({p.expected_credits} cr)",
            f"  Confirmed completed (resolved to course codes): "
            f"{p.total_completed} course(s)",
        ]
        if p.missing_core:
            sample = ", ".join(p.missing_core[:8])
            trail  = f" … (+{len(p.missing_core)-8} more)" if len(p.missing_core) > 8 else ""
            lines.append(
                f"  Delayed core courses ({p.delayed_count}): {sample}{trail}"
            )
        return "\n".join(lines)


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
        completed:    list[str],
        failed:       list[str],
        current:      list[str],
        plan_context: Optional[dict] = None,
    ) -> dict:
        """
        Engine 5: compute eligible / blocked course lists.

        A course is ELIGIBLE if:
          - Not already completed (or currently taking)
          - Not in the failed list (must retake — eligible separately)
          - All prerequisites are in the effective-done set

        Optional plan_context dict (all values are set[str]):
          current_sem_codes    — codes the official plan assigns to current semester
          next_sem_codes       — codes the official plan assigns to next semester
          premature_codes      — codes 2+ semesters ahead (suppressed from plans)
          planned_before_codes — codes the plan expects completed before now

        Returns dict with keys:
          eligible              : [(code, name, credits)]
          blocked               : [(code, name, credits, [missing_prereqs])]
          retake_eligible       : [(code, name, credits)]
          completed_codes       : set[str]
          failed_codes          : set[str]
          current_sem_eligible  : [(code, name, credits)]  plan-current, eligible
          next_sem_eligible     : [(code, name, credits)]  plan-next, eligible
          behind_plan_eligible  : [(code, name, credits)]  catch-up eligible
          premature_eligible    : [(code, name, credits)]  suppressed (too far ahead)
          plan_context_available: bool
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

        # Plan-context categorisation (only when plan data is available)
        if plan_context:
            cur_set = plan_context.get("current_sem_codes",    set())
            nxt_set = plan_context.get("next_sem_codes",       set())
            pre_set = plan_context.get("premature_codes",      set())
            bef_set = plan_context.get("planned_before_codes", set())

            current_sem_eligible = [e for e in eligible if e[0] in cur_set]
            next_sem_eligible    = [e for e in eligible if e[0] in nxt_set
                                    and e[0] not in cur_set]
            premature_eligible   = [e for e in eligible if e[0] in pre_set]
            behind_plan_eligible = [
                e for e in eligible
                if e[0] in bef_set
                and e[0] not in cur_set
                and e[0] not in nxt_set
            ]
        else:
            current_sem_eligible = []
            next_sem_eligible    = []
            premature_eligible   = []
            behind_plan_eligible = []

        return {
            "eligible":               eligible,
            "blocked":                blocked,
            "retake_eligible":        retake_eligible,
            "completed_codes":        effective_done,
            "failed_codes":           failed_codes,
            # plan-aware categories
            "current_sem_eligible":   current_sem_eligible,
            "next_sem_eligible":      next_sem_eligible,
            "premature_eligible":     premature_eligible,
            "behind_plan_eligible":   behind_plan_eligible,
            "plan_context_available": bool(plan_context),
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
#  StudyPlanEngine — Authoritative semester-by-semester plan from study_plans.json
# ══════════════════════════════════════════════════════════════════════════════

class StudyPlanEngine:
    """
    Loads data/curriculum/study_plans.json and provides structured semester
    planning context as the authoritative source for course-to-semester mapping.
    Replaces RAG-inferred semester placement with exact official study plan data.
    """

    # Ordered list: (school_upper, major_keyword) → (school_key, prog_key)
    # Earlier entries win; empty major_keyword matches any (used as fallback).
    _PROG_MAP: list[tuple[tuple[str, str], tuple[str, str]]] = [
        # CSAI programmes
        (("CSAI", "DSAI"),               ("CSAI", "DSAI")),
        (("CSAI", "DATA SCIENCE"),        ("CSAI", "DSAI")),
        (("CSAI", "IT"),                  ("CSAI", "IT")),
        (("CSAI", "INFORMATION TECH"),    ("CSAI", "IT")),
        (("CSAI", "SWD"),                 ("CSAI", "SWD")),
        (("CSAI", "SOFTWARE"),            ("CSAI", "SWD")),
        (("CSAI", "COMPUTER SCIENCE"),    ("CSAI", "SWD")),
        (("CSAI", "CS"),                  ("CSAI", "SWD")),
        (("CSAI", ""),                    ("CSAI", "SWD")),   # default CSAI
        # BUS programmes
        (("BUS",  "AARM"),                ("BUS",  "AARM")),
        (("BUS",  "ACTUARIAL"),           ("BUS",  "AARM")),
        (("BUS",  "FIM"),                 ("BUS",  "FIM")),
        (("BUS",  "FINANCE"),             ("BUS",  "FIM")),
        (("BUS",  "MEIM"),                ("BUS",  "MEIM")),
        (("BUS",  "MARKETING"),           ("BUS",  "MEIM")),
        (("BUS",  "ENTREPRENEURSHIP"),    ("BUS",  "MEIM")),
        (("BUS",  "INNOVATION"),          ("BUS",  "MEIM")),
        (("BUS",  "OSCTM"),               ("BUS",  "OSCTM")),
        (("BUS",  "OPERATIONS"),          ("BUS",  "OSCTM")),
        (("BUS",  "SUPPLY CHAIN"),        ("BUS",  "OSCTM")),
        (("BUS",  "BUSINESS ANALYTICS"),  ("BUS",  "MEIM")),
        (("BUS",  ""),                    ("BUS",  "AARM")),  # default BUS
        # SCI programmes
        (("SCI",  "BMS"),                 ("SCI",  "BMS")),
        (("SCI",  "BIOMEDICAL"),          ("SCI",  "BMS")),
        (("SCI",  "NANO"),                ("SCI",  "NANO")),
        (("SCI",  "NANOSCIENCE"),         ("SCI",  "NANO")),
        (("SCI",  "PHY"),                 ("SCI",  "PHY")),
        (("SCI",  "PHYSICS"),             ("SCI",  "PHY")),
        (("SCI",  ""),                    ("SCI",  "BMS")),   # default SCI
    ]

    def __init__(self) -> None:
        self._plans: dict = {}
        self._loaded = False
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(STUDY_PLANS_FILE.read_text(encoding="utf-8"))
            self._plans = data.get("plans", {})
            self._loaded = bool(self._plans)
        except Exception:
            self._loaded = False

    @property
    def available(self) -> bool:
        return self._loaded

    def resolve_prog(self, school: str, major: str) -> Optional[tuple[str, str]]:
        """Map profile.school + profile.major → (school_key, prog_key) or None."""
        s = school.upper().strip()
        m = major.upper().strip()
        for (sk, mk), result in self._PROG_MAP:
            if sk != s:
                continue
            if mk == "" or mk in m:
                return result
        # Fallback: try major alone (ignore school mismatch)
        for (sk, mk), result in self._PROG_MAP:
            if mk and mk in m:
                return result
        return None

    def get_track_options(self, school_key: str, prog_key: str) -> list[str]:
        """Non-_common track codes for a programme."""
        tracks = self._plans.get(school_key, {}).get(prog_key, {}).get("tracks", {})
        return [t for t in tracks if t != "_common"]

    def _get_all_semesters(
        self, school_key: str, prog_key: str, track: Optional[str]
    ) -> list[tuple[str, int, list[dict]]]:
        """
        Sorted list of (sem_key, seq, courses) combining _common + optional track.
        seq=0 is summer; sorted so summers appear after regular semesters of same year.
        """
        prog_data = self._plans.get(school_key, {}).get(prog_key, {})
        tracks    = prog_data.get("tracks", {})
        all_sems: dict[str, dict] = {}

        for src_key in ("_common", track or ""):
            for sem_key, sem_data in tracks.get(src_key, {}).get("semesters", {}).items():
                if sem_key not in all_sems:
                    all_sems[sem_key] = {"seq": sem_data["seq"], "courses": []}
                existing = {c["code"] for c in all_sems[sem_key]["courses"]}
                for c in sem_data.get("courses", []):
                    if c["code"] not in existing:
                        all_sems[sem_key]["courses"].append(c)
                        existing.add(c["code"])

        def _sort(item: tuple[str, dict]) -> tuple[int, int]:
            seq = item[1]["seq"]
            return (99 if seq == 0 else seq, 0)

        return [
            (sk, sd["seq"], sd["courses"])
            for sk, sd in sorted(all_sems.items(), key=_sort)
        ]

    def get_planned_codes_before(
        self, school_key: str, prog_key: str,
        track: Optional[str], current_seq: int,
    ) -> list[str]:
        """All course codes the official plan places before current_seq (excluding summer)."""
        return [
            c["code"]
            for _, seq, courses in self._get_all_semesters(school_key, prog_key, track)
            if seq != 0 and seq < current_seq
            for c in courses
        ]

    def get_current_sem_codes(
        self, school_key: str, prog_key: str,
        track: Optional[str], current_seq: int,
    ) -> list[str]:
        """Course codes the official plan assigns to current_seq."""
        return [
            c["code"]
            for _, seq, courses in self._get_all_semesters(school_key, prog_key, track)
            if seq == current_seq
            for c in courses
        ]

    def get_next_sem_codes(
        self, school_key: str, prog_key: str,
        track: Optional[str], current_seq: int,
    ) -> list[str]:
        """Course codes the official plan assigns to current_seq + 1."""
        return [
            c["code"]
            for _, seq, courses in self._get_all_semesters(school_key, prog_key, track)
            if seq == current_seq + 1
            for c in courses
        ]

    def get_premature_codes(
        self, school_key: str, prog_key: str,
        track: Optional[str], current_seq: int,
        threshold: int = _PREMATURE_THRESHOLD,
    ) -> list[str]:
        """Codes the plan places in semester current_seq + threshold or later."""
        return [
            c["code"]
            for _, seq, courses in self._get_all_semesters(school_key, prog_key, track)
            if seq != 0 and seq >= current_seq + threshold
            for c in courses
        ]

    def infer_presumed_completed(
        self,
        school: str,
        major: str,
        current_semester: int,
        track: Optional[str] = None,
    ) -> list[str]:
        """
        Return all course codes the official plan expects completed BEFORE
        `current_semester` (i.e. semesters 1 .. current_semester-1).

        Used to implement the Transcript Inference Rule: when a student says
        "I'm in semester N", the advisor assumes the standard plan was followed
        unless the student explicitly states failed or missing courses.
        """
        resolved = self.resolve_prog(school, major)
        if not resolved:
            return []
        s_key, p_key = resolved
        return self.get_planned_codes_before(s_key, p_key, track, current_semester)

    def get_progression_metrics(
        self,
        school_key:      str,
        prog_key:        str,
        track:           Optional[str],
        current_seq:     int,
        completed_codes: set,
        courses_dict:    dict,
    ) -> CurriculumProgressionMetrics:
        """
        Compare completed courses against what the official plan expects by now.
        Returns CurriculumProgressionMetrics with on_track / ahead / behind status.
        """
        expected     = set(self.get_planned_codes_before(school_key, prog_key, track, current_seq))
        missing_core = sorted(expected - completed_codes)
        extra_done   = sorted(completed_codes - expected)

        expected_credits = sum(
            courses_dict.get(c, {}).get("credits", 0) for c in expected
        )
        actual_credits = sum(
            courses_dict.get(c, {}).get("credits", 0) for c in completed_codes
        )

        if not missing_core:
            status = "ahead" if extra_done else "on_track"
        else:
            status = "behind"

        return CurriculumProgressionMetrics(
            status=status,
            expected_done=sorted(expected),
            missing_core=missing_core,
            extra_done=extra_done,
            delayed_count=len(missing_core),
            total_expected=len(expected),
            total_completed=len(completed_codes),
            expected_credits=expected_credits,
            actual_credits=actual_credits,
        )

    def build_study_plan_context(self, profile: "StudentProfile") -> str:
        """
        Structured study plan context for the LLM prompt.
        Shows which courses belong to which semester per the official plan,
        replacing vague RAG-based semester inference.
        """
        if not self.available:
            return "Structured study plan data not available (study_plans.json missing)."

        resolved = self.resolve_prog(profile.school, profile.major)
        if not resolved:
            return (
                f"Could not resolve study plan for school={profile.school!r}, "
                f"major={profile.major!r}."
            )

        school_key, prog_key = resolved
        prog_data    = self._plans.get(school_key, {}).get(prog_key, {})
        prog_name    = prog_data.get("full_name", prog_key)
        avail_tracks = self.get_track_options(school_key, prog_key)

        track = profile.concentration.upper().strip() if profile.concentration else None
        prog_tracks  = prog_data.get("tracks", {})
        if track and track not in prog_tracks:
            track = None

        header = f"Official Study Plan — {prog_name} ({school_key}/{prog_key})"
        if track:
            track_name = prog_tracks.get(track, {}).get("full_name", track)
            header += f" / {track_name} ({track})"
        lines = [header]

        if avail_tracks and not track:
            lines.append(
                f"  Available concentration tracks: {', '.join(avail_tracks)}"
                " (student has not specified a track — showing shared courses only)"
            )

        semesters = self._get_all_semesters(school_key, prog_key, track)
        cur_seq   = profile.semester or 0

        if cur_seq > 0:
            # What the plan says should be done before this semester
            prev_codes = [
                c["code"]
                for _, seq, courses in semesters
                if seq != 0 and seq < cur_seq
                for c in courses
            ]
            if prev_codes:
                lines.append(
                    f"\n  Expected completed before semester {cur_seq} "
                    f"(per official plan, semesters 1–{cur_seq - 1}):"
                )
                lines.append(f"    {', '.join(prev_codes)}")

            # Current semester courses
            found_current = False
            for sem_key, seq, courses in semesters:
                if seq == cur_seq:
                    found_current = True
                    course_list = ", ".join(
                        f"{c['code']} ({c['credits']}cr)" for c in courses
                    )
                    total_cr = sum(c["credits"] for c in courses)
                    lines.append(
                        f"\n  Semester {cur_seq} ({sem_key}) — official planned courses:"
                    )
                    lines.append(f"    {course_list}")
                    lines.append(f"    Planned total: {total_cr} credits")

            if not found_current and avail_tracks and not track:
                lines.append(
                    f"\n  Semester {cur_seq} — courses depend on chosen concentration. "
                    f"Tracks: {', '.join(avail_tracks)}. "
                    "Ask the student for their track to show specific semester courses."
                )

            # Next semester courses
            next_seq = cur_seq + 1
            found_next = False
            for sem_key, seq, courses in semesters:
                if seq == next_seq:
                    found_next = True
                    course_list = ", ".join(
                        f"{c['code']} ({c['credits']}cr)" for c in courses
                    )
                    lines.append(
                        f"\n  Semester {next_seq} ({sem_key}) — next semester official plan:"
                    )
                    lines.append(f"    {course_list}")

            if not found_next and avail_tracks and not track:
                lines.append(
                    f"\n  Semester {next_seq} — also track-dependent. "
                    f"Available tracks: {', '.join(avail_tracks)}."
                )
        else:
            lines.append("\n  Full programme plan:")
            for sem_key, seq, courses in semesters:
                sem_label = "Summer" if seq == 0 else f"Semester {seq}"
                codes = ", ".join(f"{c['code']} ({c['credits']}cr)" for c in courses)
                lines.append(f"    {sem_label} ({sem_key}): {codes}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Engine 6: Academic Planning Engine — deterministic plan computation
# ══════════════════════════════════════════════════════════════════════════════

class PlanningEngine:
    """
    Engine 6: builds Safe / Balanced / Fast semester plans deterministically.

    All course-selection decisions are made from structured data:
      - CurriculumGraph eligibility (prereq graph + plan categorisation)
      - StudentProfile (GPA cap, semester number)

    GPT-4o receives the pre-computed plan and only explains / justifies it.

    Priority ordering for candidates:
      P0  retake      — failed courses, must repeat (always included first)
      P1  current_sem — in official current-semester plan, eligible
      P2  next_sem    — in official next-semester plan, eligible early
      P3  catch_up    — planned for an earlier semester, eligible for catch-up
      P4  elective    — other eligible courses (not premature)
      ---  premature  — 2+ semesters ahead → excluded from all three plans
    """

    def compute(
        self,
        profile:     StudentProfile,
        analysis:    dict,
        progression: CurriculumProgressionMetrics,
    ) -> ComputedPlan:
        candidates = self._build_candidates(profile, analysis)
        tiers = [
            ("Safe",     12, 15, "LOW"),
            ("Balanced", 15, 18, "MEDIUM"),
            ("Fast",     18, 21, "HIGH"),
        ]
        plans = [
            self._fill_plan(label, lo, hi, risk, candidates, profile)
            for label, lo, hi, risk in tiers
        ]
        return ComputedPlan(
            safe=plans[0], balanced=plans[1], fast=plans[2],
            progression=progression,
        )

    # ── Candidate builder ──────────────────────────────────────────────────────

    def _build_candidates(
        self, profile: StudentProfile, analysis: dict
    ) -> list[CourseRecommendation]:
        premature_codes: set[str] = {e[0] for e in analysis.get("premature_eligible", [])}
        seen:            set[str] = set()
        candidates:      list     = []

        def _push(entries: list, priority: str, reason: str) -> None:
            for e in entries:
                code = e[0]
                if code in seen:
                    continue
                seen.add(code)
                candidates.append(CourseRecommendation(
                    code=code, name=e[1], credits=e[2],
                    reason=reason, plan_sem="", priority=priority,
                ))

        _push(
            analysis.get("retake_eligible", []),
            "retake",
            "Failed — must retake; all prerequisites satisfied",
        )
        _push(
            analysis.get("current_sem_eligible", []),
            "current_sem",
            f"Official semester {profile.semester} course (per study plan)",
        )
        _push(
            analysis.get("next_sem_eligible", []),
            "next_sem",
            f"Official semester {(profile.semester or 0) + 1} course — eligible early",
        )
        _push(
            analysis.get("behind_plan_eligible", []),
            "catch_up",
            "Planned for an earlier semester — catch-up recommended",
        )

        # Remaining eligible courses that are not premature
        for e in analysis.get("eligible", []):
            code = e[0]
            if code in seen or code in premature_codes:
                continue
            seen.add(code)
            candidates.append(CourseRecommendation(
                code=code, name=e[1], credits=e[2],
                reason="Eligible elective / out-of-plan option",
                plan_sem="", priority="elective",
            ))

        return candidates

    # ── Plan builder ───────────────────────────────────────────────────────────

    def _fill_plan(
        self,
        label: str, lo: int, hi: int, risk: str,
        candidates: list[CourseRecommendation],
        profile:    StudentProfile,
    ) -> SemesterPlan:
        notes: list[str] = []

        # Apply GPA credit cap
        if profile.gpa and profile.gpa < _GPA_PROBATION:
            hi   = min(hi, 12)
            risk = "HIGH"
            notes.append(
                f"GPA {profile.gpa} below probation threshold — "
                "credit cap reduced to 12."
            )
        elif profile.gpa and profile.gpa < _GPA_WARNING:
            hi = min(hi, 15)
            notes.append(
                f"GPA {profile.gpa} in warning zone — "
                "credit cap reduced to 15."
            )

        selected: list[CourseRecommendation] = []
        total = 0

        # Retakes are mandatory — always include them first
        retakes   = [c for c in candidates if c.priority == "retake"]
        retake_cr = sum(c.credits for c in retakes)

        if retake_cr > hi:
            for c in retakes:
                if total + c.credits <= hi:
                    selected.append(c)
                    total += c.credits
            notes.append(
                "Retake list exceeds credit cap — partial retake only. "
                "Prioritize these above all other courses."
            )
        else:
            for c in retakes:
                selected.append(c)
                total += c.credits

        # Fill remaining capacity in priority order
        for prio in ("current_sem", "next_sem", "catch_up", "elective"):
            for c in candidates:
                if c.priority != prio or c in selected:
                    continue
                if total + c.credits <= hi:
                    selected.append(c)
                    total += c.credits

        if not selected:
            notes.append(
                "No eligible courses resolved — student may need to provide "
                "exact course codes, or phase8a may need to be re-run."
            )

        return SemesterPlan(
            label=label, min_cr=lo, max_cr=hi,
            risk_level=risk, courses=selected,
            total_credits=total, notes=notes,
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

        has_question = "?" in query or bool(re.match(
            r'\b(what|which|how|when|can|do|does|is|are|will|should|would|could|tell)\b',
            q,
        ))

        # Personal self-identification signals (distinct from "SCI electives?" which
        # mentions a school name without claiming to be in it).
        is_personal = bool(
            _PERSONAL_RE.search(query)
            or _GPA_RE.search(query)
            or _COMPLETED_RE.search(query)
            or _FAILED_RE.search(query)
        )

        if is_personal:
            profile_complete = bool(
                profile.semester and (profile.school or profile.major)
            )
            if profile_complete:
                return "planning"  # auto-plan whenever we have enough to act
            if not has_question:
                return "profile"  # partial profile, no question → acknowledge

        return "general"


# ══════════════════════════════════════════════════════════════════════════════
#  Advisor System Prompt (Engine 8 — Decision Support)
# ══════════════════════════════════════════════════════════════════════════════

# Relevant course code prefixes per school — used to filter blocked lists so
# off-school courses (e.g. CHEM 203 for a CSAI/DSAI student) are not shown.
_SCHOOL_CODE_PREFIXES: dict[str, frozenset] = {
    # BIOL/CHEM removed from CSAI — those are SCI-school courses and must not
    # appear for DSAI/CSAI/SWD/IT students even as electives.
    "CSAI": frozenset({
        "CSAI", "DSAI", "CIE", "MATH", "PHYS", "ENGL", "SCH", "SW",
    }),
    "BUS":  frozenset({"BUS",  "MATH", "ENGL", "ECON", "SCH"}),
    "SCI":  frozenset({"SCI",  "MATH", "PHYS", "BIOL", "CHEM", "ENGL", "SCH"}),
    "ENGR": frozenset({"ENGR", "CIE",  "MATH", "PHYS", "ENGL", "SCH"}),
}

_ADVISOR_SYSTEM = """\
You are an academic advisor at Zewail City of Science and Technology (UST).
You are having a direct, helpful conversation with a student - not filling out a form.

IDENTITY AND TONE
Respond like a knowledgeable Zewail advisor who already has the student's file.
- Be direct and practical. Answer the question first, then explain.
- Do NOT generate sections or headers unless they add genuine value.
- Do NOT pad responses with generic risk paragraphs, empty summaries, or
  warnings that don't apply to this specific student.
- If the student asks a simple question, give a simple answer.

NON-NEGOTIABLE RULES (always apply)
1. NEVER recommend a course that is Blocked (missing prerequisites).
2. NEVER add, remove, or change any course in a pre-computed plan - the planning
   engine selected them; your job is to present and explain them.
3. Use ONLY real Zewail course codes (e.g. CSAI 201, MATH 103).
   Never use placeholders: "Elective", "General Education", "BUS 101".
4. Failed courses must be retaken. They block all downstream courses.
5. Typical load: 15-18 cr/semester. GPA < 2.0 -> limit to 12 cr.
   GPA 2.0-2.49 -> flag academic warning risk; suggest lighter load.
6. The Prerequisite Graph Analysis in the prompt is computed from the official
   Zewail curriculum. It is authoritative - trust it, do not override it.
7. Only show risks, graduation warnings, or blocked courses when they directly
   affect this student's next steps. Skip sections with nothing meaningful to say.

CURRICULUM-FIRST KNOWLEDGE
The Python planning engine has already:
  - Loaded the student's programme from the official Zewail study plan
  - Inferred completed courses from the official plan for semesters 1..N-1
    (adjusted for any failed/missing courses the student declared)
  - Computed eligibility via the prerequisite graph
  - Selected the recommended courses for Safe / Balanced / Fast plans

You do NOT re-derive any of this. You present and explain the output.

Curriculum Progression labels (when present):
  on_track -> Student completed expected courses; plans are standard this semester.
  ahead    -> Student completed extra courses; may have more options.
  behind   -> Student is missing courses from earlier semesters; catch-up courses
             are included in the plan - explain which ones and why they matter.

PLANNING RESPONSES - NATURAL FORMAT
When giving semester plans, always start with the recommended courses - not
summaries, risk scores, or statistics. The student wants to know what to register
for FIRST, then the reasoning.

Suggested flow (adapt to the question - not all sections are always needed):
  1. Recommended plan table(s) - lead with this
  2. Brief reason for each course (1 line each)
  3. Risk or warnings - ONLY if they materially affect the recommendation
  4. Graduation impact - ONLY if delayed or notable
  5. One practical note if needed (e.g. "register early for CSAI 201")

BLOCKED COURSES: Only mention the 1-5 blocked courses that directly affect what
the student would want to take this semester. Do NOT list the full blocked course
dump. If a course important to the student is blocked, explain which prerequisite
is missing and how to unlock it.

PREREQUISITE QUESTIONS
Answer directly and concisely. No semester plans unless asked.
Show: direct prerequisites -> full chain -> whether the student can register now.
List what becomes available after completing the courses (up to ~10, relevant ones).

RISK AND GENERAL QUESTIONS
Answer the question asked. Include risk information only when it is real and specific
to this student. Do not generate hypothetical or generic warnings.

PLAN-DRIVEN MODE (when Recommended Courses section is provided)
The course tables below are FINAL - do NOT add, remove, or swap any course.
Your role: write naturally around the tables. Start with the plan, then explain.
Priority labels in the plan:
  retake      -> Failed - must retake; appears in every load option
  current_sem -> Official semester N course
  next_sem    -> Official semester N+1 course - eligible early
  catch_up    -> Planned for an earlier semester; student needs to catch up
  elective    -> Other eligible course
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
        self._rag        = rag
        self._graph      = CurriculumGraph()
        self._study_plan = StudyPlanEngine()
        self._planner    = PlanningEngine()
        self._router     = IntentRouter()

        graph_status = (
            f"CurriculumGraph: {self._graph.course_count} courses loaded"
            if self._graph.available
            else "CurriculumGraph: no courses.json (run phase8a first)"
        )
        plan_status = (
            "StudyPlanEngine: study_plans.json loaded"
            if self._study_plan.available
            else "StudyPlanEngine: study_plans.json missing"
        )
        print(f"[AdvisorEngine] {graph_status} | {plan_status}")

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
            # Pre-flight safety check for planning intent
            if intent == "planning":
                missing = self._check_planning_readiness(question, profile)
                if missing:
                    return self._ask_planning_questions(missing, profile), []
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

    # ── Planning safety pre-flight check ──────────────────────────────────────

    def _check_planning_readiness(
        self, question: str, profile: StudentProfile
    ) -> list[str]:
        """
        Return list of info pieces still needed before a safe plan can be built.
        Empty list means we have enough to proceed.

        Curriculum-First Policy: only programme and semester are required.
        The system infers completed courses from the official study plan
        (semesters 1..N-1) and applies student-declared deviations (failed /
        explicitly-missing courses) on top.  Students are NEVER asked to
        reconstruct their entire course history.
        """
        missing: list[str] = []

        if not profile.school and not profile.major:
            missing.append("programme")

        if not profile.semester:
            missing.append("current_semester")

        # For Semester 2+, require a specific programme/concentration.
        # Semester 1 exception: common first-year curriculum is shared, no track needed.
        if (
            "programme" not in missing
            and "current_semester" not in missing
            and profile.semester >= 2
        ):
            school_upper = (profile.school or "").upper()
            major_upper  = (profile.major  or "").upper()

            # Level 1 — generic school code: student said "CSAI"/"SCI"/"BUS"
            # without specifying which programme (DSAI/SWD/IT, BMS/NANO/PHY, etc.)
            if school_upper in _GENERIC_SCHOOL_CODES and major_upper == school_upper:
                missing.append("track")

            # Level 2 — specific programme but it has mandatory sub-concentrations
            # (e.g. SWD → APD/GCG/HCI; BMS → CBG/MCB/DDD/MED)
            elif not profile.concentration and self._study_plan.available:
                res = self._study_plan.resolve_prog(profile.school, profile.major)
                if res:
                    sub_tracks = self._study_plan.get_track_options(res[0], res[1])
                    if sub_tracks:
                        missing.append("track")

        return missing

    def _ask_planning_questions(
        self, missing: list[str], profile: StudentProfile
    ) -> str:
        """
        Return a targeted follow-up question asking only for the minimum needed.
        The system infers course history from the official plan — students only
        need to declare their programme, semester, and any deviations (failed
        courses, courses they haven't taken yet).
        """
        lines: list[str] = []
        q_num = 1

        if "programme" in missing:
            lines.append(
                f"**{q_num}. What is your programme or major?**\n"
                "   e.g. DSAI, CSAI (Computer Science/AI), BUS (Finance, Business Analytics), "
                "ENGR (Aerospace, CIE), SCI (Biomedical, Physics), etc."
            )
            q_num += 1

        if "current_semester" in missing:
            lines.append(
                f"**{q_num}. Which semester are you currently in?**\n"
                "   e.g. Semester 3, Semester 5, etc."
            )
            q_num += 1

        if "track" in missing:
            school_upper = (profile.school or "").upper()
            major_upper  = (profile.major  or "").upper()

            if school_upper in _GENERIC_SCHOOL_CODES and major_upper == school_upper:
                # Level 1: student gave a generic school name — need specific programme
                opts = _TRACK_OPTIONS.get(school_upper, "your specific major or concentration")
                lines.append(
                    f"**{q_num}. Which specific programme or major are you in?**\n"
                    f"   Within {school_upper}, options include: {opts}"
                )
            else:
                # Level 2: programme is known but has mandatory sub-concentrations
                res = self._study_plan.resolve_prog(profile.school, profile.major) if self._study_plan.available else None
                if res:
                    sub_tracks = self._study_plan.get_track_options(res[0], res[1])
                    opts = ", ".join(sub_tracks) if sub_tracks else "your concentration"
                    lines.append(
                        f"**{q_num}. Which concentration are you in within "
                        f"{profile.major or res[1]}?**\n"
                        f"   Options: {opts}"
                    )
                else:
                    lines.append(
                        f"**{q_num}. What is your specific concentration or track?**"
                    )
            q_num += 1

        if lines:
            lines.insert(0, "To build an accurate semester plan, I just need one more detail:\n")
            lines.append(
                "\n*I load your curriculum automatically from the official Zewail study plan — "
                "you only need to mention courses you **failed** or haven't completed yet.*"
            )

        return "\n\n".join(lines)

    # ── Pre-format plan output for plan-driven GPT template ───────────────────

    def _preformat_plan_tables(
        self, computed_plan: ComputedPlan, profile: StudentProfile
    ) -> str:
        """Generate the ## Recommended Plans markdown with exact courses pre-filled."""
        sem = profile.semester or 1
        lines = ["## Recommended Plans", ""]

        for tier in (computed_plan.safe, computed_plan.balanced, computed_plan.fast):
            lines.append(f"### {tier.label} Plan (~{tier.min_cr}–{tier.max_cr} credits)")
            lines.append("| Course | Credits | Why |")
            lines.append("|--------|---------|-----|")
            for c in tier.courses:
                why = {
                    "retake":      "RETAKE — failed; must repeat",
                    "current_sem": f"Official Semester {sem} course",
                    "next_sem":    f"Official Semester {sem + 1} course — eligible early",
                    "catch_up":    "Catch-up — planned for an earlier semester",
                    "elective":    "Eligible elective",
                }.get(c.priority, c.priority)
                lines.append(f"| {c.code} | {c.credits} | {why} |")
            lines.append(
                f"**Total: {tier.total_credits} credits | Risk: {tier.risk_level}**"
            )
            for note in tier.notes:
                lines.append(f"*Note: {note}*")
            lines.append("")

        return "\n".join(lines)

    def _preformat_eligibility_for_plan(
        self, analysis: dict, computed_plan: ComputedPlan, profile: StudentProfile
    ) -> str:
        """
        Eligibility context for GPT.  Can Take / Must Retake from the plan,
        plus only the blocked courses that directly affect the student's
        next-semester registration (not the full university dump).
        """
        sem = profile.semester or 1
        cur_elig = analysis.get("current_sem_eligible", [])
        nxt_elig = analysis.get("next_sem_eligible", [])
        beh_elig = analysis.get("behind_plan_eligible", [])
        retake   = analysis.get("retake_eligible", [])
        blocked  = analysis.get("blocked", [])

        # Programme-relevance filter
        school  = profile.school.upper() if profile.school else ""
        allowed = _SCHOOL_CODE_PREFIXES.get(school, frozenset())
        if allowed:
            blocked = [b for b in blocked if b[0].split()[0] in allowed]

        # Show only blocked courses that are in the current or next-semester
        # official plan — these are the ones the student would most want to know about.
        plan_codes = (
            {e[0] for e in cur_elig}
            | {e[0] for e in nxt_elig}
            | {e[0] for e in beh_elig}
        )
        relevant_blocked = [b for b in blocked if b[0] in plan_codes]
        # If none from the plan are blocked, fall back to fewest-missing first
        if not relevant_blocked:
            relevant_blocked = blocked[:4]

        lines = ["## Eligibility Analysis", "**Can Take (prerequisites met):**"]
        any_found = False
        for code, name, cr in cur_elig:
            lines.append(f"* {code} — {name} ({cr} cr) — Semester {sem} plan course")
            any_found = True
        for code, name, cr in nxt_elig:
            lines.append(f"* {code} — {name} ({cr} cr) — Semester {sem + 1} plan, eligible early")
            any_found = True
        for code, name, cr in beh_elig:
            lines.append(f"* {code} — {name} ({cr} cr) — catch-up (planned earlier)")
            any_found = True
        if not any_found:
            lines.append("* (none from study plan)")
        lines.append("")

        if retake:
            lines.append("**Must Retake (failed):**")
            for code, name, cr in retake:
                lines.append(f"* {code} — {name} ({cr} cr)")
            lines.append("")

        if relevant_blocked:
            lines.append("**Blocked (affects next-semester registration):**")
            for code, name, cr, missing in relevant_blocked[:5]:
                lines.append(
                    f"* {code} — {name} ({cr} cr) — missing: {', '.join(missing[:2])}"
                )

        return "\n".join(lines)

    def _preformat_progression_summary(
        self,
        progression: Optional[CurriculumProgressionMetrics],
        profile: StudentProfile,
    ) -> str:
        prog = profile.school or profile.major or "unknown programme"
        sem  = profile.semester or 1
        base = f"Programme: {prog}, Semester: {sem}."

        if not progression or progression.status == "unknown":
            return base

        status_line = progression.status_line()
        parts = [
            base,
            f"Curriculum status: {status_line}.",
            f"Expected {progression.total_expected} courses completed before this semester; "
            f"confirmed {progression.total_completed}.",
        ]
        if progression.missing_core:
            sample = ", ".join(progression.missing_core[:5])
            trail  = (
                f" (+{len(progression.missing_core)-5} more)"
                if len(progression.missing_core) > 5 else ""
            )
            parts.append(f"Delayed core courses: {sample}{trail}.")
        return " ".join(parts)

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

        # 3. Resolve study-plan programme and build plan context
        plan_context:       Optional[dict]                              = None
        progression:        Optional[CurriculumProgressionMetrics]      = None
        plan_resolved:      Optional[tuple[str, str]]                   = None
        s_key = p_key = track_key = ""
        presumed_completed: list[str]                                   = []

        if self._study_plan.available and profile.semester:
            plan_resolved = self._study_plan.resolve_prog(profile.school, profile.major)
            if plan_resolved:
                s_key, p_key = plan_resolved
                conc      = profile.concentration.upper().strip() if profile.concentration else None
                track_key = (
                    conc
                    if conc and conc in self._study_plan.get_track_options(s_key, p_key)
                    else None
                )
                cur_seq = profile.semester

                cur_codes = set(self._study_plan.get_current_sem_codes(
                    s_key, p_key, track_key, cur_seq))
                nxt_codes = set(self._study_plan.get_next_sem_codes(
                    s_key, p_key, track_key, cur_seq))
                pre_codes = set(self._study_plan.get_premature_codes(
                    s_key, p_key, track_key, cur_seq))
                bef_codes = set(self._study_plan.get_planned_codes_before(
                    s_key, p_key, track_key, cur_seq))

                plan_context = {
                    "current_sem_codes":    cur_codes,
                    "next_sem_codes":       nxt_codes,
                    "premature_codes":      pre_codes,
                    "planned_before_codes": bef_codes,
                }

                # ── Transcript Inference Rule ──────────────────────────────────
                # Standard case: assume semesters 1..N-1 completed.
                if cur_seq > 1:
                    presumed_completed = self._study_plan.infer_presumed_completed(
                        profile.school or "", profile.major or "",
                        cur_seq, track_key,
                    )

        # ── Failed-in-current-semester detection ───────────────────────────────
        # If the student's failed courses include courses from the CURRENT
        # semester plan (e.g. "semester 1, failed MATH 103"), they have already
        # been through that semester.  Extend presumed_completed with the rest of
        # that semester's courses and advance the planning horizon by one.
        failed_upper   = {c.upper() for c in profile.failed_courses}
        explicit_upper = {c.upper() for c in profile.completed_courses}
        planning_sem   = cur_seq  # may be bumped to cur_seq+1 below

        if plan_resolved and plan_context:
            cur_sem_set = plan_context["current_sem_codes"]
            failed_in_cur_sem = cur_sem_set & failed_upper
            if failed_in_cur_sem:
                # Student has completed semester cur_seq (except the failed course).
                # Infer the rest of sem cur_seq as done and plan for sem cur_seq+1.
                already_presumed_upper = {c.upper() for c in presumed_completed}
                for code in cur_sem_set:
                    uc = code.upper()
                    if uc not in failed_upper and uc not in already_presumed_upper:
                        presumed_completed.append(uc)
                planning_sem = cur_seq + 1
                # Recompute plan_context for the new planning semester
                plan_context = {
                    "current_sem_codes":    set(self._study_plan.get_current_sem_codes(
                                                s_key, p_key, track_key, planning_sem)),
                    "next_sem_codes":       set(self._study_plan.get_next_sem_codes(
                                                s_key, p_key, track_key, planning_sem)),
                    "premature_codes":      set(self._study_plan.get_premature_codes(
                                                s_key, p_key, track_key, planning_sem)),
                    "planned_before_codes": set(self._study_plan.get_planned_codes_before(
                                                s_key, p_key, track_key, planning_sem)),
                }

        # Build effective completed: presumed ∪ explicit − failed
        effective_completed: list[str] = list(
            {c.upper() for c in (presumed_completed + profile.completed_courses)}
            - failed_upper
        )
        inferred_codes = [
            c.upper() for c in presumed_completed
            if c.upper() not in explicit_upper and c.upper() not in failed_upper
        ]

        if self._study_plan.available and profile.semester and plan_resolved and self._graph.available:
            completed_resolved: set[str] = {
                c
                for ref in effective_completed
                if (c := self._graph.resolve_code(ref))
            }
            progression = self._study_plan.get_progression_metrics(
                s_key, p_key, track_key, planning_sem,
                completed_resolved,
                self._graph._courses,
            )

        # 4. Engine 5: Eligibility analysis (prereq graph + plan categorisation)
        analysis:      dict = {}
        graph_section: str  = ""
        if self._graph.available:
            analysis = self._graph.analyze_eligibility(
                completed=effective_completed,
                failed=profile.failed_courses,
                current=profile.current_courses,
                plan_context=plan_context,
            )
            graph_section = self._format_graph_section(analysis, profile)

        # Prerequisite questions: add hypothetical eligibility
        if intent == "prerequisite" and self._graph.available:
            _CODE_UPPER = re.compile(r'\b([A-Z]{2,6})\s+(\d{3,4}[A-Z]?)\b')
            hypo_codes = [
                f"{m.group(1)} {m.group(2)}"
                for m in _CODE_UPPER.finditer(question.upper())
            ]
            if hypo_codes:
                hypo_completed = list(effective_completed) + [
                    c for c in hypo_codes if c not in profile.failed_courses
                ]
                hypo_analysis = self._graph.analyze_eligibility(
                    completed=hypo_completed,
                    failed=profile.failed_courses,
                    current=profile.current_courses,
                    plan_context=plan_context,
                )
                hypo_section = self._format_graph_section(hypo_analysis, profile)
                graph_section = (
                    f"Current eligibility (actual completed courses):\n{graph_section}\n\n"
                    f"Hypothetical eligibility assuming {', '.join(hypo_codes)} "
                    f"are completed:\n{hypo_section}"
                )

        # 5. Engine 7: Graduation estimate (with plan progression)
        grad_section = self._graduation_estimate(profile, progression)

        # 6. Engine 9: Risk flags
        risk_flags = self._risk_flags(profile)

        # 7. Structured study plan context (official semester mapping for LLM)
        study_plan_ctx = self._study_plan.build_study_plan_context(profile)

        # 8. Engine 6: Planning Engine — compute Safe / Balanced / Fast plans
        computed_plan:      Optional[ComputedPlan] = None
        computed_plan_text: str                    = ""
        if intent == "planning" and analysis:
            if progression is None:
                progression = CurriculumProgressionMetrics(
                    status="unknown", expected_done=[], missing_core=[],
                    extra_done=[], delayed_count=0, total_expected=0,
                    total_completed=0, expected_credits=0, actual_credits=0,
                )

            # ── Programme-plan elective filter (Bug B fix) ────────────────────
            # PlanningEngine's P4 (elective) tier draws from analysis["eligible"].
            # Without filtering, any no-prereq course in the entire catalogue
            # (e.g. BIOL 101 for a DSAI student) can appear as an elective.
            # Restrict eligible to courses that exist in the student's programme
            # plan across all semesters.
            if plan_resolved:
                all_plan_codes: set[str] = set(
                    self._study_plan.get_planned_codes_before(
                        s_key, p_key, track_key, 99  # seq < 99 = all regular semesters
                    )
                )
                # Also include the current and next semester codes
                all_plan_codes |= plan_context.get("current_sem_codes", set())
                all_plan_codes |= plan_context.get("next_sem_codes", set())
                all_plan_codes_upper = {c.upper() for c in all_plan_codes}

                plan_analysis = dict(analysis)
                plan_analysis["eligible"] = [
                    e for e in analysis["eligible"]
                    if e[0].upper() in all_plan_codes_upper
                ]
            else:
                plan_analysis = analysis

            # Use planning_sem for profile so PlanningEngine labels are correct
            profile_for_plan = StudentProfile.from_dict(profile.to_dict())
            profile_for_plan.semester = planning_sem

            computed_plan      = self._planner.compute(profile_for_plan, plan_analysis, progression)
            computed_plan_text = computed_plan.to_prompt_text()

        # 9. Assemble full prompt — instruction varies by intent
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
                "Do NOT generate semester plans. "
                "Be direct: state the prerequisite(s), the full unlock chain if relevant, "
                "and whether the student can register now. "
                "Mention what becomes available after completing the course (up to ~5, most relevant). "
                "Use real course codes only. Keep the response concise."
            )
        elif intent == "risk":
            task_instruction = (
                "Assess this student's academic risk based on their profile. "
                "Be specific — name the actual GPA, failed courses, or credit load that creates the risk. "
                "Give one concrete recommendation per risk. "
                "Do NOT generate semester plans."
            )
        elif intent == "graduation":
            task_instruction = (
                "Focus on the graduation timeline. "
                "Report: estimated credits completed, credits remaining, and how many semesters to finish "
                "at the student's current pace. "
                "If curriculum-first inference was used, label the credit count as estimated. "
                "Keep Recommended Plans to ONE concise table (most realistic option). "
                "Emphasise what the student must do to graduate on time."
            )
        else:  # planning
            prog = profile.school or profile.major or "their"
            # Use planning_sem (may be cur_seq+1 when failed-in-current-sem detected)
            sem  = planning_sem if plan_resolved else (profile.semester or 1)
            if computed_plan and computed_plan_text:
                # ── PLAN-DRIVEN MODE ─────────────────────────────────────────
                # Python has selected all courses deterministically.
                # GPT writes natural advisor language around the pre-built tables.
                plan_tables_md = self._preformat_plan_tables(computed_plan, profile_for_plan)
                eligibility_md = self._preformat_eligibility_for_plan(
                    analysis, computed_plan, profile_for_plan
                )
                prog_summary = self._preformat_progression_summary(progression, profile_for_plan)

                # Inference note (shown when curriculum-first inference was used)
                if inferred_codes:
                    sem_range = f"semesters 1-{sem - 1}" if sem > 2 else "semester 1"
                    assumption_note = (
                        f"NOTE: The student did not provide a transcript. "
                        f"The system assumed they completed the standard {sem_range} courses "
                        f"from the official {profile.major or profile.school} study plan "
                        f"({len(inferred_codes)} courses auto-loaded). "
                        f"Mention this briefly and naturally: e.g. 'Since you're in semester {sem}, "
                        f"I've loaded your {sem_range} courses from the official plan. "
                        f"Let me know if any are missing or failed.'"
                    )
                else:
                    assumption_note = ""

                task_instruction = (
                    "PLAN-DRIVEN MODE: The recommended courses below are FINAL - "
                    "do NOT add, remove, or change any course.\n\n"
                    "Write a natural, conversational advisor response that:\n"
                    "1. Starts directly with the recommended plan (lead with the table)\n"
                    "2. Gives a brief reason for each course (1 sentence)\n"
                    "3. Mentions risks or graduation impact ONLY if materially relevant\n"
                    "4. Ends with 1-2 practical notes if needed\n\n"
                    "Do NOT generate empty sections. Do NOT force headers for every topic.\n"
                    "Be concise. A student wants to know what to take and why - not read a report.\n\n"
                    + (f"{assumption_note}\n\n" if assumption_note else "")
                    + f"Context: {prog_summary}\n\n"
                    + f"{eligibility_md}\n\n"
                    + f"{plan_tables_md}"
                )
            else:
                task_instruction = (
                    f"This student is in the {prog} programme, semester {sem}. "
                    f"Recommend ONLY {prog}-relevant course codes from the official plan. "
                    "Do NOT invent codes or use placeholders. "
                    "Use the Structured Study Plan as primary source for which courses "
                    f"belong to semester {sem}. Prioritize current and next-semester plan "
                    "courses that are eligible (prerequisites met). "
                    "Start with the recommended courses, then explain briefly. "
                    "Only mention risks if they materially affect this student."
                )

        plan_section = (
            f"## Computed Course Plans (engine reference)\n{computed_plan_text}\n\n"
            if computed_plan_text else ""
        )

        # In plan-driven mode the task_instruction already contains the filtered
        # eligibility tables (_preformat_eligibility_for_plan).  The full
        # graph_section would expose an unfiltered eligibility dump (potentially
        # 60+ courses and a full blocked list) that GPT might reference verbatim.
        # Suppress it when a pre-computed plan is available; for non-plan intents
        # (prereq, risk, graduation) the full section remains useful.
        if computed_plan and intent == "planning":
            graph_ctx = ""  # eligibility is already in task_instruction
        else:
            graph_ctx = f"## Prerequisite Graph Analysis\n{graph_section}\n\n"

        user_message = (
            f"## Student Profile\n{profile.summary_for_prompt()}\n\n"
            f"## Official Study Plan\n{study_plan_ctx}\n\n"
            f"{graph_ctx}"
            f"## Graduation Estimate\n{grad_section}\n\n"
            f"## Academic Risk Flags\n{risk_flags}\n\n"
            f"{plan_section}"
            f"## Curriculum Reference Documents\n{rag_ctx}\n\n"
            f"## Student's Question\n{question}\n\n"
            f"## Task\n{task_instruction}"
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
        retake    = analysis["retake_eligible"]
        done_n    = len(analysis["completed_codes"])
        failed_n  = len(analysis["failed_codes"])

        # Filter eligible and blocked to programme-relevant codes only.
        school  = profile.school.upper() if profile.school else ""
        allowed = _SCHOOL_CODE_PREFIXES.get(school, frozenset())
        if allowed:
            eligible = [e for e in eligible  if e[0].split()[0] in allowed]
        blocked = [
            b for b in analysis["blocked"]
            if not allowed or b[0].split()[0] in allowed
        ]

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
            # Show only the most relevant blocked courses — those in the
            # current/next plan (if any), otherwise fewest-missing-prereqs first.
            plan_codes = (
                {e[0] for e in analysis.get("current_sem_eligible", [])}
                | {e[0] for e in analysis.get("next_sem_eligible", [])}
                | {e[0] for e in analysis.get("behind_plan_eligible", [])}
            )
            plan_blocked = [b for b in blocked if b[0] in plan_codes] or blocked
            lines.append("Blocked courses directly relevant to next-semester planning (up to 5):")
            for code, name, cr, missing in plan_blocked[:5]:
                lines.append(f"  x {code} — {name} ({cr} cr) | needs: {', '.join(missing[:2])}")

        if not eligible and not retake:
            lines.append(
                "NOTE: Could not resolve any completed courses to known course codes. "
                "The student may have stated them by name rather than code. "
                "Use the curriculum documents to infer eligible courses from the stated course names."
            )

        # Plan-categorised overview (only when plan context was available)
        if analysis.get("plan_context_available"):
            cur_elig = analysis.get("current_sem_eligible", [])
            nxt_elig = analysis.get("next_sem_eligible",    [])
            beh_elig = analysis.get("behind_plan_eligible", [])
            pre_elig = analysis.get("premature_eligible",   [])
            sem      = profile.semester or 0

            lines.append("\nPlan-categorised eligible courses:")
            if cur_elig:
                lines.append(
                    f"  [current_sem] In official semester {sem} plan "
                    f"({len(cur_elig)}): "
                    + ", ".join(f"{c[0]} ({c[2]}cr)" for c in cur_elig[:12])
                    + (" …" if len(cur_elig) > 12 else "")
                )
            if nxt_elig:
                lines.append(
                    f"  [next_sem] In official semester {sem + 1} plan "
                    f"({len(nxt_elig)}): "
                    + ", ".join(f"{c[0]} ({c[2]}cr)" for c in nxt_elig[:12])
                    + (" …" if len(nxt_elig) > 12 else "")
                )
            if beh_elig:
                lines.append(
                    f"  [catch_up] Planned for an earlier semester — catch-up "
                    f"({len(beh_elig)}): "
                    + ", ".join(f"{c[0]} ({c[2]}cr)" for c in beh_elig[:8])
                    + (" …" if len(beh_elig) > 8 else "")
                )
            if pre_elig:
                lines.append(
                    f"  [premature] Prerequisites met but {_PREMATURE_THRESHOLD}+ "
                    f"semesters ahead — SUPPRESSED from plans ({len(pre_elig)}): "
                    + ", ".join(c[0] for c in pre_elig[:10])
                    + (" …" if len(pre_elig) > 10 else "")
                )

        return "\n".join(lines)

    # ── Engine 7: Graduation estimate ─────────────────────────────────────────

    def _graduation_estimate(
        self,
        profile:     StudentProfile,
        progression: Optional[CurriculumProgressionMetrics] = None,
    ) -> str:
        total_needed = profile.total_credits_needed()

        if profile.completed_credits:
            done      = profile.completed_credits
            remaining = max(0, total_needed - done)
            basis     = f"Based on {done} completed credits (as stated by student)"
        elif progression and progression.actual_credits > 0:
            # Curriculum-first inference: use credits estimated from the official study plan
            done      = progression.actual_credits
            remaining = max(0, total_needed - done)
            basis     = (
                f"Based on ~{done} estimated credits "
                f"(inferred from official study plan — semesters 1–{max(1, profile.semester - 1)}). "
                "Label this as estimated in your response."
            )
        else:
            remaining = total_needed
            done      = 0
            basis     = (
                "NOTE: No completed credits or transcript provided. "
                "Remaining credits shown as full programme total. "
                "Do NOT invent a credit count — state that the estimate is unavailable "
                "without a transcript."
            )

        sems_at_safe     = max(1, -(-remaining // _LOAD_SAFE[1]))
        sems_at_balanced = max(1, -(-remaining // _LOAD_BALANCED[1]))
        sems_at_fast     = max(1, -(-remaining // _LOAD_FAST[1]))

        lines = [
            basis,
            f"Total credits needed for graduation: {total_needed}",
            f"Credits completed (confirmed): {done}",
            f"Credits remaining: {remaining}",
            f"Semesters to graduation at Safe load (~{_LOAD_SAFE[1]} cr): {sems_at_safe}",
            f"Semesters to graduation at Balanced load (~{_LOAD_BALANCED[1]} cr): {sems_at_balanced}",
            f"Semesters to graduation at Fast load (~{_LOAD_FAST[1]} cr): {sems_at_fast}",
        ]
        if profile.failed_courses:
            lines.append(
                f"Note: {len(profile.failed_courses)} failed course(s) must be retaken "
                "— add ~1 semester if they are not completed in the next cycle."
            )

        # Plan-based progression comparison
        if progression and progression.status != "unknown":
            lines.append("")
            lines.append("Curriculum Progression (vs official study plan):")
            lines.append(f"  Status: {progression.status_line()}")
            lines.append(
                f"  Expected completed before semester {profile.semester}: "
                f"{progression.total_expected} course(s) "
                f"({progression.expected_credits} cr)"
            )
            lines.append(
                f"  Confirmed completed (resolved to course codes): "
                f"{progression.total_completed} course(s) "
                f"({progression.actual_credits} cr)"
            )
            if progression.missing_core:
                sample = ", ".join(progression.missing_core[:8])
                trail  = (
                    f" … (+{len(progression.missing_core) - 8} more)"
                    if len(progression.missing_core) > 8 else ""
                )
                lines.append(
                    f"  Delayed core courses ({progression.delayed_count}): {sample}{trail}"
                )
            if progression.status == "behind":
                est_extra = max(1, -(-progression.delayed_count // 4))
                lines.append(
                    f"  Estimated additional semesters due to plan delays: ~{est_extra}"
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
