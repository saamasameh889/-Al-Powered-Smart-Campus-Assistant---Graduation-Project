"""
Google Classroom Importer — Learning Analytics XAI

Fetches all historical course performance data from Google Classroom API
and normalises it into CourseRecord objects for the XAI pipeline.

Source of truth: Google Classroom historical data.
Never infers course history from semester number.
Never assumes the student followed the official study plan sequence.

Requires (add to requirements.txt):
  google-api-python-client
  google-auth-oauthlib
  google-auth-httplib2

Environment variables:
  GOOGLE_CLIENT_ID       — from GCP OAuth 2.0 credentials
  GOOGLE_CLIENT_SECRET   — from GCP OAuth 2.0 credentials
  GOOGLE_REDIRECT_URI    — e.g. http://localhost:8501 (Streamlit)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DATA = Path(__file__).parent.parent / "data"

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from google.oauth2.credentials import Credentials
    _GOOGLE_AVAILABLE = True
except ImportError:
    _GOOGLE_AVAILABLE = False

# ── Scopes required ───────────────────────────────────────────────────────────
CLASSROOM_SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.me.readonly",
    "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
]

# ── Component type detection keywords ─────────────────────────────────────────
_COMP_KW: dict[str, list[str]] = {
    "midterm":    ["midterm", "mid term", "mid-term", "mid exam", "midexam"],
    "final":      ["final", "final exam", "end term", "endterm", "end-term"],
    "assignment": ["assignment", "homework", "project", "coursework", "report", "task"],
    "quiz":       ["quiz", "quizzes", "test", "pop quiz"],
    "lab":        ["lab", "laboratory", "practical"],
    "attendance": ["attendance", "attend", "lab attendance"],
}

# ── Zewail City academic regulations (loaded from data files) ─────────────────
# Source: academic_regulations.json + gpa_rules.json
# These values are the ground truth from the Zewail City handbook.

def _load_final_max_marks() -> float:
    """
    Load the maximum final exam mark from academic_regulations.json.
    Used only for predicted_final_score computation (informational display only —
    NOT used for pass/fail inference or GPA calculation).
    Falls back to 40.0 (Zewail standard).
    """
    try:
        regs_data = json.loads((_DATA / "academic_regulations.json").read_text("utf-8"))
        return float(
            regs_data.get("passing_conditions", {}).get("minimum_final_exam_marks_out_of")
            or regs_data.get("grade_structure", {}).get("final_exam_marks", 40.0)
        )
    except Exception:
        return 40.0

_FINAL_MAX_MARKS: float = _load_final_max_marks()

# ── Grade component structure (Zewail standard) ───────────────────────────────
# Full course = 100 marks:
#   Classroom portion = 60 marks (assignments, quizzes, labs, midterm — all in Classroom)
#   Final exam        = 40 marks (secret; never uploaded to Classroom; predicted from coursework)
# Exact classroom split varies per course; we use actual scores from Classroom API.
_CLASSROOM_MAX_MARKS: float = 60.0
_WEIGHTS: dict[str, float] = {
    "assignment": 0.15,   # ─┐
    "quiz":       0.10,   #  │  Classroom portion  (60 / 100)
    "lab":        0.15,   #  │  exact split varies per course
    "midterm":    0.20,   # ─┘
    "attendance": 0.00,   # tracked separately, not in grade total
    "final":      0.40,   # secret final exam (40 / 100); predicted from coursework %
}


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ComponentRecord:
    """A single assessment component within a course."""
    name:       str
    comp_type:  str    # midterm | final | assignment | quiz | lab | attendance
    max_points: float
    score:      Optional[float]   # None = not graded yet
    pct:        Optional[float]   # None = not graded yet


@dataclass
class CourseRecord:
    """
    One course from Google Classroom, normalised for the XAI pipeline.
    All values are sourced from actual Classroom data.
    matched_code and credits are filled by StudentProfileBuilder.

    DESIGN RULE — Classroom is a Coursework Analytics source only.
    It is NOT an official transcript source.  This record describes
    observed coursework activity; it does NOT encode academic outcomes
    (pass/fail, official GPA, graduation eligibility).

    Coursework-status taxonomy
    --------------------------
    pass_status = "completed_coursework"   — graded data available; describes coursework
                                             completion only — NOT an academic outcome
    pass_status = "coursework_in_progress" — active enrollment (with or without initial grades)
    pass_status = "insufficient_data"      — archived with zero or minimal graded components

    Performance-category taxonomy  (derived from overall_pct; Classroom activity only)
    -------------------------------------------------------------------------------------
    performance_category = "excellent"    — overall_pct ≥ 85
    performance_category = "strong"       — overall_pct ≥ 70
    performance_category = "average"      — overall_pct ≥ 55
    performance_category = "weak"         — overall_pct ≥ 40
    performance_category = "at_risk"      — overall_pct > 0 and < 40
    performance_category = "in_progress"  — coursework_in_progress, no grades yet
    performance_category = "no_data"      — insufficient_data
    """
    classroom_id:    str
    course_name:     str
    course_section:  str          # semester label from Classroom section field

    matched_code:    str  = ""    # official code e.g. "CSAI101" (filled by builder)
    credits:         int  = 0     # filled from course_catalog.json by builder; 0 = unresolved
    credits_verified: bool = False  # True only when credits come from the official catalog

    # Source metadata from the Google Classroom API course object
    creation_time:   str  = ""   # ISO 8601, e.g. "2022-09-15T10:30:00Z"
    term_label:      str  = ""   # normalised academic term e.g. "Fall 2022" (derived)

    # Component averages (0–100; 0 means not tracked in this course)
    assignments_avg: float = 0.0
    quizzes_avg:     float = 0.0
    labs_avg:        float = 0.0
    midterm_score:   float = 0.0
    final_score:     float = 0.0    # always 0 — finals never uploaded to Classroom
    attendance_pct:  float = 0.0    # 0 = no explicit attendance component

    # Predicted final exam score (out of 40 marks) — ESTIMATED, informational display only.
    # Computed as:  overall_pct × _FINAL_MAX_MARKS / 100
    # NEVER used for pass/fail inference or GPA calculation.
    predicted_final_score: float = 0.0

    # overall_pct = coursework score as a percentage of observed Classroom marks only.
    # This is NOT the official course grade — the actual total requires the real final exam.
    overall_pct:          float = 0.0

    # Coursework status — describes Classroom activity, NOT academic outcome.
    pass_status:          str   = "insufficient_data"
    # Convenience booleans derived from pass_status.
    coursework_complete:  bool  = False   # True when pass_status == "completed_coursework"
    in_progress:          bool  = False   # True when pass_status == "coursework_in_progress"

    # Performance category — derived from overall_pct; Classroom activity only.
    performance_category: str   = "no_data"

    components: list[ComponentRecord] = field(default_factory=list)


# ── Main importer ─────────────────────────────────────────────────────────────

class ClassroomImporter:
    """
    Fetches and normalises all Google Classroom courses into CourseRecord list.

    Parameters
    ----------
    access_token : str
        OAuth 2.0 access token for the authenticated student.
    """

    def __init__(self, access_token: str) -> None:
        if not _GOOGLE_AVAILABLE:
            raise ImportError(
                "Google API packages not installed. "
                "Run: pip install google-api-python-client "
                "google-auth-oauthlib google-auth-httplib2"
            )
        creds = Credentials(token=access_token)
        self._svc = build("classroom", "v1", credentials=creds, cache_discovery=False)

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_all_records(self) -> list[CourseRecord]:
        """
        Fetch all enrolled courses (active + archived) and their grade data.
        Returns one CourseRecord per course, sorted by section then name.
        """
        raw_courses = self._list_all_courses()
        records: list[CourseRecord] = []
        for course in raw_courses:
            rec = self._build_record(course)
            if rec is not None:
                records.append(rec)
        records.sort(key=lambda r: (_section_sort_key(r.course_section), r.course_name))
        return records

    def get_user_email(self) -> str:
        """Return the authenticated student's email address."""
        try:
            profile = self._svc.userProfiles().get(userId="me").execute()
            return profile.get("emailAddress", "")
        except Exception:
            return ""

    # ── Private helpers ───────────────────────────────────────────────────────

    def _list_all_courses(self) -> list[dict]:
        courses: list[dict] = []
        page_token: Optional[str] = None
        while True:
            try:
                resp = self._svc.courses().list(
                    studentId    = "me",
                    courseStates = ["ACTIVE", "ARCHIVED"],
                    pageSize     = 100,
                    pageToken    = page_token,
                ).execute()
            except HttpError:
                break
            courses.extend(resp.get("courses", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return courses

    def _list_coursework(self, course_id: str) -> list[dict]:
        items: list[dict] = []
        page_token: Optional[str] = None
        while True:
            try:
                resp = self._svc.courses().courseWork().list(
                    courseId  = course_id,
                    pageSize  = 100,
                    pageToken = page_token,
                ).execute()
            except HttpError:
                break
            items.extend(resp.get("courseWork", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return items

    def _get_student_submission(
        self, course_id: str, coursework_id: str
    ) -> Optional[dict]:
        try:
            resp = self._svc.courses().courseWork().studentSubmissions().list(
                courseId     = course_id,
                courseWorkId = coursework_id,
                userId       = "me",
            ).execute()
            subs = resp.get("studentSubmissions", [])
            return subs[0] if subs else None
        except HttpError:
            return None

    def _build_record(self, course: dict) -> Optional[CourseRecord]:
        course_id     = course.get("id", "")
        name          = (course.get("name") or "").strip()
        section       = (course.get("section") or "").strip()
        creation_time = (course.get("creationTime") or "").strip()
        term_lbl      = _infer_term_label(section, creation_time)

        if not name or not course_id:
            return None

        # Fetch coursework catalogue
        try:
            coursework_items = self._list_coursework(course_id)
        except Exception:
            coursework_items = []

        components: list[ComponentRecord] = []
        for cw in coursework_items:
            max_pts = float(cw.get("maxPoints") or 0)
            if max_pts <= 0:
                continue

            sub = self._get_student_submission(course_id, cw["id"])
            if sub is None:
                continue

            assigned = sub.get("assignedGrade")
            draft    = sub.get("draftGrade")
            raw_score = assigned if assigned is not None else draft
            score_val = float(raw_score) if raw_score is not None else None
            pct_val   = (score_val / max_pts * 100) if score_val is not None else None

            comp_type = _detect_component_type(cw.get("title", ""))
            components.append(ComponentRecord(
                name       = cw.get("title", "").strip(),
                comp_type  = comp_type,
                max_points = max_pts,
                score      = score_val,
                pct        = pct_val,
            ))

        # Determine in-progress status.
        # A course is in_progress ONLY when there are zero graded components.
        # Courses where the professor never uploads finals to Classroom (common at Zewail)
        # will have partial grades; we compute the overall from whatever was graded.
        graded_count = sum(1 for c in components if c.score is not None)
        active       = course.get("courseState") == "ACTIVE"
        in_progress  = active and graded_count == 0

        # Bucket graded components by type (for per-component averages)
        buckets: dict[str, list[tuple[float, float]]] = {k: [] for k in _COMP_KW}
        for c in components:
            if c.score is not None and c.max_points > 0:
                buckets[c.comp_type].append((c.score, c.max_points))

        def _avg_pct(pairs: list[tuple[float, float]]) -> float:
            if not pairs:
                return 0.0
            return round(sum(s for s, _ in pairs) / sum(m for _, m in pairs) * 100, 1)

        asgn_avg = _avg_pct(buckets["assignment"])
        quiz_avg = _avg_pct(buckets["quiz"])
        lab_avg  = _avg_pct(buckets["lab"])
        att_pct  = _avg_pct(buckets["attendance"])
        mid_avg  = _avg_pct(buckets["midterm"])
        fin_avg  = _avg_pct(buckets["final"])

        # Overall = total score earned / total marks available (actual distribution).
        # This is correct regardless of how the professor weights each component.
        if graded_count > 0:
            raw_score = sum(c.score      for c in components if c.score is not None)
            raw_max   = sum(c.max_points for c in components if c.score is not None)
            overall   = round(raw_score / raw_max * 100, 1) if raw_max > 0 else 0.0
        else:
            raw_max   = 0
            overall   = 0.0

        # Predict the final exam score from Classroom performance.
        # Regulation: final exam = 40 marks (secret, never in Classroom).
        # Assumption: student performs at the same percentage on the final as on
        # their Classroom work.  Note: predicted_total_pct == overall_pct
        # (see CourseRecord docstring), so overall_pct already represents
        # the full 100-mark predicted grade.
        predicted_final = round(overall * _FINAL_MAX_MARKS / 100.0, 1) if overall > 0 else 0.0

        # ── Coursework-status determination ───────────────────────────────────
        #
        # Classroom is a Coursework Analytics source only — it does NOT contain
        # official final exam results or registrar-confirmed grades.
        # These statuses describe observed Classroom activity, NOT academic outcomes.
        #
        #   "completed_coursework"   — graded data exists; does NOT mean academically passed
        #   "coursework_in_progress" — active enrollment
        #   "insufficient_data"      — no or minimal graded components
        if in_progress:
            pass_status = "coursework_in_progress"
        elif graded_count == 0:
            pass_status = "insufficient_data"
        else:
            pass_status = "completed_coursework"

        coursework_complete = (pass_status == "completed_coursework")
        in_progress_flag    = (pass_status == "coursework_in_progress")

        # ── Performance category (Classroom activity only) ────────────────────
        if pass_status == "coursework_in_progress" and graded_count == 0:
            perf_cat = "in_progress"
        elif pass_status == "insufficient_data":
            perf_cat = "no_data"
        elif overall >= 85:
            perf_cat = "excellent"
        elif overall >= 70:
            perf_cat = "strong"
        elif overall >= 55:
            perf_cat = "average"
        elif overall >= 40:
            perf_cat = "weak"
        else:
            perf_cat = "at_risk"

        return CourseRecord(
            classroom_id          = course_id,
            course_name           = name,
            course_section        = section,
            creation_time         = creation_time,
            term_label            = term_lbl,
            assignments_avg       = asgn_avg,
            quizzes_avg           = quiz_avg,
            labs_avg              = lab_avg,
            midterm_score         = mid_avg,
            final_score           = fin_avg,
            attendance_pct        = att_pct,
            predicted_final_score = predicted_final,
            overall_pct           = overall,
            pass_status           = pass_status,
            coursework_complete   = coursework_complete,
            in_progress           = in_progress_flag,
            performance_category  = perf_cat,
            components            = components,
        )


# ── OAuth helpers (used by analytics_page.py) ────────────────────────────────

def build_oauth_flow(
    redirect_uri: str,
    client_id: str = "",
    client_secret: str = "",
):
    """
    Build a google-auth-oauthlib Flow.
    Credentials are resolved in order: explicit params → env vars.
    Returns None if credentials are not available.
    """
    import os
    if not _GOOGLE_AVAILABLE:
        return None
    try:
        from google_auth_oauthlib.flow import Flow
    except ImportError:
        return None

    cid  = client_id  or os.getenv("GOOGLE_CLIENT_ID",  "")
    csec = client_secret or os.getenv("GOOGLE_CLIENT_SECRET", "")
    if not cid or not csec:
        return None

    client_config = {
        "web": {
            "client_id":     cid,
            "client_secret": csec,
            "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            "token_uri":     "https://oauth2.googleapis.com/token",
            "redirect_uris": [redirect_uri],
        }
    }
    flow = Flow.from_client_config(
        client_config,
        scopes       = CLASSROOM_SCOPES,
        redirect_uri = redirect_uri,
    )
    return flow


def exchange_code_for_token(
    code: str,
    redirect_uri: str,
    state: str = "",
    client_id: str = "",
    client_secret: str = "",
) -> tuple[Optional[str], str]:
    """
    Exchange an OAuth authorization code for an access token.
    Returns (access_token, error_message). access_token is None on failure.

    state: the OAuth state value from the original authorization URL.
           Must be passed so the recreated Flow can verify it correctly.
    """
    import os
    # oauthlib refuses http:// by default; allow it for localhost dev
    os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

    if not _GOOGLE_AVAILABLE:
        return None, "Google API packages not installed."

    flow = build_oauth_flow(redirect_uri, client_id=client_id, client_secret=client_secret)
    if flow is None:
        return None, "OAuth not configured — check GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."

    # Restore the original state so oauthlib does not raise a state mismatch error.
    # The Flow object is recreated on every Streamlit render; the state is preserved
    # via session state and injected here before fetch_token is called.
    if state:
        flow.oauth2session._state = state

    try:
        flow.fetch_token(code=code)
        return flow.credentials.token, ""
    except Exception as exc:
        return None, str(exc)


# ── Pure utility functions ─────────────────────────────────────────────────────

def _extract_code_from_section(section: str) -> Optional[str]:
    """
    Extract an official course code from a Classroom section identifier.

    Handles formats like:
        "CSAI 100-LCTR-02"  → "CSAI100"
        "MATH 103-LCTR-02"  → "MATH103"
        "DSAI104-LCTR-01"   → "DSAI104"
    Returns None if the section string doesn't start with a recognised code pattern.
    """
    if not section:
        return None
    # Allow optional space or dash between prefix and number ("CSAI-490-LCTR-01" → "CSAI490")
    m = re.match(r'^([A-Za-z]{2,8})[\s-]*(\d{3,4})', section.strip())
    return (m.group(1).upper() + m.group(2)) if m else None


def _infer_term_label(section: str, creation_time: str = "") -> str:
    """
    Derive a normalised academic term label from available Classroom metadata.

    Priority:
      1. Explicit term pattern in section  (e.g. "Fall 2022", "Spring 2023")
      2. Derive from creationTime ISO 8601  (Aug–Dec → Fall, Jan–May → Spring, Jun–Jul → Summer)
      3. Return original section as a fallback (deduplication still works within same section)
    """
    # 1. Explicit term pattern in section string
    m = re.search(r'\b(Fall|Spring|Summer|Winter)\s+(\d{4})\b', section, re.IGNORECASE)
    if m:
        return m.group(0).title()

    # 2. Derive from creationTime  e.g. "2022-09-15T10:30:00.000Z"
    if creation_time:
        dt = re.match(r'(\d{4})-(\d{2})-', creation_time)
        if dt:
            year  = int(dt.group(1))
            month = int(dt.group(2))
            if 8 <= month <= 12:
                return f"Fall {year}"
            elif 1 <= month <= 5:
                return f"Spring {year}"
            else:
                return f"Summer {year}"

    # 3. No term information available
    return section or "Unknown"


def _detect_component_type(title: str) -> str:
    """Map a coursework title to the closest component bucket."""
    lower = title.lower()
    # Ordered by specificity: attendance before lab, midterm before final
    for comp_type in ("attendance", "midterm", "final", "lab", "quiz", "assignment"):
        if any(kw in lower for kw in _COMP_KW[comp_type]):
            return comp_type
    return "assignment"


def _pct_to_grade(pct: float) -> tuple[str, float]:
    """Convert percentage to Zewail letter grade and GPA points."""
    for min_pct, letter, pts in _GPA_SCALE:
        if pct >= min_pct:
            return letter, pts
    return "F", 0.0


def get_coursework_status(record: "CourseRecord") -> str:
    """
    Return the coursework status of a CourseRecord.

    Handles session-state objects cached before the schema migration by deriving
    the new status from legacy fields when present.

    Returns one of:
      "completed_coursework"   — graded data available (NOT an academic outcome)
      "coursework_in_progress" — active enrollment
      "insufficient_data"      — no or minimal graded data
    """
    ps = getattr(record, "pass_status", None)
    # Already on new schema
    if ps in ("completed_coursework", "coursework_in_progress", "insufficient_data"):
        return ps
    # Migrate from old schema values
    if ps in ("passed", "failed", "pending_final"):
        return "completed_coursework"
    if ps in ("in_progress",):
        return "coursework_in_progress"
    if ps in ("ungraded", "unknown") or not ps:
        return "insufficient_data"
    # Derive from legacy booleans when pass_status is absent
    if getattr(record, "in_progress", False):
        return "coursework_in_progress"
    if getattr(record, "overall_pct", 0) > 0:
        return "completed_coursework"
    return "insufficient_data"


# Backward-compatibility alias — prefer get_coursework_status in new code.
get_pass_status = get_coursework_status


def _section_sort_key(section: str) -> int:
    """Extract a numeric sort key from a section/semester label."""
    m = re.search(r'\d+', section)
    return int(m.group()) if m else 999
