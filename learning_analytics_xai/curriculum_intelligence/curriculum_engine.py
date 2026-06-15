"""
Curriculum Intelligence Layer — Zewail City Academic Advisor

Bridges official curriculum JSON (extracted from PDF handbooks) with the
Learning Analytics & XAI pipeline.  All other analytics modules remain
unchanged; this layer adds curriculum-aware context on top.

Public API
----------
get_engine()                          → CurriculumEngine singleton
engine.match_course_code(text)        → official course code or None
engine.get_prerequisites(code)        → list of prerequisite codes (no self-refs)
engine.get_dependents(code)           → courses that require `code`
engine.compute_features(...)          → dict of curriculum-aware feature values
engine.get_blocked_courses(...)       → list of blocked-course dicts
engine.get_graduation_status(...)     → GraduationStatus dataclass
engine.get_academic_standing(gpa)     → standing string
engine.generate_curriculum_recs(...)  → list of curriculum Recommendation dicts
engine.get_curriculum_narratives(...) → list of plain-English XAI narratives
engine.simulate_course_scenario(...)  → CurriculumScenarioResult dataclass
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).parent
_DATA = _HERE.parent / "data"


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class GraduationStatus:
    programme:             str
    total_required:        int
    credits_passed:        int
    credits_remaining:     int
    progress_ratio:        float   # 0–1
    expected_progress:     float   # 0–1 based on current semester
    delay_semesters:       float   # positive = behind
    on_track:              bool
    graduation_gpa_ok:     bool
    eligible_for_honours:  bool
    message:               str


@dataclass
class CurriculumScenarioResult:
    scenario_name:         str
    course_code:           str
    course_title:          str
    outcome:               str   # "pass" | "fail" | "retake" | "postpone"
    delta_credits:         int
    new_blocked:           list[str]
    unblocked:             list[str]
    new_graduation_delay:  float
    new_graduation_ok:     bool
    curriculum_message:    str


# ── Engine ─────────────────────────────────────────────────────────────────────

class CurriculumEngine:
    """
    Loads all curriculum JSON files once (lazy, cached) and provides
    pure-function analysis methods used by every analytics module.
    """

    def __init__(self):
        self._catalog:       list[dict] = []      # course_catalog.json
        self._degree_reqs:   dict       = {}      # degree_requirements.json
        self._prereq_graph:  list[dict] = []      # prerequisites_graph.json
        self._regulations:   dict       = {}      # academic_regulations.json
        self._gpa_rules:     dict       = {}      # gpa_rules.json
        self._loaded = False

        # Derived look-ups built once after loading
        self._by_code:     dict[str, dict]        = {}  # code → course dict
        self._prereqs_of:  dict[str, list[str]]   = {}  # code → [prereq codes]
        self._dependents:  dict[str, list[str]]   = {}  # code → [dependent codes]
        self._prog_courses: dict[str, list[str]]  = {}  # programme → [codes]

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if self._loaded:
            return
        try:
            self._catalog = json.loads((_DATA / "course_catalog.json").read_text("utf-8"))
        except Exception:
            self._catalog = []
        try:
            self._degree_reqs = json.loads((_DATA / "degree_requirements.json").read_text("utf-8"))
        except Exception:
            self._degree_reqs = {}
        try:
            self._prereq_graph = json.loads((_DATA / "prerequisites_graph.json").read_text("utf-8"))
        except Exception:
            self._prereq_graph = []
        try:
            self._regulations = json.loads((_DATA / "academic_regulations.json").read_text("utf-8"))
        except Exception:
            self._regulations = {}
        try:
            self._gpa_rules = json.loads((_DATA / "gpa_rules.json").read_text("utf-8"))
        except Exception:
            self._gpa_rules = {"probation_threshold": 1.7, "honours_threshold": 3.5,
                                "graduation_minimum": 2.0, "dismissal_threshold": 1.0}

        self._build_indexes()
        self._loaded = True

    def _build_indexes(self):
        # Course lookup by code
        for c in self._catalog:
            code = c.get("code", "").strip().upper()
            if code:
                self._by_code[code] = c

        # Prerequisite map: exclude self-references
        for edge in self._prereq_graph:
            course  = edge.get("course",   "").strip().upper()
            prereq  = edge.get("requires", "").strip().upper()
            if course and prereq and course != prereq:
                self._prereqs_of.setdefault(course, [])
                if prereq not in self._prereqs_of[course]:
                    self._prereqs_of[course].append(prereq)
                # reverse index: dependents
                self._dependents.setdefault(prereq, [])
                if course not in self._dependents[prereq]:
                    self._dependents[prereq].append(course)

        # Programme → course codes (by code prefix matching known programmes)
        prog_prefixes = {
            "CSAI": ["CSAI"],
            "DSAI": ["DSAI", "CSAI"],
            "SWE":  ["CSAI"],
            "MECH": ["MECH", "ENGR"],
            "EEE":  ["EEE", "ENGR"],
            "CIV":  ["CIV", "ENGR"],
            "MATH": ["MATH"],
            "PHYS": ["PHYS"],
            "CHEM": ["CHEM"],
            "BUS":  ["BUS"],
            "FIN":  ["FIN", "BUS"],
        }
        for prog, prefixes in prog_prefixes.items():
            codes = [c["code"] for c in self._catalog
                     if any(c["code"].startswith(p) for p in prefixes)]
            self._prog_courses[prog] = codes

    # ── Course matching ────────────────────────────────────────────────────────

    def match_course_code(self, user_text: str) -> Optional[str]:
        """
        Try to extract / match an official course code from free-form text.

        Priority:
          1. Exact match (normalised, spaces removed)
          2. Regex extraction of code pattern from text
          3. Title substring match
        Returns the official code string or None.
        """
        self._load()
        if not user_text:
            return None

        # Normalise: remove extra spaces inside codes like "CSAI 100" → "CSAI100"
        normalised = re.sub(r'([A-Za-z]+)\s+(\d+)', r'\1\2', user_text).upper().strip()

        # 1. Exact code lookup
        if normalised in self._by_code:
            return normalised

        # 2. Extract code pattern anywhere in text
        pattern = r'\b([A-Z]{2,8})\s*(\d{3,4})\b'
        for m in re.finditer(pattern, user_text.upper()):
            candidate = m.group(1) + m.group(2)
            if candidate in self._by_code:
                return candidate

        # 3. Title substring match (case-insensitive)
        lower = user_text.lower()
        best_match = None
        best_len   = 0
        for code, info in self._by_code.items():
            title = info.get("title", "").lower()
            if title and title in lower and len(title) > best_len:
                best_match = code
                best_len   = len(title)
        return best_match

    def get_course(self, code: str) -> Optional[dict]:
        self._load()
        return self._by_code.get(code.strip().upper())

    def get_prerequisites(self, code: str) -> list[str]:
        """Real prerequisites (self-references excluded)."""
        self._load()
        return list(self._prereqs_of.get(code.strip().upper(), []))

    def get_dependents(self, code: str) -> list[str]:
        """Courses that directly require `code` as a prerequisite."""
        self._load()
        return list(self._dependents.get(code.strip().upper(), []))

    def get_all_dependents(self, code: str) -> list[str]:
        """All courses transitively blocked if `code` is failed."""
        self._load()
        visited: set[str] = set()
        stack = [code.strip().upper()]
        while stack:
            cur = stack.pop()
            for dep in self._dependents.get(cur, []):
                if dep not in visited:
                    visited.add(dep)
                    stack.append(dep)
        return sorted(visited)

    def get_programme_courses(self, programme: str) -> list[str]:
        """All course codes associated with a programme."""
        self._load()
        return list(self._prog_courses.get(programme.upper(), []))

    def get_degree_requirements(self, programme: str) -> dict:
        self._load()
        return dict(self._degree_reqs.get(programme.upper(), {
            "total_credits": 132, "core_credits": 86,
            "elective_credits": 26, "general_ed_credits": 20,
            "minimum_gpa": 2.0,
        }))

    # ── Academic standing ──────────────────────────────────────────────────────

    def get_academic_standing(self, gpa: float) -> str:
        """
        Returns standing string per Zewail academic regulations.
        Source: gpa_rules.json + academic_regulations.json
        """
        self._load()
        rules = self._gpa_rules
        dismissal  = rules.get("dismissal_threshold",  1.0)
        probation  = rules.get("probation_threshold",  1.7)
        good       = rules.get("good_standing_minimum", 2.0)
        honours    = rules.get("honours_threshold",     3.5)
        high_hon   = rules.get("high_honours_threshold", 3.8)

        if gpa < dismissal:
            return "Academic Dismissal Risk"
        if gpa < probation:
            return "Academic Probation"
        if gpa < good:
            return "Warning — Below Good Standing"
        if gpa >= high_hon:
            return "High Honours"
        if gpa >= honours:
            return "Dean's Honours"
        return "Good Standing"

    def percent_to_grade(self, pct: float) -> tuple[str, float]:
        """Convert percentage to Zewail letter grade and GPA points."""
        self._load()
        for entry in self._gpa_rules.get("scale", []):
            if pct >= entry["min_percent"]:
                return entry["grade"], entry["gpa_points"]
        return "F", 0.0

    # ── Curriculum features ────────────────────────────────────────────────────

    def compute_features(
        self,
        programme:          str,
        passed_codes:       list[str],
        failed_codes:       list[str],
        credits_passed:     float,
        semester:           int,
        total_credits_reg:  float = 45.0,
    ) -> dict:
        """
        Compute all curriculum-aware features.  These EXTEND the existing 21
        ML features but are NOT fed to the GPA/risk models (which stay frozen).
        They are used for recommendations, XAI narratives, and what-if analysis.

        Returns
        -------
        dict with keys matching the new feature names defined in the spec.
        All values have sensible defaults if data is unavailable.
        """
        self._load()
        programme = (programme or "CSAI").upper()
        passed_set = set(c.upper() for c in passed_codes)
        failed_set = set(c.upper() for c in failed_codes)

        req = self.get_degree_requirements(programme)
        total_req     = req.get("total_credits",    132)
        core_req      = req.get("core_credits",      86)
        elective_req  = req.get("elective_credits",  26)

        prog_codes = set(self.get_programme_courses(programme))

        # ── 1. Graduation progress ratio ──────────────────────────────────────
        grad_progress = min(credits_passed / max(total_req, 1), 1.0)

        # ── 2. Expected progress based on semester ────────────────────────────
        # Zewail typical: 8 semesters for a full degree
        expected_progress = min(semester / 8.0, 1.0)
        delay_ratio = expected_progress - grad_progress   # + means behind
        delay_semesters = round(delay_ratio * 8, 2)

        # ── 3. Programme-course completion ───────────────────────────────────
        if prog_codes:
            prog_passed  = passed_set & prog_codes
            prog_failed  = failed_set & prog_codes
            core_ratio   = len(prog_passed) / max(len(prog_codes), 1)
        else:
            prog_passed  = set()
            prog_failed  = set()
            core_ratio   = 0.5   # unknown

        # ── 4. Prerequisite completion ratio ─────────────────────────────────
        # For each passed course, were all its prerequisites met?
        total_prereq_checks = 0
        met_prereq_checks   = 0
        for code in passed_set:
            prereqs = self.get_prerequisites(code)
            for p in prereqs:
                total_prereq_checks += 1
                if p in passed_set:
                    met_prereq_checks += 1

        prereq_ratio = (met_prereq_checks / max(total_prereq_checks, 1))

        # ── 5. Blocked courses (due to failed prerequisites) ──────────────────
        blocked_codes: set[str] = set()
        for code in failed_set:
            for dep in self.get_dependents(code):
                if dep not in passed_set:
                    blocked_codes.add(dep)

        blocked_credits  = sum(
            self._by_code.get(c, {}).get("credits", 3) for c in blocked_codes
        )
        blocked_prog_courses = len(blocked_codes & prog_codes)

        # ── 6. Failed prerequisite flag ───────────────────────────────────────
        failed_prereq_flag = 1 if blocked_codes else 0

        # ── 7. Prerequisite chain depth (max depth of blocked chains) ─────────
        max_chain_depth = 0
        for code in failed_set:
            all_blocked = self.get_all_dependents(code)
            # Rough chain depth: BFS levels
            depth = self._chain_depth(code)
            max_chain_depth = max(max_chain_depth, depth)

        # ── 8. Curriculum alignment score ────────────────────────────────────
        # How many of the student's current courses belong to their programme?
        current_codes = passed_set | failed_set
        if current_codes:
            alignment = len(current_codes & prog_codes) / len(current_codes)
        else:
            alignment = 0.5

        # ── 9. Graduation readiness ───────────────────────────────────────────
        grad_readiness = (
            0.50 * grad_progress
            + 0.20 * core_ratio
            + 0.15 * prereq_ratio
            + 0.15 * (1 - min(delay_ratio, 1.0))
        )

        return {
            "graduation_progress_ratio":    round(grad_progress,       4),
            "expected_progress_ratio":      round(expected_progress,    4),
            "graduation_delay_semesters":   round(delay_semesters,      2),
            "core_course_completion_ratio": round(core_ratio,           4),
            "prerequisite_completion_ratio": round(prereq_ratio,        4),
            "blocked_credit_hours":         int(blocked_credits),
            "blocked_core_courses":         int(blocked_prog_courses),
            "failed_prerequisite_flag":     int(failed_prereq_flag),
            "prerequisite_chain_depth":     int(max_chain_depth),
            "curriculum_alignment_score":   round(alignment,            4),
            "graduation_readiness_score":   round(grad_readiness,       4),
            # raw lists for recommendation / narrative use
            "_blocked_codes":      sorted(blocked_codes),
            "_prog_passed":        sorted(prog_passed),
            "_prog_failed":        sorted(prog_failed),
            "_programme":          programme,
            "_total_req":          total_req,
            "_core_req":           core_req,
            "_delay_semesters":    delay_semesters,
        }

    def _chain_depth(self, code: str, visited: Optional[set] = None, depth: int = 0) -> int:
        """Compute the longest prerequisite chain depth rooted at `code`."""
        if visited is None:
            visited = set()
        if code in visited:
            return depth
        visited.add(code)
        deps = self._dependents.get(code, [])
        if not deps:
            return depth
        return max(self._chain_depth(d, visited, depth + 1) for d in deps)

    # ── Graduation status ──────────────────────────────────────────────────────

    def get_graduation_status(
        self,
        programme:      str,
        credits_passed: float,
        semester:       int,
        predicted_gpa:  float,
    ) -> GraduationStatus:
        self._load()
        req = self.get_degree_requirements(programme)
        total_req  = req.get("total_credits", 132)
        min_gpa    = req.get("minimum_gpa",   2.0)
        honours_t  = self._gpa_rules.get("honours_threshold",     3.5)

        remaining        = max(total_req - credits_passed, 0)
        progress         = min(credits_passed / max(total_req, 1), 1.0)
        expected         = min(semester / 8.0, 1.0)
        delay_semesters  = round((expected - progress) * 8, 2)
        on_track         = delay_semesters <= 0.5
        grad_gpa_ok      = predicted_gpa >= min_gpa
        eligible_honours = predicted_gpa >= honours_t

        if credits_passed >= total_req and grad_gpa_ok:
            msg = f"Eligible for graduation from {programme} ({credits_passed}/{total_req} credits, GPA {predicted_gpa:.2f})."
        elif not grad_gpa_ok:
            msg = (f"GPA {predicted_gpa:.2f} is below the {programme} graduation minimum of {min_gpa:.1f}. "
                   f"{remaining} credits remaining.")
        elif on_track:
            msg = (f"On track for graduation. {remaining} credits remaining "
                   f"({progress*100:.0f}% complete), expected on schedule.")
        else:
            msg = (f"Approximately {delay_semesters:.1f} semester(s) behind the expected {programme} curriculum pace. "
                   f"{remaining} credits remaining to graduation.")

        return GraduationStatus(
            programme           = programme,
            total_required      = total_req,
            credits_passed      = int(credits_passed),
            credits_remaining   = int(remaining),
            progress_ratio      = round(progress, 4),
            expected_progress   = round(expected, 4),
            delay_semesters     = delay_semesters,
            on_track            = on_track,
            graduation_gpa_ok   = grad_gpa_ok,
            eligible_for_honours= eligible_honours,
            message             = msg,
        )

    # ── Blocked course analysis ────────────────────────────────────────────────

    def get_blocked_courses(
        self,
        failed_codes: list[str],
        programme:    str,
    ) -> list[dict]:
        """
        For each failed course, find which future courses are blocked.
        Returns a list of dicts with course details and impact.
        """
        self._load()
        programme  = (programme or "CSAI").upper()
        prog_codes = set(self.get_programme_courses(programme))
        failed_set = set(c.upper() for c in failed_codes)
        result     = []

        for failed_code in failed_set:
            info      = self._by_code.get(failed_code, {})
            direct    = self.get_dependents(failed_code)
            all_dep   = self.get_all_dependents(failed_code)
            core_dep  = [c for c in all_dep if c in prog_codes]
            dep_creds = sum(self._by_code.get(c, {}).get("credits", 3) for c in all_dep)

            if direct or all_dep:
                result.append({
                    "failed_course":      failed_code,
                    "failed_title":       info.get("title", failed_code),
                    "failed_credits":     info.get("credits", 3),
                    "direct_blocked":     direct,
                    "all_blocked":        all_dep,
                    "core_courses_blocked": core_dep,
                    "blocked_credit_hours": dep_creds,
                    "chain_depth":        self._chain_depth(failed_code),
                })
        result.sort(key=lambda x: x["blocked_credit_hours"], reverse=True)
        return result

    # ── Curriculum-aware recommendations ──────────────────────────────────────

    def generate_curriculum_recs(
        self,
        programme:          str,
        curriculum_features: dict,
        passed_codes:       list[str],
        failed_codes:       list[str],
        predicted_gpa:      float,
        semester:           int,
    ) -> list[dict]:
        """
        Generate curriculum-specific recommendations as plain dicts matching
        the Recommendation dataclass schema (priority, category, icon, title,
        detail, impact, shap_driver).
        """
        self._load()
        programme = (programme or "CSAI").upper()
        recs: list[dict] = []

        gpa_rules   = self._gpa_rules
        grad_min    = gpa_rules.get("graduation_minimum",  2.0)
        prob_thresh = gpa_rules.get("probation_threshold", 1.7)
        hon_thresh  = gpa_rules.get("honours_threshold",   3.5)
        req         = self.get_degree_requirements(programme)

        delay       = curriculum_features.get("graduation_delay_semesters", 0)
        blocked_cr  = curriculum_features.get("blocked_credit_hours",       0)
        blocked_core= curriculum_features.get("blocked_core_courses",       0)
        failed_flag = curriculum_features.get("failed_prerequisite_flag",   0)
        core_ratio  = curriculum_features.get("core_course_completion_ratio", 0.5)
        grad_prog   = curriculum_features.get("graduation_progress_ratio",  0.5)
        prereq_rat  = curriculum_features.get("prerequisite_completion_ratio", 1.0)
        blocked_list= curriculum_features.get("_blocked_codes",             [])
        prog_failed = curriculum_features.get("_prog_failed",               [])

        # ── 1. Blocked prerequisite chains ────────────────────────────────────
        if failed_flag and blocked_cr > 0:
            for entry in self.get_blocked_courses(failed_codes, programme)[:2]:
                fc   = entry["failed_course"]
                ft   = entry["failed_title"]
                deps = entry["direct_blocked"][:3]
                dep_titles = [self._by_code.get(d, {}).get("title", d) for d in deps]
                dep_str = ", ".join(dep_titles) if dep_titles else "several required courses"
                recs.append({
                    "priority": 1,
                    "category": "Curriculum",
                    "icon": "🔗",
                    "title": f"Retake {fc} — Prerequisite Bottleneck",
                    "detail": (
                        f"{fc} ({ft}) is a prerequisite for {dep_str}. "
                        f"Failing it blocks {entry['blocked_credit_hours']} credit hours "
                        f"and {len(entry['core_courses_blocked'])} programme core courses. "
                        f"Retaking and passing {fc} is the single highest-leverage action you can take."
                    ),
                    "impact": (
                        f"Unblocks {len(entry['all_blocked'])} course(s) "
                        f"worth {entry['blocked_credit_hours']} credits"
                    ),
                    "shap_driver": "Prerequisite Chain",
                })

        # ── 2. Graduation pace ────────────────────────────────────────────────
        if delay > 1.0:
            recs.append({
                "priority": 2,
                "category": "Curriculum",
                "icon": "📅",
                "title": f"Graduation Pace: ~{delay:.1f} Semester(s) Behind",
                "detail": (
                    f"At current pace you are approximately {delay:.1f} semester(s) behind the "
                    f"expected {programme} curriculum progression. "
                    f"The degree requires {req.get('total_credits', 132)} credits; "
                    f"you have completed {curriculum_features.get('graduation_progress_ratio', 0)*100:.0f}% so far. "
                    "Consider increasing your credit load or attending summer sessions "
                    "to stay on schedule for graduation."
                ),
                "impact": "Each semester of delay costs one additional year of tuition and delays career start",
                "shap_driver": "Curriculum Delay",
            })
        elif delay < -0.5:
            recs.append({
                "priority": 4,
                "category": "Curriculum",
                "icon": "🏎️",
                "title": "Ahead of Curriculum Schedule",
                "detail": (
                    f"You are approximately {abs(delay):.1f} semester(s) ahead of the expected "
                    f"{programme} curriculum pace. "
                    "Consider applying for research assistantships, advanced electives, or "
                    "preparing for graduate school applications."
                ),
                "impact": "Strong candidate for early graduation or double-programme track",
                "shap_driver": "Curriculum Delay",
            })

        # ── 3. Core course completion ─────────────────────────────────────────
        if core_ratio < 0.5 and semester >= 3:
            recs.append({
                "priority": 2,
                "category": "Curriculum",
                "icon": "📋",
                "title": f"Prioritise {programme} Core Requirements",
                "detail": (
                    f"Only {core_ratio*100:.0f}% of known {programme} core courses have been completed. "
                    f"Core courses are foundational for upper-level study and must be completed "
                    f"before advanced electives. "
                    "Ensure your next semester plan prioritises remaining core requirements."
                ),
                "impact": "Core completion directly enables advanced course access",
                "shap_driver": "Core Course Completion",
            })

        # ── 4. Graduation GPA ──────────────────────────────────────────────────
        if predicted_gpa < grad_min:
            recs.append({
                "priority": 1,
                "category": "Curriculum",
                "icon": "🎓",
                "title": "Below Graduation GPA Requirement",
                "detail": (
                    f"Your predicted GPA ({predicted_gpa:.2f}) is below the {programme} "
                    f"graduation minimum of {grad_min:.1f} required by Zewail academic regulations. "
                    "You must raise your cumulative GPA before applying for graduation. "
                    "Retaking failed courses and improving in current semester courses are "
                    "the most direct path to meeting this requirement."
                ),
                "impact": f"GPA must reach {grad_min:.1f} before graduation clearance can be granted",
                "shap_driver": "Graduation GPA",
            })

        # ── 5. Academic probation per official regulations ────────────────────
        if prob_thresh <= predicted_gpa < grad_min:
            recs.append({
                "priority": 2,
                "category": "Curriculum",
                "icon": "⚠️",
                "title": "Warning: Approaching Academic Probation",
                "detail": (
                    f"Your GPA ({predicted_gpa:.2f}) is between the probation threshold "
                    f"({prob_thresh}) and good standing ({grad_min}). "
                    "Per Zewail academic regulations, students on probation may not register "
                    "for more than 12 credit hours per semester and have two consecutive "
                    "warning semesters (WS1 and WS2) to raise their CGPA above 2.0. "
                    "Failure to do so leads to dismissal."
                ),
                "impact": "Probation limits registration to 12 credit hours per semester",
                "shap_driver": "Academic Regulations",
            })

        # ── 6. Honours eligibility ─────────────────────────────────────────────
        if predicted_gpa >= hon_thresh:
            recs.append({
                "priority": 4,
                "category": "Curriculum",
                "icon": "🌟",
                "title": f"Eligible for Dean's Honours — {programme}",
                "detail": (
                    f"Your GPA ({predicted_gpa:.2f}) meets the Dean's Honours threshold "
                    f"({hon_thresh}). Maintain this level to graduate with honours distinction. "
                    "Consider research opportunities, senior projects, or advanced electives "
                    "to strengthen your academic record further."
                ),
                "impact": "Honours distinction noted on transcript and degree certificate",
                "shap_driver": "Academic Excellence",
            })

        # ── 7. Failed programme-core courses ─────────────────────────────────
        if prog_failed:
            titles = [self._by_code.get(c, {}).get("title", c) for c in prog_failed[:3]]
            recs.append({
                "priority": 2,
                "category": "Curriculum",
                "icon": "📚",
                "title": f"Retake Failed {programme} Core Course(s)",
                "detail": (
                    f"You have failed {len(prog_failed)} {programme} programme course(s): "
                    f"{', '.join(titles)}. "
                    "These are required for degree completion and may be prerequisites for "
                    "upper-level courses. Retaking them should be your top priority in the "
                    "next available semester."
                ),
                "impact": "Required for degree completion and prerequisite chains",
                "shap_driver": "Programme Core Courses",
            })

        # Sort and cap
        recs.sort(key=lambda r: r["priority"])
        return recs[:5]

    # ── XAI narrative generation ───────────────────────────────────────────────

    def get_curriculum_narratives(
        self,
        curriculum_features: dict,
        programme:           str,
        passed_codes:        list[str],
        failed_codes:        list[str],
        predicted_gpa:       float,
    ) -> list[str]:
        """
        Generate plain-English XAI narrative sentences that explain the
        curriculum-aware aspects of the student's academic situation.
        These complement the quantitative SHAP waterfall plot.
        """
        self._load()
        programme = (programme or "CSAI").upper()
        narratives: list[str] = []
        req = self.get_degree_requirements(programme)
        total_req  = req.get("total_credits", 132)
        grad_min   = req.get("minimum_gpa",   2.0)

        grad_prog  = curriculum_features.get("graduation_progress_ratio",   0.0)
        delay      = curriculum_features.get("graduation_delay_semesters",  0.0)
        core_ratio = curriculum_features.get("core_course_completion_ratio", 0.0)
        blocked_cr = curriculum_features.get("blocked_credit_hours",         0)
        blocked_core = curriculum_features.get("blocked_core_courses",       0)
        prereq_rat = curriculum_features.get("prerequisite_completion_ratio", 1.0)
        blocked_list = curriculum_features.get("_blocked_codes",             [])

        # Graduation progress narrative
        cr_done = int(grad_prog * total_req)
        narratives.append(
            f"Graduation Progress: {cr_done}/{total_req} credits completed "
            f"({grad_prog*100:.0f}%) toward the {programme} degree requirement."
        )

        # Delay narrative
        if delay > 0.5:
            narratives.append(
                f"Curriculum Pace: Approximately {delay:.1f} semester(s) behind the expected "
                f"{programme} programme timeline."
            )
        elif delay < -0.5:
            narratives.append(
                f"Curriculum Pace: {abs(delay):.1f} semester(s) ahead of the expected "
                f"programme timeline — excellent progression."
            )

        # Core course narrative
        narratives.append(
            f"Core Course Completion: {core_ratio*100:.0f}% of known {programme} "
            f"programme courses have been completed."
        )

        # Blocked courses narrative
        if blocked_cr > 0:
            blocked_examples = blocked_list[:3]
            be_titles = [self._by_code.get(c, {}).get("title", c) for c in blocked_examples]
            suffix = f" (including {', '.join(be_titles)})" if be_titles else ""
            narratives.append(
                f"Blocked Academic Path: Failed prerequisite(s) are blocking access to "
                f"{blocked_cr} credit hours of future courses{suffix}. "
                f"This includes {blocked_core} programme core course(s)."
            )

        # Failed courses are prerequisites
        if failed_codes:
            for fc in failed_codes[:2]:
                deps = self.get_dependents(fc)
                if deps:
                    dep_titles = [self._by_code.get(d, {}).get("title", d) for d in deps[:3]]
                    fc_title = self._by_code.get(fc, {}).get("title", fc)
                    narratives.append(
                        f"Prerequisite Risk: {fc} ({fc_title}) is a required prerequisite "
                        f"for {len(deps)} future course(s) including "
                        f"{', '.join(dep_titles[:2])}."
                    )

        # Graduation GPA narrative
        standing = self.get_academic_standing(predicted_gpa)
        narratives.append(
            f"Academic Standing: {standing} (Predicted GPA {predicted_gpa:.2f}, "
            f"graduation minimum {grad_min:.1f})."
        )

        return narratives

    # ── What-if curriculum scenarios ───────────────────────────────────────────

    def simulate_course_scenario(
        self,
        programme:     str,
        passed_codes:  list[str],
        failed_codes:  list[str],
        credits_passed: float,
        semester:      int,
        course_code:   str,
        outcome:       str,            # "pass" | "fail" | "retake" | "postpone"
        predicted_gpa: float,
    ) -> CurriculumScenarioResult:
        """
        Simulate the curriculum impact of a specific course outcome.
        Does NOT change GPA/risk predictions (those are model-based).
        Returns curriculum-level impact only.
        """
        self._load()
        programme  = (programme or "CSAI").upper()
        course_code = course_code.strip().upper()
        info        = self._by_code.get(course_code, {})
        title       = info.get("title", course_code)
        credits     = info.get("credits", 3)

        passed_set = set(c.upper() for c in passed_codes)
        failed_set = set(c.upper() for c in failed_codes)

        # Apply the scenario
        new_passed = set(passed_set)
        new_failed = set(failed_set)
        delta_cred = 0

        if outcome == "pass" or outcome == "retake":
            new_passed.add(course_code)
            new_failed.discard(course_code)
            delta_cred = credits if course_code not in passed_set else 0
        elif outcome == "fail":
            new_failed.add(course_code)
            new_passed.discard(course_code)
            delta_cred = 0
        elif outcome == "postpone":
            new_passed.discard(course_code)
            new_failed.discard(course_code)
            delta_cred = 0

        # Old and new blocked sets
        old_feat = self.compute_features(programme, list(passed_set),  list(failed_set),  credits_passed, semester)
        new_feat = self.compute_features(programme, list(new_passed), list(new_failed), credits_passed + delta_cred, semester)

        old_blocked = set(old_feat.get("_blocked_codes", []))
        new_blocked = set(new_feat.get("_blocked_codes", []))

        newly_blocked  = sorted(new_blocked - old_blocked)
        newly_unblocked = sorted(old_blocked - new_blocked)

        old_delay = old_feat.get("graduation_delay_semesters", 0)
        new_delay = new_feat.get("graduation_delay_semesters", 0)
        new_grad_ok = predicted_gpa >= self.get_degree_requirements(programme).get("minimum_gpa", 2.0)

        # Build message
        msgs = [f"Scenario: {outcome.capitalize()} {course_code} ({title}, {credits} cr)"]
        if newly_unblocked:
            ul_titles = [self._by_code.get(c, {}).get("title", c) for c in newly_unblocked[:3]]
            msgs.append(f"Unblocks: {', '.join(ul_titles)}")
        if newly_blocked:
            bl_titles = [self._by_code.get(c, {}).get("title", c) for c in newly_blocked[:3]]
            msgs.append(f"New blocks: {', '.join(bl_titles)}")
        if delta_cred > 0:
            msgs.append(f"+{delta_cred} credits toward graduation")
        if new_delay != old_delay:
            msgs.append(f"Graduation delay: {old_delay:.1f} → {new_delay:.1f} semesters")

        return CurriculumScenarioResult(
            scenario_name       = f"{outcome.capitalize()} {course_code}",
            course_code         = course_code,
            course_title        = title,
            outcome             = outcome,
            delta_credits       = delta_cred,
            new_blocked         = newly_blocked,
            unblocked           = newly_unblocked,
            new_graduation_delay = new_delay,
            new_graduation_ok   = new_grad_ok,
            curriculum_message  = " | ".join(msgs),
        )


    # ── Curriculum SHAP-equivalent values ─────────────────────────────────────

    def get_curriculum_shap_values(
        self,
        curriculum_features: dict,
        predicted_gpa: float,
    ) -> dict:
        """
        Compute rule-based curriculum impact scores in GPA units, analogous to
        SHAP values for the ML model.  These represent how much each curriculum
        factor contributes to or detracts from the student's academic outcome,
        grounded in Zewail degree requirements and academic regulations.

        Returns a dict of {factor_label: impact_value} where positive values
        help GPA/progress and negative values hurt it.
        """
        self._load()
        scores: dict[str, float] = {}
        rules = self._gpa_rules

        grad_prog  = curriculum_features.get("graduation_progress_ratio",       0.5)
        exp_prog   = curriculum_features.get("expected_progress_ratio",          0.5)
        delay      = curriculum_features.get("graduation_delay_semesters",       0.0)
        core_ratio = curriculum_features.get("core_course_completion_ratio",     0.5)
        prereq_rat = curriculum_features.get("prerequisite_completion_ratio",    1.0)
        blocked_cr = curriculum_features.get("blocked_credit_hours",             0)
        blocked_core = curriculum_features.get("blocked_core_courses",           0)
        failed_flag  = curriculum_features.get("failed_prerequisite_flag",       0)
        chain_depth  = curriculum_features.get("prerequisite_chain_depth",       0)
        alignment    = curriculum_features.get("curriculum_alignment_score",     0.5)
        readiness    = curriculum_features.get("graduation_readiness_score",     0.5)

        prob_thresh = rules.get("probation_threshold",  1.7)
        grad_min    = rules.get("graduation_minimum",   2.0)
        hon_thresh  = rules.get("honours_threshold",    3.5)

        # Graduation pace — being ahead is positive, behind is negative
        # Each semester behind ≈ −0.10 GPA impact (from academic literature)
        scores["Graduation Pace"] = round(-0.10 * max(delay, 0) + 0.05 * abs(min(delay, 0)), 4)

        # Core course completion — foundational for GPA trajectory
        # Completing >70% of core = positive; <40% late in programme = strongly negative
        core_impact = (core_ratio - 0.55) * 0.35
        scores["Core Course Completion"] = round(core_impact, 4)

        # Prerequisite chain integrity — broken chains compound risk
        prereq_impact = (prereq_rat - 0.5) * 0.20
        scores["Prerequisite Integrity"] = round(prereq_impact, 4)

        # Blocked credits — future courses prevented = latent GPA risk
        # Per 3 blocked credits ≈ −0.04 GPA impact
        blocked_impact = -0.04 * min(blocked_cr / 3.0, 6)
        scores["Blocked Prerequisites"] = round(blocked_impact, 4)

        # Prerequisite chain depth — deep chains = compounding future risk
        depth_impact = -0.05 * min(chain_depth, 4)
        scores["Prerequisite Chain Depth"] = round(depth_impact, 4)

        # Curriculum alignment — taking non-programme courses reduces depth
        alignment_impact = (alignment - 0.5) * 0.15
        scores["Curriculum Alignment"] = round(alignment_impact, 4)

        # Graduation readiness — composite driver
        readiness_impact = (readiness - 0.5) * 0.40
        scores["Graduation Readiness"] = round(readiness_impact, 4)

        # Failed prerequisite penalty — any blocked chain = immediate warning
        if failed_flag:
            scores["Failed Prerequisite Penalty"] = round(-0.12 - 0.04 * min(blocked_core, 4), 4)
        else:
            scores["Failed Prerequisite Penalty"] = 0.0

        # GPA-regulation alignment bonus/penalty
        if predicted_gpa >= hon_thresh:
            scores["Academic Standing Bonus"] = round(0.08, 4)
        elif predicted_gpa < prob_thresh:
            scores["Academic Standing Penalty"] = round(-0.15, 4)
        elif predicted_gpa < grad_min:
            scores["Academic Standing Penalty"] = round(-0.08, 4)

        # Remove zero-impact entries for clean display
        scores = {k: v for k, v in scores.items() if v != 0.0}
        return scores

    def get_curriculum_feature_table(
        self,
        curriculum_features: dict,
        programme: str,
    ) -> list[dict]:
        """
        Return curriculum features as a structured list for tabular display.
        Each row: {feature, value, status, interpretation}
        """
        self._load()
        programme = (programme or "CSAI").upper()
        req = self.get_degree_requirements(programme)
        total_req = req.get("total_credits", 132)
        rows = []

        def pct(v): return f"{v*100:.0f}%"
        def ok(v, threshold): return "✅" if v >= threshold else ("⚠️" if v >= threshold * 0.7 else "🔴")

        grad_prog  = curriculum_features.get("graduation_progress_ratio", 0)
        delay      = curriculum_features.get("graduation_delay_semesters", 0)
        core_ratio = curriculum_features.get("core_course_completion_ratio", 0)
        prereq_rat = curriculum_features.get("prerequisite_completion_ratio", 1)
        blocked_cr = curriculum_features.get("blocked_credit_hours", 0)
        blocked_core = curriculum_features.get("blocked_core_courses", 0)
        failed_flag  = curriculum_features.get("failed_prerequisite_flag", 0)
        chain_depth  = curriculum_features.get("prerequisite_chain_depth", 0)
        alignment    = curriculum_features.get("curriculum_alignment_score", 1)
        readiness    = curriculum_features.get("graduation_readiness_score", 0)

        rows.append({"Feature": "Graduation Progress", "Value": pct(grad_prog),
                     "Status": ok(grad_prog, 0.5),
                     "Detail": f"{int(grad_prog * total_req)}/{total_req} credits completed"})
        rows.append({"Feature": "Curriculum Pace", "Value": f"{abs(delay):.1f} sem {'behind' if delay > 0 else 'ahead'}",
                     "Status": "✅" if delay <= 0.5 else ("⚠️" if delay <= 1.5 else "🔴"),
                     "Detail": "On schedule" if delay <= 0.5 else f"~{delay:.1f} semesters behind expected"})
        rows.append({"Feature": "Core Course Completion", "Value": pct(core_ratio),
                     "Status": ok(core_ratio, 0.6),
                     "Detail": f"{pct(core_ratio)} of {programme} core courses completed"})
        rows.append({"Feature": "Prerequisite Integrity", "Value": pct(prereq_rat),
                     "Status": ok(prereq_rat, 0.8),
                     "Detail": "All prerequisites met" if prereq_rat >= 0.95 else f"{pct(prereq_rat)} of prerequisites satisfied"})
        rows.append({"Feature": "Blocked Credits", "Value": f"{blocked_cr} cr",
                     "Status": "✅" if blocked_cr == 0 else ("⚠️" if blocked_cr <= 9 else "🔴"),
                     "Detail": "No blocked courses" if blocked_cr == 0 else f"{blocked_cr} credit hours blocked by failed prerequisites"})
        rows.append({"Feature": "Blocked Core Courses", "Value": str(blocked_core),
                     "Status": "✅" if blocked_core == 0 else "🔴",
                     "Detail": "None" if blocked_core == 0 else f"{blocked_core} core programme courses currently blocked"})
        rows.append({"Feature": "Prerequisite Chain Depth", "Value": str(chain_depth),
                     "Status": "✅" if chain_depth <= 1 else ("⚠️" if chain_depth <= 3 else "🔴"),
                     "Detail": f"Max dependency chain depth: {chain_depth} level(s)"})
        rows.append({"Feature": "Curriculum Alignment", "Value": pct(alignment),
                     "Status": ok(alignment, 0.7),
                     "Detail": f"{pct(alignment)} of taken courses belong to {programme} programme"})
        rows.append({"Feature": "Graduation Readiness", "Value": pct(readiness),
                     "Status": ok(readiness, 0.5),
                     "Detail": "Composite: progress + core + prerequisites + pace"})
        return rows


# ── Singleton ──────────────────────────────────────────────────────────────────

_engine: Optional[CurriculumEngine] = None


def get_engine() -> CurriculumEngine:
    global _engine
    if _engine is None:
        _engine = CurriculumEngine()
    return _engine
