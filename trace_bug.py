#!/usr/bin/env python3
"""Trace execution path for: DSAI semester 1, failed MATH 103"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from phase8_advisor_engine import (
    StudentProfile, update_profile, StudyPlanEngine, CurriculumGraph,
    PlanningEngine, CurriculumProgressionMetrics, _SCHOOL_CODE_PREFIXES,
)

graph    = CurriculumGraph()
plan_eng = StudyPlanEngine()
planner  = PlanningEngine()

# ── Step 1: profile extraction ──────────────────────────────────────────────
msg = "I am DSAI semester 1 and I failed MATH 103"
p = StudentProfile()
p = update_profile(msg, p)
print("=== STEP 1: update_profile() ===")
print(f"  school={p.school!r}  major={p.major!r}  semester={p.semester}")
print(f"  completed_courses: {p.completed_courses}")
print(f"  failed_courses:    {p.failed_courses}")

# ── Step 2: resolve programme ───────────────────────────────────────────────
plan_resolved = plan_eng.resolve_prog(p.school, p.major)
s_key, p_key = plan_resolved
track_key = None
cur_seq = p.semester  # <-- THIS is the planning semester
print("\n=== STEP 2: resolve_prog() ===")
print(f"  plan_resolved: {plan_resolved}   cur_seq: {cur_seq}")

# ── Step 3: infer_presumed_completed ───────────────────────────────────────
print("\n=== STEP 3: infer_presumed_completed(cur_seq=1) ===")
presumed = plan_eng.infer_presumed_completed(p.school, p.major, cur_seq, track_key)
print(f"  Returns: {presumed!r}")
print(f"  (get_planned_codes_before with seq < {cur_seq} -> nothing because sem 1 is the first)")

# ── Step 4: effective_completed ────────────────────────────────────────────
failed_upper   = {c.upper() for c in p.failed_courses}
explicit_upper = {c.upper() for c in p.completed_courses}
effective_completed = list(
    {c.upper() for c in (presumed + p.completed_courses)} - failed_upper
)
print("\n=== STEP 4: effective_completed ===")
print(f"  presumed={presumed}")
print(f"  explicit={list(explicit_upper)}")
print(f"  failed_upper={failed_upper}")
print(f"  effective_completed: {effective_completed}")

# ── Step 5: plan_context ───────────────────────────────────────────────────
cur_codes = set(plan_eng.get_current_sem_codes(s_key, p_key, track_key, cur_seq))
nxt_codes = set(plan_eng.get_next_sem_codes(s_key, p_key, track_key, cur_seq))
bef_codes = set(plan_eng.get_planned_codes_before(s_key, p_key, track_key, cur_seq))
print("\n=== STEP 5: plan_context for cur_seq=1 ===")
print(f"  current_sem_codes (sem 1): {sorted(cur_codes)}")
print(f"  next_sem_codes    (sem 2): {sorted(nxt_codes)}")
print(f"  planned_before    (<sem1): {sorted(bef_codes)}")

plan_context = {
    "current_sem_codes":    cur_codes,
    "next_sem_codes":       nxt_codes,
    "premature_codes":      set(plan_eng.get_premature_codes(s_key, p_key, track_key, cur_seq)),
    "planned_before_codes": bef_codes,
}

# ── Step 6: analyze_eligibility ───────────────────────────────────────────
analysis = graph.analyze_eligibility(
    completed=effective_completed,
    failed=p.failed_courses,
    current=[],
    plan_context=plan_context,
)
print("\n=== STEP 6: analyze_eligibility() results ===")
print(f"  completed_codes count: {len(analysis['completed_codes'])}")
print(f"  retake_eligible: {[c[0] for c in analysis['retake_eligible']]}")
print(f"  current_sem_eligible: {[c[0] for c in analysis['current_sem_eligible']]}")
print(f"  next_sem_eligible: {[c[0] for c in analysis['next_sem_eligible']]}")
print(f"  behind_plan_eligible: {[c[0] for c in analysis['behind_plan_eligible']]}")
eligible_codes = [c[0] for c in analysis['eligible']]
print(f"  eligible (all, {len(eligible_codes)} courses): {eligible_codes[:20]}")
biol_in_eligible = [c for c in eligible_codes if c.startswith('BIOL')]
print(f"  BIOL in eligible: {biol_in_eligible}")

# ── Step 7: school prefix filter for eligible ─────────────────────────────
school  = p.school.upper() if p.school else ""
allowed = _SCHOOL_CODE_PREFIXES.get(school, frozenset())
print(f"\n=== STEP 7: _SCHOOL_CODE_PREFIXES[{school!r}] ===")
print(f"  allowed prefixes: {sorted(allowed)}")
print(f"  BIOL in allowed: {'BIOL' in allowed}")
filtered_eligible = [c for c in eligible_codes if c.split()[0] in allowed]
biol_after_filter = [c for c in filtered_eligible if c.startswith('BIOL')]
print(f"  BIOL after school-prefix filter: {biol_after_filter}")

# ── Step 8: PlanningEngine.compute() ──────────────────────────────────────
progression = CurriculumProgressionMetrics(
    status='unknown', expected_done=[], missing_core=[],
    extra_done=[], delayed_count=0, total_expected=0,
    total_completed=0, expected_credits=0, actual_credits=0,
)
computed_plan = planner.compute(p, analysis, progression)
print("\n=== STEP 8: PlanningEngine.compute() output ===")
print("  Safe plan courses:")
for c in computed_plan.safe.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")
print("  Balanced plan courses:")
for c in computed_plan.balanced.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")
print("  Fast plan courses:")
for c in computed_plan.fast.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")

# show if BIOL appears in any plan
all_plan_codes = (
    {c.code for c in computed_plan.safe.courses}
    | {c.code for c in computed_plan.balanced.courses}
    | {c.code for c in computed_plan.fast.courses}
)
biol_in_plan = [c for c in all_plan_codes if c.startswith('BIOL')]
csai_sem1_in_plan = [c for c in all_plan_codes if c in cur_codes]
print(f"\n  BIOL codes in any plan tier: {biol_in_plan}")
print(f"  Semester-1 plan codes appearing in recommendation: {csai_sem1_in_plan}")

# ── Root cause summary ────────────────────────────────────────────────────
print("\n=== ROOT CAUSE SUMMARY ===")
print(f"  Bug A: infer_presumed_completed(cur_seq=1) returns {len(presumed)} courses")
print(f"         -> effective_completed is EMPTY")
print(f"         -> ALL {len(analysis['current_sem_eligible'])} sem-1 courses are 'current_sem_eligible'")
print(f"         -> PlanningEngine recommends them as semester-1 plan courses")
print(f"  Bug B: BIOL in _SCHOOL_CODE_PREFIXES['CSAI']: {'BIOL' in allowed}")
print(f"         BIOL courses in filtered eligible list: {biol_after_filter}")
print(f"         -> PlanningEngine P4 (elective) tier picks BIOL as electives")
