#!/usr/bin/env python3
"""Trace: DSAI sem 3, GPA 2.0, failed CSAI 151 → CSAI 201/202 in plan?"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from phase8_advisor_engine import (
    StudentProfile, update_profile, StudyPlanEngine, CurriculumGraph,
    PlanningEngine, CurriculumProgressionMetrics,
)

graph    = CurriculumGraph()
plan_eng = StudyPlanEngine()
planner  = PlanningEngine()

# ── Step 1: Profile ──────────────────────────────────────────────────────────
p = StudentProfile()
p = update_profile("I am DSAI semester 3, GPA 2.0, failed CSAI 151", p)
print("=== STEP 1: StudentProfile ===")
print(f"  school={p.school!r}  major={p.major!r}  semester={p.semester}  gpa={p.gpa}")
print(f"  failed_courses: {p.failed_courses}")

# ── Step 2: plan resolution ───────────────────────────────────────────────────
plan_resolved = plan_eng.resolve_prog(p.school, p.major)
s_key, p_key = plan_resolved
track_key = None
cur_seq = p.semester
print(f"\n=== STEP 2: resolve_prog -> {plan_resolved}, cur_seq={cur_seq} ===")

# ── Step 3: failed_in_cur_sem detection ──────────────────────────────────────
cur_sem_codes = set(plan_eng.get_current_sem_codes(s_key, p_key, track_key, cur_seq))
failed_upper  = {c.upper() for c in p.failed_courses}
failed_in_cur = cur_sem_codes & failed_upper
print(f"\n=== STEP 3: failed-in-current-sem detection ===")
print(f"  cur_sem_codes (sem 3): {sorted(cur_sem_codes)}")
print(f"  failed_upper: {failed_upper}")
print(f"  failed_in_cur_sem: {failed_in_cur}")
print(f"  -> planning_sem advances? {bool(failed_in_cur)}")

# ── Step 4: infer_presumed_completed ─────────────────────────────────────────
presumed = plan_eng.infer_presumed_completed(p.school, p.major, cur_seq, track_key)
print(f"\n=== STEP 4: infer_presumed_completed(cur_seq={cur_seq}) ===")
print(f"  presumed ({len(presumed)} courses): {sorted(presumed)}")
print(f"  CSAI 151 in presumed: {'CSAI 151' in [c.upper() for c in presumed]}")

# ── Step 5: effective_completed ──────────────────────────────────────────────
effective_completed = list(
    {c.upper() for c in (presumed + p.completed_courses)} - failed_upper
)
print(f"\n=== STEP 5: effective_completed ===")
print(f"  effective_completed ({len(effective_completed)} courses): {sorted(effective_completed)}")
print(f"  CSAI 151 in effective_completed: {'CSAI 151' in effective_completed}")

# ── Step 6: analyze_eligibility internals ────────────────────────────────────
plan_context = {
    "current_sem_codes":    set(plan_eng.get_current_sem_codes(s_key, p_key, track_key, cur_seq)),
    "next_sem_codes":       set(plan_eng.get_next_sem_codes(s_key, p_key, track_key, cur_seq)),
    "premature_codes":      set(plan_eng.get_premature_codes(s_key, p_key, track_key, cur_seq)),
    "planned_before_codes": set(plan_eng.get_planned_codes_before(s_key, p_key, track_key, cur_seq)),
}

analysis = graph.analyze_eligibility(
    completed=effective_completed,
    failed=p.failed_courses,
    current=[],
    plan_context=plan_context,
)
print(f"\n=== STEP 6: analyze_eligibility() ===")
print(f"  completed_codes in analysis: {len(analysis['completed_codes'])}")
print(f"  CSAI 151 in completed_codes: {'CSAI 151' in analysis['completed_codes']}")
print(f"  CSAI 151 in failed_codes:    {'CSAI 151' in analysis['failed_codes']}")

# Check prereqs for CSAI 201 and 202
print(f"\n  Prereqs for CSAI 201: {graph.get_prereqs('CSAI 201')}")
print(f"  Prereqs for CSAI 202: {graph.get_prereqs('CSAI 202')}")

retake = [c[0] for c in analysis['retake_eligible']]
cur_elig = [c[0] for c in analysis['current_sem_eligible']]
blocked_codes = [b[0] for b in analysis['blocked']]
print(f"\n  retake_eligible: {retake}")
print(f"  current_sem_eligible: {cur_elig}")
print(f"  CSAI 201 in current_sem_eligible: {'CSAI 201' in cur_elig}")
print(f"  CSAI 202 in current_sem_eligible: {'CSAI 202' in cur_elig}")
print(f"  CSAI 201 in blocked: {'CSAI 201' in blocked_codes}")
print(f"  CSAI 202 in blocked: {'CSAI 202' in blocked_codes}")

# ── Step 7: PlanningEngine.compute() ─────────────────────────────────────────
# Apply plan-elective filter (as in _full_advisory)
all_plan_codes = set(plan_eng.get_planned_codes_before(s_key, p_key, track_key, 99))
all_plan_codes |= plan_context.get("current_sem_codes", set())
all_plan_codes |= plan_context.get("next_sem_codes", set())
all_plan_codes_upper = {c.upper() for c in all_plan_codes}
plan_analysis = dict(analysis)
plan_analysis["eligible"] = [
    e for e in analysis["eligible"] if e[0].upper() in all_plan_codes_upper
]

progression = CurriculumProgressionMetrics(
    status='unknown', expected_done=[], missing_core=[],
    extra_done=[], delayed_count=0, total_expected=0,
    total_completed=0, expected_credits=0, actual_credits=0,
)
computed = planner.compute(p, plan_analysis, progression)

print(f"\n=== STEP 7: PlanningEngine.compute() ===")
print("  Safe plan:")
for c in computed.safe.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")
print("  Balanced plan:")
for c in computed.balanced.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")

all_plan_codes_out = {c.code for t in (computed.safe, computed.balanced, computed.fast) for c in t.courses}
print(f"\n  CSAI 201 in any plan tier: {'CSAI 201' in all_plan_codes_out}")
print(f"  CSAI 202 in any plan tier: {'CSAI 202' in all_plan_codes_out}")
print(f"  CSAI 151 in any plan tier: {'CSAI 151' in all_plan_codes_out}")

# ── Step 8: Read PlanningEngine source to understand what it does ────────────
print(f"\n=== STEP 8: What does PlanningEngine use as input? ===")
print(f"  PlanningEngine receives:")
print(f"    analysis['retake_eligible'] = {[c[0] for c in plan_analysis['retake_eligible']]}")
print(f"    analysis['current_sem_eligible'] = {[c[0] for c in plan_analysis['current_sem_eligible']]}")
print(f"    analysis['next_sem_eligible'] = {[c[0] for c in plan_analysis['next_sem_eligible']]}")
print(f"    analysis['behind_plan_eligible'] = {[c[0] for c in plan_analysis['behind_plan_eligible']]}")
print(f"    analysis['eligible'] (filtered) = {[c[0] for c in plan_analysis['eligible']][:10]}")
