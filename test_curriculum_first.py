#!/usr/bin/env python3
"""
Test: Curriculum-First Mode (Transcript Inference Rule)

Verifies that when a student says "I am DSAI semester 3",
the system automatically assumes semesters 1-2 are completed
WITHOUT asking the student to list every course.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from phase8_advisor_engine import (
    StudentProfile, update_profile, StudyPlanEngine, CurriculumGraph,
    PlanningEngine, CurriculumProgressionMetrics, AdvisorEngine,
)

graph    = CurriculumGraph()
plan_eng = StudyPlanEngine()

# ── Test 1: Semester 3 DSAI, no explicit course list ──────────────────────────
print("=" * 60)
print("Test 1: 'I am DSAI semester 3' — no course list given")
print("=" * 60)

p = StudentProfile()
p = update_profile("I am in DSAI semester 3", p)
print(f"  Extracted: school={p.school!r} major={p.major!r} sem={p.semester}")
print(f"  Explicit completed: {p.completed_courses}")

# Simulate what _full_advisory does
plan_resolved = plan_eng.resolve_prog(p.school, p.major)
print(f"  Plan resolved: {plan_resolved}")

if plan_resolved and p.semester > 1:
    s_key, p_key = plan_resolved
    presumed = plan_eng.infer_presumed_completed(p.school, p.major, p.semester)
    print(f"  Presumed completed ({len(presumed)} courses from sems 1-{p.semester-1}):")
    print(f"    {', '.join(presumed)}")

    failed_upper = {c.upper() for c in p.failed_courses}
    effective = list(
        {c.upper() for c in (presumed + p.completed_courses)} - failed_upper
    )
    print(f"  Effective completed: {len(effective)} courses")

    cur_codes = set(plan_eng.get_current_sem_codes(s_key, p_key, None, p.semester))
    nxt_codes = set(plan_eng.get_next_sem_codes(s_key, p_key, None, p.semester))
    pre_codes = set(plan_eng.get_premature_codes(s_key, p_key, None, p.semester))
    bef_codes = set(plan_eng.get_planned_codes_before(s_key, p_key, None, p.semester))
    plan_context = {
        "current_sem_codes": cur_codes,
        "next_sem_codes": nxt_codes,
        "premature_codes": pre_codes,
        "planned_before_codes": bef_codes,
    }

    analysis = graph.analyze_eligibility(
        completed=effective,
        failed=p.failed_courses,
        current=p.current_courses,
        plan_context=plan_context,
    )
    print(f"  current_sem_eligible: {[c[0] for c in analysis['current_sem_eligible']]}")
    print(f"  behind_plan_eligible (catch-up): {[c[0] for c in analysis['behind_plan_eligible']]}")
    print(f"  next_sem_eligible: {[c[0] for c in analysis['next_sem_eligible']]}")
    print(f"  Completed count in analysis: {len(analysis['completed_codes'])}")

# ── Test 2: Semester 3 DSAI, failed MATH 103 ──────────────────────────────────
print()
print("=" * 60)
print("Test 2: 'DSAI semester 3, failed MATH 103' — minimal input")
print("=" * 60)

p2 = StudentProfile()
p2 = update_profile("I am DSAI semester 3 and I failed MATH 103", p2)
print(f"  Extracted: school={p2.school!r} major={p2.major!r} sem={p2.semester}")
print(f"  Failed: {p2.failed_courses}")

if plan_resolved and p2.semester > 1:
    presumed2 = plan_eng.infer_presumed_completed(p2.school, p2.major, p2.semester)
    failed_upper2 = {c.upper() for c in p2.failed_courses}
    effective2 = list(
        {c.upper() for c in (presumed2 + p2.completed_courses)} - failed_upper2
    )
    print(f"  Effective completed (presumed minus failed): {len(effective2)} courses")
    print(f"  MATH 103 in effective_completed: {'MATH 103' in effective2}")
    print(f"  MATH 103 in presumed: {'MATH 103' in [c.upper() for c in presumed2]}")

    analysis2 = graph.analyze_eligibility(
        completed=effective2,
        failed=p2.failed_courses,
        current=[],
        plan_context=plan_context,
    )
    print(f"  MATH 103 in retake_eligible: {'MATH 103' in [c[0] for c in analysis2['retake_eligible']]}")
    print(f"  current_sem_eligible: {[c[0] for c in analysis2['current_sem_eligible']]}")

# ── Test 3: Planning readiness — only programme+semester needed ────────────────
print()
print("=" * 60)
print("Test 3: _check_planning_readiness — no course list needed")
print("=" * 60)

ae = object.__new__(AdvisorEngine)
ae._graph = graph
ae._study_plan = plan_eng
ae._planner = PlanningEngine()

p3 = StudentProfile(school="CSAI", major="DSAI", semester=3)
missing = ae._check_planning_readiness("what should I take?", p3)
print(f"  Missing (expect []): {missing}")

p4 = StudentProfile()
missing4 = ae._check_planning_readiness("what courses should I take?", p4)
print(f"  Missing without profile (expect ['programme', 'current_semester']): {missing4}")

p5 = StudentProfile(school="CSAI", major="DSAI")
missing5 = ae._check_planning_readiness("what courses to take?", p5)
print(f"  Missing without semester (expect ['current_semester']): {missing5}")

print()
print("ALL TESTS DONE")
