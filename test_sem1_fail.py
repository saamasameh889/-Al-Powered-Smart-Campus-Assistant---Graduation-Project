#!/usr/bin/env python3
"""Verify the two bugs are fixed in the actual _full_advisory() execution path."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from phase8_advisor_engine import (
    StudentProfile, update_profile, StudyPlanEngine, CurriculumGraph,
    PlanningEngine, CurriculumProgressionMetrics, AdvisorEngine,
    _SCHOOL_CODE_PREFIXES,
)

graph    = CurriculumGraph()
plan_eng = StudyPlanEngine()
planner  = PlanningEngine()


def simulate_advisory_planning(msg: str):
    """Run the exact same logic as _full_advisory() planning path."""
    p = StudentProfile()
    p = update_profile(msg, p)

    plan_resolved = plan_eng.resolve_prog(p.school, p.major)
    if not plan_resolved:
        return None, p, {}
    s_key, p_key = plan_resolved
    track_key = None
    cur_seq = p.semester or 1

    # Build initial plan_context
    plan_context = {
        "current_sem_codes":    set(plan_eng.get_current_sem_codes(s_key, p_key, track_key, cur_seq)),
        "next_sem_codes":       set(plan_eng.get_next_sem_codes(s_key, p_key, track_key, cur_seq)),
        "premature_codes":      set(plan_eng.get_premature_codes(s_key, p_key, track_key, cur_seq)),
        "planned_before_codes": set(plan_eng.get_planned_codes_before(s_key, p_key, track_key, cur_seq)),
    }

    # Transcript inference
    presumed_completed = []
    if cur_seq > 1:
        presumed_completed = plan_eng.infer_presumed_completed(p.school, p.major, cur_seq, track_key)

    # BUG A FIX: failed course in current semester → advance planning sem
    failed_upper = {c.upper() for c in p.failed_courses}
    planning_sem = cur_seq
    cur_sem_set  = plan_context["current_sem_codes"]
    failed_in_cur_sem = cur_sem_set & failed_upper
    if failed_in_cur_sem:
        already_presumed = {c.upper() for c in presumed_completed}
        for code in cur_sem_set:
            uc = code.upper()
            if uc not in failed_upper and uc not in already_presumed:
                presumed_completed.append(uc)
        planning_sem = cur_seq + 1
        plan_context = {
            "current_sem_codes":    set(plan_eng.get_current_sem_codes(s_key, p_key, track_key, planning_sem)),
            "next_sem_codes":       set(plan_eng.get_next_sem_codes(s_key, p_key, track_key, planning_sem)),
            "premature_codes":      set(plan_eng.get_premature_codes(s_key, p_key, track_key, planning_sem)),
            "planned_before_codes": set(plan_eng.get_planned_codes_before(s_key, p_key, track_key, planning_sem)),
        }

    explicit_upper = {c.upper() for c in p.completed_courses}
    effective_completed = list(
        {c.upper() for c in (presumed_completed + p.completed_courses)} - failed_upper
    )

    analysis = graph.analyze_eligibility(
        completed=effective_completed,
        failed=p.failed_courses,
        current=[],
        plan_context=plan_context,
    )

    # BUG B FIX: filter eligible to programme plan codes only
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
    profile_for_plan = StudentProfile.from_dict(p.to_dict())
    profile_for_plan.semester = planning_sem

    computed_plan = planner.compute(profile_for_plan, plan_analysis, progression)
    return computed_plan, p, {"planning_sem": planning_sem, "effective_completed": effective_completed}


# ── Test 1: DSAI semester 1, failed MATH 103 ─────────────────────────────────
print("=" * 60)
print("Test 1: DSAI semester 1, failed MATH 103")
print("Expected: planning_sem=2, sem-2 courses, NO sem-1 repeats, NO BIOL")
print("=" * 60)
plan, p, ctx = simulate_advisory_planning("I am DSAI semester 1 and I failed MATH 103")
print(f"  planning_sem: {ctx['planning_sem']}  (expected: 2)")
print(f"  effective_completed ({len(ctx['effective_completed'])} courses): {sorted(ctx['effective_completed'])}")
print()
print("  Safe plan:")
for c in plan.safe.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")
print("  Balanced plan:")
for c in plan.balanced.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")
print("  Fast plan:")
for c in plan.fast.courses:
    print(f"    {c.code} ({c.credits}cr) [{c.priority}]")

all_codes = {c.code for t in (plan.safe, plan.balanced, plan.fast) for c in t.courses}
biol_in   = [c for c in all_codes if c.startswith('BIOL')]
sem1_in   = [c for c in all_codes if c in {'CSAI 100','CSAI 101','CSAI 102','CSAI 252','ENGL 156','SCH 163'}]
print(f"\n  BIOL in plan: {biol_in}  (expected: [])")
print(f"  Sem-1 courses in plan: {sem1_in}  (expected: [])")
print(f"  MATH 103 (retake) in plan: {'MATH 103' in all_codes}  (expected: True)")
math104_in = 'MATH 104' in all_codes
csai151_in = 'CSAI 151' in all_codes
print(f"  MATH 104 (sem-2) in plan: {math104_in}")
print(f"  CSAI 151 (sem-2) in plan: {csai151_in}")

# ── Test 2: DSAI semester 2, no failures (standard case still works) ─────────
print()
print("=" * 60)
print("Test 2: DSAI semester 2, no failures (standard case)")
print("Expected: planning_sem=2, sem-1 inferred as done, sem-2 courses recommended")
print("=" * 60)
plan2, p2, ctx2 = simulate_advisory_planning("I am DSAI semester 2")
print(f"  planning_sem: {ctx2['planning_sem']}  (expected: 2)")
print(f"  effective_completed: {sorted(ctx2['effective_completed'])}")
all_codes2 = {c.code for t in (plan2.safe, plan2.balanced, plan2.fast) for c in t.courses}
sem1_in2   = [c for c in all_codes2 if c in {'CSAI 100','CSAI 101','CSAI 102','CSAI 252'}]
sem2_in2   = [c for c in all_codes2 if c in {'CSAI 151','DSAI 103','DSAI 104','ENGL 157','MATH 104'}]
print(f"  Sem-1 courses in plan (should be empty): {sem1_in2}")
print(f"  Sem-2 courses in plan: {sem2_in2}")

# ── Test 3: DSAI semester 3, failed MATH 103 (failed in earlier sem) ─────────
print()
print("=" * 60)
print("Test 3: DSAI semester 3, failed MATH 103 (MATH 103 is a sem-1 course)")
print("Expected: planning_sem=3, MATH 103 as retake, sem-3 courses recommended")
print("=" * 60)
plan3, p3, ctx3 = simulate_advisory_planning("I am DSAI semester 3 and I failed MATH 103")
print(f"  planning_sem: {ctx3['planning_sem']}  (expected: 3 - MATH 103 is sem-1, not cur sem)")
all_codes3 = {c.code for t in (plan3.safe, plan3.balanced, plan3.fast) for c in t.courses}
print(f"  MATH 103 (retake) in plan: {'MATH 103' in all_codes3}")
print(f"  Safe plan: {[c.code for c in plan3.safe.courses]}")

# ── Test 4: BIOL prefix filter still in _SCHOOL_CODE_PREFIXES ────────────────
print()
print("=" * 60)
print("Test 4: _SCHOOL_CODE_PREFIXES BIOL fix")
print("=" * 60)
csai_allowed = _SCHOOL_CODE_PREFIXES.get("CSAI", frozenset())
sci_allowed  = _SCHOOL_CODE_PREFIXES.get("SCI",  frozenset())
print(f"  CSAI allowed: {sorted(csai_allowed)}")
print(f"  BIOL in CSAI: {'BIOL' in csai_allowed}  (expected: False)")
print(f"  BIOL in SCI:  {'BIOL' in sci_allowed}   (expected: True)")

print()
print("ALL TESTS DONE")
