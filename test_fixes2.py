#!/usr/bin/env python3
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from phase8_advisor_engine import (
    StudentProfile, update_profile, CurriculumGraph, StudyPlanEngine,
    PlanningEngine, AdvisorEngine, CurriculumProgressionMetrics,
)

# Test 1: Multiline PDF-pasted course list
msg1 = (
    "I am in my first year CSAI and first semester and I have complete this courses "
    "CSAI 100 Introduction to Computational\nSciences and AI\n"
    "CSAI 101 Fundamentals of Programming and\nComputer Science\n"
    "CSAI 102 Digital Logic and Computer\n"
    "Architecture CSAI 252 Introduction to Computer Networks ENGL 156 Technical English\n"
    "MATH 103 Calculus for Computational Sciences SCH 163 Sustainability, Social and Ethical\n"
    "Issues in Computing , And after finishing the semester I failed in CSAI 151 "
    "what should I take the next semester if I will be in DSAI ?"
)

p = StudentProfile()
p = update_profile(msg1, p)
print("Test 1 - Multiline PDF input:")
print("  school:", p.school, "| major:", p.major, "| semester:", p.semester)
print("  completed:", p.completed_courses)
print("  failed:", p.failed_courses)
print()

# Test 2: Ordinal semester
p2 = StudentProfile()
p2 = update_profile("I am in my first semester at Zewail, CSAI.", p2)
print("Test 2a - 'first semester':", p2.semester)

p3 = StudentProfile()
p3 = update_profile("I am a second year student in BUS.", p3)
print("Test 2b - 'second year':", p3.semester)
print()

# Test 3: courses taken format
p5 = StudentProfile()
p5 = update_profile("Semester 1 and here is the courses taken: CSAI 101, CSAI 102, CSAI 252, MATH 103", p5)
print("Test 3 - 'courses taken:' format:")
print("  semester:", p5.semester, "| completed:", p5.completed_courses)
print()

# Test 4: School prefix filter for blocked list
graph    = CurriculumGraph()
plan_eng = StudyPlanEngine()

p6 = StudentProfile(
    school='CSAI', major='DSAI', semester=3, gpa=3.2,
    completed_courses=['CSAI 100', 'CSAI 101', 'CSAI 102', 'MATH 101', 'MATH 102'],
)
s_key, p_key = plan_eng.resolve_prog(p6.school, p6.major)
plan_ctx = {
    'current_sem_codes':    set(plan_eng.get_current_sem_codes(s_key, p_key, None, 3)),
    'next_sem_codes':       set(plan_eng.get_next_sem_codes(s_key, p_key, None, 3)),
    'premature_codes':      set(plan_eng.get_premature_codes(s_key, p_key, None, 3)),
    'planned_before_codes': set(plan_eng.get_planned_codes_before(s_key, p_key, None, 3)),
}
analysis = graph.analyze_eligibility(
    completed=p6.completed_courses, failed=[], current=[], plan_context=plan_ctx,
)

raw_blocked_codes = [b[0] for b in analysis['blocked']]
chem_in_raw = [c for c in raw_blocked_codes if c.startswith('CHEM')]
print("Test 4 - School prefix filter:")
print("  Raw blocked count:", len(analysis['blocked']))
print("  CHEM in raw blocked:", chem_in_raw[:3])

progression = CurriculumProgressionMetrics(
    status='unknown', expected_done=[], missing_core=[], extra_done=[],
    delayed_count=0, total_expected=0, total_completed=0,
    expected_credits=0, actual_credits=0,
)
computed_plan = PlanningEngine().compute(p6, analysis, progression)

ae = object.__new__(AdvisorEngine)
ae._graph = graph
ae._study_plan = plan_eng
ae._planner = PlanningEngine()

elig_md = ae._preformat_eligibility_for_plan(analysis, computed_plan, p6)
chem_in_output = 'CHEM' in elig_md
print("  CHEM in eligibility output:", chem_in_output)
for line in elig_md.split('\n'):
    if 'Blocked' in line or ('**' in line and 'CSAI' in line):
        print("  ", line[:80])

# Show first blocked entries in filtered graph section
gs = ae._format_graph_section(analysis, p6)
blocked_section = False
blocked_lines = []
for line in gs.split('\n'):
    if 'Blocked' in line:
        blocked_section = True
    if blocked_section and line.strip():
        blocked_lines.append(line)
    if len(blocked_lines) >= 6:
        break
print()
print("  Graph section blocked (first 5):")
for l in blocked_lines:
    print("   ", l[:80])

print()
print("ALL TESTS DONE")
