#!/usr/bin/env python3
"""
validate_plans.py
Validate data/curriculum/study_plans.json against known constraints.

Exit codes:
  0 — all checks passed (warnings may still be printed)
  1 — one or more FAIL checks found

Usage:
  python validate_plans.py
  python validate_plans.py --strict      # treat warnings as failures
  python validate_plans.py --json        # machine-readable JSON report to stdout
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT     = Path(__file__).parent
STUDY_PLANS_FILE = PROJECT_ROOT / "data" / "curriculum" / "study_plans.json"
COURSES_FILE     = PROJECT_ROOT / "data" / "curriculum" / "courses.json"

# Expected semester sequence for a full 4-year programme
_EXPECTED_SEMS = [
    "Y1S1", "Y1S2",
    "Y2S1", "Y2S2",
    "Y3S1", "Y3S2",
    "Y4S1", "Y4S2",
]
# Tracks that are expected to only have Y3+ semesters (concentration-only)
_CONC_ONLY_TRACKS = {
    "APD", "GCG", "HCI",           # CSAI/SWD concentrations
    "ITNS", "ITCC",                 # CSAI/IT concentrations
    "CBG", "MCB", "DDD", "MED",    # SCI/BMS concentrations
    "NPHY", "NCHEM", "BIONANO", "NMED",  # SCI/NANO concentrations
    "AST", "HEP",                   # SCI/PHY concentrations
}
_CONC_EXPECTED_SEMS = ["Y3S1", "Y3S2", "Y4S1", "Y4S2"]

# Credit bounds: warn if a regular semester falls outside this range
_MIN_CREDITS_WARN = 9
_MAX_CREDITS_WARN = 25


class CheckResult(NamedTuple):
    level:   str   # "PASS" | "WARN" | "FAIL"
    check:   str   # short check name
    detail:  str   # human-readable detail


def _check_schema(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    required_sem_fields = {"year", "semester", "seq", "courses", "total_credits"}

    for school, progs in plans.items():
        if not isinstance(progs, dict):
            results.append(CheckResult("FAIL", "schema",
                f"{school}: expected dict of programmes, got {type(progs).__name__}"))
            continue

        for prog, pdata in progs.items():
            if not isinstance(pdata, dict):
                results.append(CheckResult("FAIL", "schema",
                    f"{school}/{prog}: expected dict, got {type(pdata).__name__}"))
                continue

            tracks = pdata.get("tracks")
            if not tracks:
                results.append(CheckResult("FAIL", "schema",
                    f"{school}/{prog}: missing 'tracks' key"))
                continue

            for track, tdata in tracks.items():
                semesters = tdata.get("semesters")
                if semesters is None:
                    results.append(CheckResult("FAIL", "schema",
                        f"{school}/{prog}/{track}: missing 'semesters' key"))
                    continue

                for sem_key, sdata in semesters.items():
                    missing = required_sem_fields - set(sdata.keys())
                    if missing:
                        results.append(CheckResult("FAIL", "schema",
                            f"{school}/{prog}/{track}/{sem_key}: missing fields {sorted(missing)}"))

                    courses = sdata.get("courses", [])
                    if not isinstance(courses, list):
                        results.append(CheckResult("FAIL", "schema",
                            f"{school}/{prog}/{track}/{sem_key}: 'courses' must be a list"))
                        continue

                    for i, c in enumerate(courses):
                        for cf in ("code", "name"):
                            if cf not in c:
                                results.append(CheckResult("FAIL", "schema",
                                    f"{school}/{prog}/{track}/{sem_key} course[{i}]: missing '{cf}'"))

    if not any(r.level == "FAIL" for r in results):
        results.append(CheckResult("PASS", "schema", "All required fields present"))
    return results


def _check_credit_totals(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    mismatches = 0

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    courses = sdata.get("courses", [])
                    stored  = sdata.get("total_credits", 0)
                    computed = sum(c.get("credits") or 0 for c in courses)
                    if stored != computed:
                        mismatches += 1
                        results.append(CheckResult("FAIL", "credit_totals",
                            f"{school}/{prog}/{track}/{sem_key}: stored={stored}, computed={computed}"))

    if mismatches == 0:
        results.append(CheckResult("PASS", "credit_totals",
            "All total_credits match sum of course credits"))
    return results


def _check_duplicate_codes(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    dup_count = 0

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    codes = [c.get("code", "") for c in sdata.get("courses", [])]
                    seen: set = set()
                    for code in codes:
                        if code in seen:
                            dup_count += 1
                            results.append(CheckResult("FAIL", "duplicate_codes",
                                f"{school}/{prog}/{track}/{sem_key}: duplicate code '{code}'"))
                        seen.add(code)

    if dup_count == 0:
        results.append(CheckResult("PASS", "duplicate_codes",
            "No duplicate course codes within any semester"))
    return results


def _check_empty_semesters(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    empty_count = 0

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    if not sdata.get("courses"):
                        empty_count += 1
                        results.append(CheckResult("FAIL", "empty_semesters",
                            f"{school}/{prog}/{track}/{sem_key}: 0 courses"))

    if empty_count == 0:
        results.append(CheckResult("PASS", "empty_semesters",
            "All semesters have at least 1 course"))
    return results


def _check_seq_field(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    bad = 0

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    seq = sdata.get("seq", -1)
                    is_summer = sdata.get("semester") == "summer"
                    if not is_summer and seq <= 0:
                        bad += 1
                        results.append(CheckResult("WARN", "seq_field",
                            f"{school}/{prog}/{track}/{sem_key}: seq={seq} (expected > 0 for non-summer)"))

    if bad == 0:
        results.append(CheckResult("PASS", "seq_field",
            "All non-summer semesters have valid seq values"))
    return results


def _check_credit_loads(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    warnings = 0

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    is_summer = sdata.get("semester") == "summer"
                    cr = sdata.get("total_credits", 0)
                    if is_summer:
                        continue
                    if cr < _MIN_CREDITS_WARN or cr > _MAX_CREDITS_WARN:
                        warnings += 1
                        results.append(CheckResult("WARN", "credit_loads",
                            f"{school}/{prog}/{track}/{sem_key}: {cr} credits "
                            f"(expected {_MIN_CREDITS_WARN}-{_MAX_CREDITS_WARN})"))

    if warnings == 0:
        results.append(CheckResult("PASS", "credit_loads",
            f"All regular semesters have credit loads within "
            f"[{_MIN_CREDITS_WARN}, {_MAX_CREDITS_WARN}]"))
    return results


def _check_null_credits(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    null_count = 0

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    for c in sdata.get("courses", []):
                        if c.get("credits") is None:
                            null_count += 1

    if null_count > 0:
        results.append(CheckResult("WARN", "null_credits",
            f"{null_count} course(s) have credits=null (PDF parse ambiguity)"))
    else:
        results.append(CheckResult("PASS", "null_credits",
            "All courses have explicit credit values"))
    return results


def _check_track_completeness(plans: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    warnings = 0

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                sems = set(tdata.get("semesters", {}).keys())
                summer_keys = {s for s in sems if "Summer" in s}
                regular_sems = sems - summer_keys

                if track == "_common":
                    # _common tracks hold partial data by design
                    continue

                expected = (
                    _CONC_EXPECTED_SEMS if track in _CONC_ONLY_TRACKS
                    else _EXPECTED_SEMS
                )

                missing_sems = [s for s in expected if s not in regular_sems]
                if missing_sems:
                    warnings += 1
                    results.append(CheckResult("WARN", "track_completeness",
                        f"{school}/{prog}/{track}: missing expected semesters {missing_sems}"))

    if warnings == 0:
        results.append(CheckResult("PASS", "track_completeness",
            "All non-_common tracks have expected semesters"))
    return results


def _check_course_catalog(plans: dict, courses: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    known = set(courses.keys())
    all_codes: set[str] = set()

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    for c in sdata.get("courses", []):
                        code = c.get("code", "")
                        if code:
                            all_codes.add(code)

    missing = all_codes - known
    coverage_pct = 100.0 * len(known & all_codes) / len(all_codes) if all_codes else 0.0

    results.append(CheckResult(
        "PASS" if coverage_pct >= 30 else "WARN",
        "course_catalog",
        f"{len(known & all_codes)}/{len(all_codes)} codes in courses.json "
        f"({coverage_pct:.1f}% coverage). "
        f"{len(missing)} codes not in catalog — expected for SCI/BMS/NANO codes "
        f"extracted from PDF but not yet in courses.json."
    ))

    if missing:
        sample = sorted(missing)[:20]
        results.append(CheckResult("WARN", "course_catalog",
            f"First 20 uncatalogued codes: {sample}"))

    return results


def _check_school_credit_totals(plans: dict) -> list[CheckResult]:
    """Estimate total credits per complete programme path."""
    results: list[CheckResult] = []

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            # Sum up _common + first concentration (if any)
            tracks = pdata.get("tracks", {})
            common_sems = tracks.get("_common", {}).get("semesters", {})

            conc_tracks = {k: v for k, v in tracks.items() if k != "_common"}

            # Build a representative full plan: _common + one concentration
            all_sems: dict = dict(common_sems)
            if conc_tracks:
                first_conc = next(iter(conc_tracks.values()))
                for sk, sd in first_conc.get("semesters", {}).items():
                    if sk not in all_sems:
                        all_sems[sk] = sd

            total = sum(sd.get("total_credits", 0) for sd in all_sems.values())
            sem_count = len(all_sems)

            # Sanity check: expect 100-145 credits for a 4-year programme
            level = "PASS" if 80 <= total <= 160 else "WARN"
            results.append(CheckResult(level, "programme_credits",
                f"{school}/{prog}: ~{total} credits over {sem_count} semesters "
                f"({'_common' + ('+' + next(iter(conc_tracks)) if conc_tracks else '')})"))

    return results


def run_all_checks(plans: dict, courses: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    results += _check_schema(plans)
    results += _check_credit_totals(plans)
    results += _check_duplicate_codes(plans)
    results += _check_empty_semesters(plans)
    results += _check_seq_field(plans)
    results += _check_credit_loads(plans)
    results += _check_null_credits(plans)
    results += _check_track_completeness(plans)
    results += _check_course_catalog(plans, courses)
    results += _check_school_credit_totals(plans)
    return results


def _load_json(path: Path, key: str | None = None) -> dict:
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, {}) if key else data


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate study_plans.json")
    parser.add_argument("--strict", action="store_true",
                        help="Treat WARN as failure (exit 1)")
    parser.add_argument("--json",   action="store_true",
                        help="Output machine-readable JSON to stdout")
    args = parser.parse_args()

    full_plans = _load_json(STUDY_PLANS_FILE)
    plans   = full_plans.get("plans", {})
    courses = _load_json(COURSES_FILE).get("courses", {})

    results = run_all_checks(plans, courses)

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in results:
        counts[r.level] += 1

    if args.json:
        output = {
            "study_plans_file": str(STUDY_PLANS_FILE),
            "summary": counts,
            "results": [{"level": r.level, "check": r.check, "detail": r.detail}
                        for r in results],
        }
        print(json.dumps(output, indent=2))
    else:
        symbols = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}
        print(f"\nValidating: {STUDY_PLANS_FILE.name}")
        print("=" * 70)

        current_check = ""
        for r in results:
            if r.check != current_check:
                current_check = r.check
                print(f"\n[{r.check.upper()}]")
            sym = symbols[r.level]
            print(f"  {sym} {r.level:4s}  {r.detail}")

        print("\n" + "=" * 70)
        print(f"  PASS={counts['PASS']}  WARN={counts['WARN']}  FAIL={counts['FAIL']}")
        print("=" * 70)

    fail = counts["FAIL"] > 0 or (args.strict and counts["WARN"] > 0)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
