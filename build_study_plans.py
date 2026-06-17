#!/usr/bin/env python3
"""
build_study_plans.py
Extract semester-by-semester study plans from Zewail curricula JSONL.

Output: data/curriculum/study_plans.json

Schema:
  plans.<SCHOOL>.<PROG>.tracks.<TRACK>.semesters.<YnSm> = {
      year, semester, seq, courses: [{code, name, credits}], total_credits
  }
  TRACK="_common" means courses shared by all concentrations in that year.
  TRACK="_foundation" NOT used; Foundation Year stored under SCI/FY/_common.
"""
from __future__ import annotations
import json, re, copy
from pathlib import Path
from collections import defaultdict

JSONL_PATH = Path(r"d:\FINAL_PROJECT_v3\cloned_repo\data\clean\cleaned_documents.jsonl")
COURSES_F  = Path(r"d:\FINAL_PROJECT_v3\cloned_repo\data\curriculum\courses.json")
OUTPUT     = Path(r"d:\FINAL_PROJECT_v3\cloned_repo\data\curriculum\study_plans.json")

CURRICULA_SOURCES = {
    "CSAI - Curricula 2022.pdf": "CSAI",
    "BUS - Curricula 2023.pdf":  "BUS",
    "SCI Curricula 2023.pdf":    "SCI",
}

# ── Programme detection ───────────────────────────────────────────────────────
PROG_MAP = {
    "CSAI": [
        (re.compile(r"SOFTWARE DEVELOPMENT",                             re.I), "SWD",   "Software Development"),
        (re.compile(r"DATA SCIENCE AND ARTIFICIAL INTELLIGENCE",         re.I), "DSAI",  "Data Science and Artificial Intelligence"),
        (re.compile(r"\bINFORMATION TECHNOLOGY\b",                       re.I), "IT",    "Information Technology"),
    ],
    "BUS": [
        (re.compile(r"ACTUARIAL ANALYSIS AND RISK MANAGEMENT",           re.I), "AARM",  "Actuarial Analysis and Risk Management"),
        (re.compile(r"FINANCE AND INVESTMENT MANAGEMENT",                re.I), "FIM",   "Finance and Investment Management"),
        (re.compile(r"MARKETING.*ENTREPRENEURSHIP.*INNOVATION|MEIM\b",  re.I), "MEIM",  "Marketing, Entrepreneurship and Innovation Management"),
        (re.compile(r"OPERATIONS.*SUPPLY CHAIN|OSCTM\b",                re.I), "OSCTM", "Operations, Supply Chain and Technology Management"),
    ],
    "SCI": [
        (re.compile(r"\bBIOMEDICAL SCIENCES?\b",                        re.I), "BMS",   "Biomedical Sciences"),
        (re.compile(r"\bNANO\s*SCIENCE\b|\bNANOSCIENCE\b|\bNANOSC\b", re.I), "NANO",  "Nanoscience"),
        (re.compile(r"\bPHYSICS OF THE UNIVERSE\b|\bPHYSICS OF UNIVERSE\b", re.I), "PHY", "Physics of the Universe"),
        (re.compile(r"\bFOUNDATION YEAR\b",                             re.I), "FY",    "Foundation Year"),
    ],
}

TRACK_MAP = {
    "SWD": [
        (re.compile(r"\bAPD\b|APPLICATION DEVELOPMENT",        re.I), "APD",   "Application Development"),
        (re.compile(r"\bGCG\b|GAMING AND COMPUTER GRAPHICS",   re.I), "GCG",   "Gaming and Computer Graphics"),
        (re.compile(r"\bHCI\b|HUMAN COMPUTER INTERACTION",     re.I), "HCI",   "Human Computer Interaction"),
    ],
    "IT": [
        (re.compile(r"\bITNS\b|NETWORKS.*SECURITY.*GOVERNANCE",re.I), "ITNS",  "Networks, Security and Governance"),
        (re.compile(r"\bITCC\b|INFRASTRUCTURE.*CLOUD",         re.I), "ITCC",  "Infrastructure and Cloud Computing"),
    ],
    "BMS": [
        (re.compile(r"COMPUTATIONAL BIOLOGY.*GENOMICS",        re.I), "CBG",   "Computational Biology and Genomics"),
        (re.compile(r"MOLECULAR CELL BIOLOGY\b",               re.I), "MCB",   "Molecular Cell Biology"),
        (re.compile(r"\bMEDICAL SCIENCES?\b|\bMEDICAL SCIENCE\b", re.I), "MED", "Medical Sciences"),
        (re.compile(r"DRUG DESIGN.*DEVELOPMENT|\bDDD\b",       re.I), "DDD",   "Drug Design and Development"),
    ],
    "NANO": [
        (re.compile(r"NANOPHYSICS",           re.I), "NPHY",   "Nanophysics"),
        (re.compile(r"NANOCHEMISTRY",         re.I), "NCHEM",  "Nanochemistry"),
        (re.compile(r"BIO.NANOTECHNOLOGY",    re.I), "BIONANO","Bio-Nanotechnology"),
        (re.compile(r"NANO\s*MEDICINE",       re.I), "NMED",   "Nanomedicine"),
    ],
    "PHY": [
        (re.compile(r"ASTROPHYSICS",          re.I), "AST",    "Astrophysics"),
        (re.compile(r"HIGH\s*ENERGY\s*PHYSICS",re.I), "HEP",  "High Energy Physics"),
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_formatted(text: str) -> bool:
    return bool(re.match(r".+\(page\s+\d+\)\s*:", text.strip().split("\n")[0]))


def detect_prog(text: str, school: str):
    for pat, code, name in PROG_MAP.get(school, []):
        if pat.search(text):
            return code, name
    return None, None


def detect_track(text: str, prog: str):
    for pat, code, name in TRACK_MAP.get(prog, []):
        if pat.search(text):
            return code, name
    return None, None


def sem_to_seq(year: int, sem) -> int:
    return (year - 1) * 2 + sem if isinstance(sem, int) else 0


def parse_sem_header(line: str):
    """
    Returns (year, sem|'summer', is_summer, track_hint|None) or None.

    IMPORTANT: specific patterns are checked BEFORE generic patterns to avoid
    a generic match stealing the track hint from a more specific header.
    """
    s = line.strip()

    # ── Foundation Year Semester N ──────────────────────────────────────────
    m = re.search(r"Foundation\s+Year\s+Semester\s+(\d+)", s, re.I)
    if m:
        return 1, int(m.group(1)), False, None   # stored in FY/_common

    # ── BMS track-specific: BMS / TRACK / YEAR N / SEMESTER N ──────────────
    # Must be checked BEFORE generic "Year N / Semester N"
    m = re.search(r"BMS\s*/\s*(.+?)\s*/\s*YEAR\s+(\d+)\s*/\s*SEMESTER\s+(\d+)", s, re.I)
    if m:
        return int(m.group(2)), int(m.group(3)), False, m.group(1).strip()

    # ── BMS YEAR N / SEMESTER 3 (SUMMER) — common to all concentrations ────
    m = re.search(r"\bBMS\s+YEAR\s+(\d+)\s*/\s*SEMESTER\s+3\s*\(SUMMER\)", s, re.I)
    if m:
        return int(m.group(1)), "summer", True, "_bms_common"

    # ── BMS YEAR N / SEMESTER N — common to all concentrations ─────────────
    m = re.search(r"\bBMS\s+YEAR\s+(\d+)\s*/\s*SEMESTER\s+(\d+)(?:\s*\((.+?)\))?", s, re.I)
    if m:
        hint_in_parens = (m.group(3) or "").strip()
        # Plain "BMS YEAR N / SEMESTER N" = common; "BMS YEAR N / SEMESTER N (track)" = track-specific
        return int(m.group(1)), int(m.group(2)), False, hint_in_parens or "_bms_common"

    # ── NANO track-specific summer: NANOMEDICINE YEAR N / SEMESTER 3 (SUMMER) ─
    m = re.search(
        r"(NANOMEDICINE|NANOPHYSICS|NANOCHEMISTRY|BIO.NANOTECHNOLOGY)\s+YEAR\s+(\d+)"
        r"\s*/\s*SEMESTER\s+3\s*\(SUMMER\)",
        s, re.I
    )
    if m:
        return int(m.group(2)), "summer", True, m.group(1).strip()

    # ── NANO track-specific: NANOMEDICINE/etc. YEAR N / SEMESTER N ──────────
    m = re.search(
        r"(NANOMEDICINE|NANOPHYSICS|NANOCHEMISTRY|BIO.NANOTECHNOLOGY)\s+YEAR\s+(\d+)"
        r"\s*/\s*SEMESTER\s+(\d+)",
        s, re.I
    )
    if m:
        return int(m.group(2)), int(m.group(3)), False, m.group(1).strip()

    # ── NANO Y4 concentration courses: "NANO MEDICINE CONCENTRATION COURSES/ SEMESTER N"
    m = re.search(
        r"(NANO\s*MEDICINE|NANOPHYSICS|NANOCHEMISTRY|BIO.NANOTECHNOLOGY)"
        r"\s+CONCENTRATION\s+COURSES\s*/\s*SEMESTER\s+(\d+)",
        s, re.I
    )
    if m:
        return 4, int(m.group(2)), False, m.group(1).strip()

    # ── NANOSC YEAR N / SEMESTER N — common Y2 data (shared by all concentrations) ─
    m = re.search(r"NANOSC\s+YEAR\s+(\d+)\s*/\s*SEMESTER\s+(\d+)(?:\s*\((.+?)\))?", s, re.I)
    if m:
        # Store in _common regardless of any parenthetical track note
        return int(m.group(1)), int(m.group(2)), False, "_nano_common"

    # ── PU / TRACK / YEAR N / SEMESTER N (spaces or no spaces around /) ────
    m = re.search(r"PU\s*/\s*(.+?)\s*/\s*YEAR\s+(\d+)\s*/\s*SEMESTER\s+(\d+)", s, re.I)
    if m:
        return int(m.group(2)), int(m.group(3)), False, m.group(1).strip()

    # ── PU YEAR N / SEMESTER 3 / SUMMER ─────────────────────────────────────
    m = re.search(r"\bPU\s+YEAR\s+(\d+)\s*/\s*SEMESTER\s+3\s*/\s*SUMMER", s, re.I)
    if m:
        return int(m.group(1)), "summer", True, None

    # ── PU YEAR N / SEMESTER N — common PHY data ────────────────────────────
    m = re.search(r"\bPU\s+YEAR\s+(\d+)\s*/\s*SEMESTER\s+(\d+)", s, re.I)
    if m:
        return int(m.group(1)), int(m.group(2)), False, "_phy_common"

    # ── Generic summer: "Year N / (Summer)" ─────────────────────────────────
    # Must be BEFORE the generic Year N / Semester N check
    m = re.search(r"Year\s+(\d+)\s*/\s*(?:Semester\s+\d+\s*\()?Summer\)?", s, re.I)
    if m:
        return int(m.group(1)), "summer", True, None

    # ── Generic: Year N / Semester N (CSAI / BUS) ───────────────────────────
    m = re.search(r"Year\s+(\d+)\s*/\s*Semester\s+(\d+)", s, re.I)
    if m:
        return int(m.group(1)), int(m.group(2)), False, None

    return None


_COURSE_CODE_RE = re.compile(r"^([A-Z]{2,6})\s+(\d{3,4}[A-Z]?)\s*\|")
_SKIP_RE        = re.compile(r"^(TOTAL|COURSE\s+CODE|Course\s+Code|Cr\b)", re.I)


def parse_course_row(row: str):
    row = row.strip()
    if not row or _SKIP_RE.match(row):
        return None
    m = _COURSE_CODE_RE.match(row)
    if not m:
        return None
    code   = f"{m.group(1)} {m.group(2)}"
    parts  = row.split("|")
    name   = parts[1].strip() if len(parts) > 1 else ""
    credits = None
    if len(parts) > 2:
        try:
            cr = parts[2].strip()
            if cr and cr not in ("-", ""):
                credits = int(cr)
        except ValueError:
            pass
    return {"code": code, "name": name, "credits": credits}


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_school(src_chunks: list, school: str) -> dict:
    # Sort: raw chunks before formatted on the same page
    src_chunks.sort(key=lambda c: (
        int(c.get("page", 0)) if str(c.get("page", "")).isdigit() else 0,
        1 if is_formatted(c.get("text", "")) else 0,
    ))

    programmes: dict = {}

    cur_prog  = None
    cur_pname = None
    cur_track = None   # None → "_common"
    cur_tname = None

    # Programme triggers (must be more specific to avoid false positives)
    prog_triggers = (
        "SAMPLE STUDY PLAN FOR",   # "SAMPLE STUDY PLAN FOR B. SC. IN ..."
        "- YEAR 1 (", "- YEAR 2 (", "- YEAR 3 (", "- YEAR 4 (",
        "PROGRAM - YEAR", "PROGRAMME - YEAR",
    )

    for chunk in src_chunks:
        text = chunk.get("text", "")

        if not is_formatted(text):
            # ── Raw text chunk: detect programme / track context ───────────
            upper = text.upper()

            # Skip table of contents pages — they list all section titles and
            # would falsely match "SAMPLE STUDY PLAN FOR" for every programme.
            if "TABLE OF CONTENTS" in upper:
                continue

            # General programme detection triggers
            if any(t in upper for t in prog_triggers):
                pc, pn = detect_prog(text, school)
                if pc:
                    if pc != cur_prog:
                        cur_prog  = pc
                        cur_pname = pn
                        cur_track = None
                        cur_tname = None
                    if "SAMPLE STUDY PLAN FOR" in upper:
                        cur_track = None
                        cur_tname = None

            # Track detection (only when in a programme)
            if cur_prog:
                track_triggers_general = ("CONCENTRATION", "PROGRAM -", "PROGRAMME -")
                triggered = any(t in upper for t in track_triggers_general)

                # NANO/PHY: also detect from study-plan section headers in RAW
                # (these pages don't use the word "CONCENTRATION")
                # Use specific "TRACKNAME YEAR N" patterns to avoid false positives
                # from description text that merely lists concentration names.
                if not triggered and cur_prog in ("NANO", "PHY"):
                    nano_phy_section_pats = [
                        re.compile(r"(NANOMEDICINE|NANO\s+MEDICINE)\s+YEAR\s+\d+",  re.I),
                        re.compile(r"NANOPHYSICS\s+YEAR\s+\d+",                      re.I),
                        re.compile(r"NANOCHEMISTRY\s+YEAR\s+\d+",                    re.I),
                        re.compile(r"BIO.NANOTECHNOLOGY\s+YEAR\s+\d+",               re.I),
                        re.compile(r"ASTROPHYSICS\s+YEAR\s+\d+",                     re.I),
                        re.compile(r"HIGH\s*ENERGY\s*PHYSICS\s+YEAR\s+\d+",          re.I),
                    ]
                    triggered = any(p.search(text) for p in nano_phy_section_pats)

                if triggered:
                    tc, tn = detect_track(text, cur_prog)
                    if tc:
                        cur_track = tc
                        cur_tname = tn

        else:
            # ── Formatted pipe-table chunk: parse semesters ───────────────
            lines = text.strip().split("\n")[1:]  # strip "Source (page N):" header

            # SCI Foundation Year: if no programme yet, check for Foundation Year header
            if cur_prog is None and school == "SCI":
                for ln in lines:
                    if re.search(r"Foundation\s+Year\s+Semester", ln, re.I):
                        cur_prog  = "FY"
                        cur_pname = "Foundation Year"
                        cur_track = None
                        cur_tname = None
                        break

            seg_year   = None
            seg_sem    = None
            seg_summer = False
            seg_track  = cur_track   # snapshot at start of chunk
            seg_tname  = cur_tname
            seg_courses: list = []

            # "_*_common" sentinel hints: force into _common regardless of cur_track
            _COMMON_HINTS = {"_bms_common", "_nano_common", "_phy_common"}

            def flush(y, s, summer, track, tname, courses):
                if y is None or not courses:
                    return
                _store(programmes, school,
                       cur_prog, cur_pname,
                       track, tname,
                       y, s, summer, courses)

            for line in lines:
                parsed = parse_sem_header(line)
                if parsed:
                    flush(seg_year, seg_sem, seg_summer,
                          seg_track, seg_tname, seg_courses)

                    seg_year, seg_sem, seg_summer, hint = parsed
                    seg_courses = []

                    if hint in _COMMON_HINTS:
                        # Force common: clear track so courses go to _common
                        seg_track = None
                        seg_tname = None
                    elif hint and cur_prog:
                        tc, tn = detect_track(hint, cur_prog)
                        if tc:
                            seg_track = tc
                            seg_tname = tn
                            cur_track = tc
                            cur_tname = tn
                        else:
                            seg_track = cur_track
                            seg_tname = cur_tname
                    else:
                        seg_track = cur_track
                        seg_tname = cur_tname
                    continue

                if seg_year is None:
                    continue

                course = parse_course_row(line)
                if course:
                    seg_courses.append(course)

            flush(seg_year, seg_sem, seg_summer,
                  seg_track, seg_tname, seg_courses)

    return programmes


def _store(programmes, school, prog, pname, track, tname, year, sem, is_summer, courses):
    if not prog or not courses:
        return

    sem_key = f"Y{year}Summer" if is_summer else f"Y{year}S{sem}"
    seq     = 0 if is_summer else sem_to_seq(year, sem)
    tk      = track or "_common"
    tn      = tname or tk

    p = programmes.setdefault(prog, {"full_name": pname or prog, "tracks": {}})
    t = p["tracks"].setdefault(tk, {"full_name": tn, "semesters": {}})

    if sem_key in t["semesters"]:
        existing = t["semesters"][sem_key]["courses"]
        existing_codes = {c["code"] for c in existing}
        added = 0
        for c in courses:
            if c["code"] not in existing_codes:
                existing.append(c)
                existing_codes.add(c["code"])
                added += 1
        if added:
            t["semesters"][sem_key]["total_credits"] = sum(
                c.get("credits") or 0 for c in existing
            )
    else:
        t["semesters"][sem_key] = {
            "year":          year,
            "semester":      "summer" if is_summer else sem,
            "seq":           seq,
            "courses":       list(courses),
            "total_credits": sum(c.get("credits") or 0 for c in courses),
        }


# ── Post-processing ───────────────────────────────────────────────────────────

def _copy_semesters(src_track_data: dict, dst_track_data: dict, only_missing: bool = True):
    for sem_key, sdata in src_track_data.get("semesters", {}).items():
        if only_missing and sem_key in dst_track_data.get("semesters", {}):
            continue
        dst_track_data.setdefault("semesters", {})[sem_key] = copy.deepcopy(sdata)


def post_process(plans: dict):
    """
    Fix structural issues discovered after extraction:
    1. CSAI/SWD: move Y1-Y2 from HCI track to _common
    2. CSAI/DSAI and IT: add missing Y1S1 from SWD/_common
    3. BUS: propagate AARM Y1-Y2 to FIM, MEIM, OSCTM
    4. SCI/FY: distribute Foundation Year (Y1S1/Y1S2) to BMS, NANO, PHY _common
    5. SCI/BMS: ensure MED has Y2S2 (same as MCB)
    """

    # ── 1. CSAI/SWD: Y1-Y2 in HCI → move to _common ─────────────────────────
    csai = plans.get("CSAI", {})
    swd  = csai.get("SWD", {}).get("tracks", {})

    hci_sems    = swd.get("HCI", {}).get("semesters", {})
    common_sems = swd.setdefault("_common", {"full_name": "Common (all tracks)", "semesters": {}}).get("semesters", {})

    early_keys  = {"Y1S1", "Y1S2", "Y2S1", "Y2S2"}
    to_move: dict = {}
    to_remove: list = []

    for sk, sd in hci_sems.items():
        if sk in early_keys:
            to_move[sk] = sd
            to_remove.append(sk)

    for sk in to_remove:
        del hci_sems[sk]

    for sk, sd in to_move.items():
        if sk not in common_sems:
            common_sems[sk] = copy.deepcopy(sd)

    swd["_common"] = {"full_name": "Common (all tracks)", "semesters": common_sems}

    # ── 2. CSAI/DSAI & IT: add Y1S1 from SWD/_common ────────────────────────
    swd_common = swd.get("_common", {}).get("semesters", {})
    if "Y1S1" in swd_common:
        for prog_code in ("DSAI", "IT"):
            prog = csai.get(prog_code, {})
            prog_tracks = prog.get("tracks", {})
            common = prog_tracks.setdefault("_common", {"full_name": "Common", "semesters": {}})
            if "Y1S1" not in common.get("semesters", {}):
                common.setdefault("semesters", {})["Y1S1"] = copy.deepcopy(swd_common["Y1S1"])

    # ── 3. BUS: propagate AARM Y1-Y2 to FIM, MEIM, OSCTM ───────────────────
    bus  = plans.get("BUS", {})
    aarm = bus.get("AARM", {}).get("tracks", {}).get("_common", {})

    aarm_early = {
        sk: sd for sk, sd in aarm.get("semesters", {}).items()
        if sd.get("year", 9) <= 2
    }

    for prog_code in ("FIM", "MEIM", "OSCTM"):
        prog = bus.get(prog_code)
        if prog is None:
            bus[prog_code] = {
                "full_name": {
                    "FIM":   "Finance and Investment Management",
                    "MEIM":  "Marketing, Entrepreneurship and Innovation Management",
                    "OSCTM": "Operations, Supply Chain and Technology Management",
                }.get(prog_code, prog_code),
                "tracks": {
                    "_common": {
                        "full_name": "Common (shared with AARM Y1-Y2)",
                        "semesters": {sk: copy.deepcopy(sd) for sk, sd in aarm_early.items()},
                    }
                }
            }
        else:
            common = prog["tracks"].setdefault("_common", {"full_name": "Common", "semesters": {}})
            for sk, sd in aarm_early.items():
                if sk not in common.get("semesters", {}):
                    common.setdefault("semesters", {})[sk] = copy.deepcopy(sd)

    # ── 4. SCI: distribute Foundation Year (FY/_common Y1) to BMS, NANO, PHY ─
    sci = plans.get("SCI", {})
    fy_common = sci.get("FY", {}).get("tracks", {}).get("_common", {})
    fy_y1_sems = {
        sk: sd for sk, sd in fy_common.get("semesters", {}).items()
        if sd.get("year", 9) == 1
    }

    # Fallback: if FY wasn't detected, get Y1 from BMS/_common (old extraction)
    if not fy_y1_sems:
        bms_cm = sci.get("BMS", {}).get("tracks", {}).get("_common", {})
        fy_y1_sems = {
            sk: sd for sk, sd in bms_cm.get("semesters", {}).items()
            if sd.get("year", 9) == 1
        }
        # Remove Y1 from BMS/_common if we found them there (they belong to FY)
        for sk in list(fy_y1_sems):
            bms_cm.get("semesters", {}).pop(sk, None)

    if fy_y1_sems:
        for prog_code in ("BMS", "NANO", "PHY"):
            prog = sci.get(prog_code)
            if prog is None:
                continue
            common = prog["tracks"].setdefault(
                "_common", {"full_name": "Common (all concentrations)", "semesters": {}}
            )
            for sk, sd in fy_y1_sems.items():
                if sk not in common.get("semesters", {}):
                    common.setdefault("semesters", {})[sk] = copy.deepcopy(sd)
        # Also put Foundation Year in its own entry in SCI
        if "FY" not in sci:
            sci["FY"] = {
                "full_name": "Foundation Year",
                "tracks": {"_common": {"full_name": "Foundation Year", "semesters": copy.deepcopy(fy_y1_sems)}},
            }

    # ── 5. SCI/NANO: distribute _common Y2 to all NANO tracks (if missing) ──
    nano = sci.get("NANO", {})
    nano_common = nano.get("tracks", {}).get("_common", {})
    nano_y2_sems = {
        sk: sd for sk, sd in nano_common.get("semesters", {}).items()
        if sd.get("year", 9) == 2
    }
    if nano_y2_sems:
        for track, tdata in nano.get("tracks", {}).items():
            if track == "_common":
                continue
            for sk, sd in nano_y2_sems.items():
                if sk not in tdata.get("semesters", {}):
                    tdata.setdefault("semesters", {})[sk] = copy.deepcopy(sd)

    # ── 6. SCI/BMS: ensure MED/Y2S2 exists (same concentration courses as MCB) ─
    bms_tracks = sci.get("BMS", {}).get("tracks", {})
    mcb_sems   = bms_tracks.get("MCB", {}).get("semesters", {})
    if "Y2S2" in mcb_sems and "MED" in bms_tracks:
        med_sems = bms_tracks["MED"].setdefault("semesters", {})
        if "Y2S2" not in med_sems:
            med_sems["Y2S2"] = copy.deepcopy(mcb_sems["Y2S2"])

    # ── 7. SCI/PHY: distribute _common Y2-Y3 to both PHY tracks ─────────────
    phy = sci.get("PHY", {})
    phy_common = phy.get("tracks", {}).get("_common", {})
    phy_early = {
        sk: sd for sk, sd in phy_common.get("semesters", {}).items()
    }
    if phy_early:
        for track, tdata in phy.get("tracks", {}).items():
            if track == "_common":
                continue
            for sk, sd in phy_early.items():
                if sk not in tdata.get("semesters", {}):
                    tdata.setdefault("semesters", {})[sk] = copy.deepcopy(sd)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(plans: dict, courses: dict) -> dict:
    known = set(courses.keys())
    all_codes: set = set()
    missing:   set = set()

    for school, progs in plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                for sem_key, sdata in tdata.get("semesters", {}).items():
                    for c in sdata.get("courses", []):
                        all_codes.add(c["code"])
                        if c["code"] not in known:
                            missing.add(c["code"])

    summary = {}
    for school, progs in plans.items():
        entry = {}
        for prog, pdata in progs.items():
            tracks_info = {}
            for track, tdata in pdata.get("tracks", {}).items():
                sems = sorted(tdata.get("semesters", {}).keys())
                tracks_info[track] = sems
            entry[prog] = tracks_info
        summary[school] = entry

    return {
        "extracted_structure": summary,
        "total_unique_course_codes": len(all_codes),
        "codes_in_courses_json":     len(all_codes - missing),
        "codes_not_in_courses_json": len(missing),
        "missing_codes_sample":      sorted(missing)[:30],
        "notes": [
            "ENGR curricula PDF not available — no ENGR plans",
            "SCI/FY/_common contains Foundation Year Y1S1/Y1S2 (shared by all SCI programmes)",
            "BUS FIM/MEIM/OSCTM: Y1-Y2 propagated from AARM (shared school requirements)",
            "CSAI DSAI/IT: Y1S1 propagated from SWD (same Foundation)",
            "BMS MED/Y2S2 copied from MCB (same courses for both concentrations in Y2)",
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    chunks: list = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                try:
                    chunks.append(json.loads(raw))
                except Exception:
                    pass
    print(f"Loaded {len(chunks)} chunks")

    courses: dict = {}
    if COURSES_F.exists():
        with open(COURSES_F, encoding="utf-8") as f:
            courses = json.load(f).get("courses", {})
    print(f"Loaded {len(courses)} known course codes")

    by_src: dict = defaultdict(list)
    for c in chunks:
        if c.get("source") in CURRICULA_SOURCES:
            by_src[c["source"]].append(c)

    all_plans: dict = {}
    for src, school in CURRICULA_SOURCES.items():
        src_chunks = by_src.get(src, [])
        print(f"\n--- {school} ({len(src_chunks)} chunks) ---")
        progs = extract_school(src_chunks, school)
        all_plans[school] = progs
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                sems = sorted(tdata.get("semesters", {}).keys())
                n = sum(len(tdata["semesters"][s]["courses"]) for s in sems)
                print(f"  {prog}/{track}: {sems}  ({n} courses)")

    post_process(all_plans)
    print("\n--- After post-processing ---")
    for school, progs in all_plans.items():
        for prog, pdata in progs.items():
            for track, tdata in pdata.get("tracks", {}).items():
                sems = sorted(tdata.get("semesters", {}).keys())
                n = sum(len(tdata["semesters"][s]["courses"]) for s in sems)
                print(f"  {school}/{prog}/{track}: {sems}  ({n} courses)")

    val = validate(all_plans, courses)
    print(f"\nValidation: {val['codes_in_courses_json']}/{val['total_unique_course_codes']} codes in courses.json")

    output = {"metadata": val, "plans": all_plans}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWritten to {str(OUTPUT)}")


if __name__ == "__main__":
    main()
