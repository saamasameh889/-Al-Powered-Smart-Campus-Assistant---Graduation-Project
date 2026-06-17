"""
study_tools_page.py — Academic Study Tools  (6 calculators)
═══════════════════════════════════════════════════════════════
Tools:
  1. GPA Calculator        — weighted GPA from courses + grades
  2. CGPA → Percentage     — converts CGPA to % using 4 formula variants
  3. Credit Hour Tracker   — full degree progress + GPA tracker
  4. Grade Calculator      — required final exam score to hit target
  5. Attendance Calculator — attendance % + missable/needed classes
  6. Task Priority Matrix  — Eisenhower urgency×importance matrix
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

# ── Shared theme helpers ──────────────────────────────────────────────────────

def _card(content: str, color: str = "#8B5CF6") -> None:
    st.markdown(
        f'<div style="background:rgba(255,255,255,.025);border:1px solid {color}33;'
        f'border-top:3px solid {color};border-radius:14px;padding:18px 20px;'
        f'margin-bottom:10px">{content}</div>',
        unsafe_allow_html=True,
    )

def _big_result(value: str, label: str, color: str = "#10B981") -> str:
    return (
        f'<div style="text-align:center;padding:14px 0">'
        f'<div style="font-size:2.4rem;font-weight:800;color:{color}">{value}</div>'
        f'<div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.1em;'
        f'color:{color}88;margin-top:4px">{label}</div></div>'
    )

def _section(title: str) -> None:
    st.markdown(
        f"<div style='font-size:.72rem;font-weight:700;color:#9D77F5;"
        f"text-transform:uppercase;letter-spacing:.1em;margin:16px 0 8px'>"
        f"{title}</div>",
        unsafe_allow_html=True,
    )

# ── Grade scale definitions ───────────────────────────────────────────────────

GRADE_SCALE_40 = {
    "A+": 4.0, "A": 4.0, "A-": 3.7,
    "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7,
    "D+": 1.3, "D": 1.0, "F": 0.0,
}
GRADE_SCALE_50 = {
    "A+": 5.0, "A": 4.75, "A-": 4.5,
    "B+": 4.0, "B":  3.75, "B-": 3.5,
    "C+": 3.0, "C":  2.75, "C-": 2.5,
    "D":  2.0, "F":  0.0,
}

def _gpa_classification(gpa: float, scale: float = 4.0) -> tuple[str, str]:
    ratio = gpa / scale
    if ratio >= 0.925: return "Summa Cum Laude",  "#10B981"
    if ratio >= 0.875: return "Magna Cum Laude",  "#34D399"
    if ratio >= 0.825: return "Cum Laude",        "#6EE7B7"
    if ratio >= 0.750: return "First Class",      "#8B5CF6"
    if ratio >= 0.625: return "Second Class",     "#A78BFA"
    if ratio >= 0.500: return "Pass",             "#F59E0B"
    return "At Risk",  "#EF4444"


# ══════════════════════════════════════════════════════════════════════════════
#  1 — GPA CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def render_gpa_calculator() -> None:
    st.markdown("**Calculate your semester or cumulative GPA from individual course grades.**")

    col_opt1, col_opt2 = st.columns(2)
    scale_name = col_opt1.selectbox("Grading Scale", ["4.0 (US Standard)", "5.0"], key="gpac_scale")
    scale = GRADE_SCALE_40 if scale_name.startswith("4") else GRADE_SCALE_50
    scale_max = 4.0 if scale_name.startswith("4") else 5.0
    grade_options = list(scale.keys())

    # ── Course table ─────────────────────────────────────────────────────────
    _section("Courses")

    if "gpac_courses" not in st.session_state:
        st.session_state["gpac_courses"] = pd.DataFrame([
            {"Course": "Course 1", "Grade": "A",  "Credits": 3},
            {"Course": "Course 2", "Grade": "B+", "Credits": 3},
            {"Course": "Course 3", "Grade": "A-", "Credits": 3},
            {"Course": "Course 4", "Grade": "B",  "Credits": 3},
        ])

    edited = st.data_editor(
        st.session_state["gpac_courses"],
        column_config={
            "Grade":   st.column_config.SelectboxColumn("Grade",   options=grade_options, required=True),
            "Credits": st.column_config.NumberColumn("Credits (hrs)", min_value=1, max_value=6, step=1),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="gpac_table",
    )
    st.session_state["gpac_courses"] = edited

    # ── Calculate ─────────────────────────────────────────────────────────────
    total_qp = total_cr = 0.0
    rows = []
    for _, row in edited.iterrows():
        gp  = scale.get(str(row.get("Grade", "F")), 0.0)
        cr  = float(row.get("Credits", 3))
        qp  = gp * cr
        total_qp += qp
        total_cr += cr
        rows.append({"course": row.get("Course","?"), "gp": gp, "cr": cr, "qp": qp})

    gpa   = total_qp / total_cr if total_cr else 0.0
    label, color = _gpa_classification(gpa, scale_max)

    _section("Results")
    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(_big_result(f"{gpa:.3f}", f"GPA  / {scale_max}", color), unsafe_allow_html=True)
    r2.markdown(_big_result(f"{total_qp:.1f}", "Quality Points", "#8B5CF6"), unsafe_allow_html=True)
    r3.markdown(_big_result(f"{total_cr:.0f}", "Credit Hours", "#6366F1"), unsafe_allow_html=True)
    r4.markdown(_big_result(label, "Classification", color), unsafe_allow_html=True)

    # ── Cumulative GPA projection ─────────────────────────────────────────────
    with st.expander("📊  Cumulative GPA Projection"):
        cc1, cc2 = st.columns(2)
        curr_cgpa = cc1.number_input("Current CGPA", 0.0, scale_max, 3.0, 0.01, key="gpac_curr")
        curr_cred = cc2.number_input("Credits already earned", 0, 240, 60, 1, key="gpac_prev")
        if curr_cred > 0 or total_cr > 0:
            new_total_cr = curr_cred + total_cr
            new_total_qp = curr_cgpa * curr_cred + total_qp
            new_cgpa = new_total_qp / new_total_cr if new_total_cr else gpa
            c_label, c_color = _gpa_classification(new_cgpa, scale_max)
            st.markdown(
                _big_result(f"{new_cgpa:.3f}", f"New Cumulative GPA · {c_label}", c_color),
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  2 — CGPA TO PERCENTAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_cgpa_converter() -> None:
    st.markdown("**Convert your CGPA to percentage using standard university formulas.**")

    c1, c2, c3 = st.columns(3)
    cgpa  = c1.number_input("CGPA", 0.0, 10.0, 8.5, 0.01, key="cgpa_val")
    scale = c2.selectbox("Grading Scale", ["4.0", "5.0", "10.0"], key="cgpa_scale")
    formula = c3.selectbox(
        "Formula",
        ["Standard UGC  (×9.5)", "Anna University  (×10 − 7.5)",
         "VTU  (×10 − 5)", "International  (÷scale × 100)"],
        key="cgpa_formula",
    )

    scale_f = float(scale)

    if formula.startswith("Standard"):
        pct = cgpa * 9.5
        formula_str = f"{cgpa} × 9.5"
    elif formula.startswith("Anna"):
        pct = cgpa * 10 - 7.5
        formula_str = f"({cgpa} × 10) − 7.5"
    elif formula.startswith("VTU"):
        pct = cgpa * 10 - 5
        formula_str = f"({cgpa} × 10) − 5"
    else:
        pct = (cgpa / scale_f) * 100
        formula_str = f"({cgpa} ÷ {scale_f}) × 100"

    pct = max(0.0, min(pct, 100.0))

    _section("Result")
    p1, p2, p3 = st.columns(3)

    pct_color = "#10B981" if pct >= 75 else ("#F59E0B" if pct >= 60 else "#EF4444")
    p1.markdown(_big_result(f"{pct:.2f}%", "Percentage", pct_color), unsafe_allow_html=True)

    # Letter grade
    letter = ("A+" if pct >= 90 else "A" if pct >= 80 else "B+" if pct >= 75
              else "B" if pct >= 70 else "C+" if pct >= 65 else "C" if pct >= 60
              else "D" if pct >= 50 else "F")
    p2.markdown(_big_result(letter, "Letter Grade", "#8B5CF6"), unsafe_allow_html=True)

    # Classification
    cls = ("Distinction" if pct >= 75 else "First Class" if pct >= 60
           else "Second Class" if pct >= 50 else "Pass" if pct >= 40 else "Fail")
    cls_color = "#10B981" if pct >= 75 else ("#8B5CF6" if pct >= 50 else "#EF4444")
    p3.markdown(_big_result(cls, "Classification", cls_color), unsafe_allow_html=True)

    st.caption(f"Formula used: **{formula_str} = {pct:.2f}%**")

    # All formulas comparison table
    with st.expander("📋  Compare All Formulas"):
        data = {
            "Formula": ["UGC (×9.5)", "Anna Univ (×10−7.5)", "VTU (×10−5)", f"International (÷{scale_f}×100)"],
            "Percentage": [
                round(max(0, min(cgpa * 9.5, 100)), 2),
                round(max(0, min(cgpa * 10 - 7.5, 100)), 2),
                round(max(0, min(cgpa * 10 - 5, 100)), 2),
                round(max(0, min((cgpa / scale_f) * 100, 100)), 2),
            ],
        }
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  3 — CREDIT HOUR TRACKER
# ══════════════════════════════════════════════════════════════════════════════

def render_credit_tracker() -> None:
    st.markdown("**Track your full academic record: credits, GPA, and degree progress.**")

    c1, c2, c3 = st.columns(3)
    degree_credits = c1.number_input("Total credits required for degree", 60, 240, 140, 5, key="cht_req")
    degree_type    = c2.selectbox("Degree", ["B.Sc.", "B.Eng.", "B.A.", "B.Tech.", "M.Sc."], key="cht_deg")
    scale_name     = c3.selectbox("Grading Scale", ["4.0", "5.0"], key="cht_scale")
    scale = GRADE_SCALE_40 if scale_name == "4.0" else GRADE_SCALE_50
    scale_max = float(scale_name)

    _section("Course Record")

    if "cht_courses" not in st.session_state:
        st.session_state["cht_courses"] = pd.DataFrame([
            {"Course": "Calculus I",      "Credits": 3, "Grade": "A",  "Category": "Core",     "Semester": "Y1-S1"},
            {"Course": "Programming I",   "Credits": 3, "Grade": "A-", "Category": "Major",    "Semester": "Y1-S1"},
            {"Course": "Linear Algebra",  "Credits": 3, "Grade": "B+", "Category": "Core",     "Semester": "Y1-S2"},
            {"Course": "Data Structures", "Credits": 3, "Grade": "B",  "Category": "Major",    "Semester": "Y2-S1"},
            {"Course": "Elective I",      "Credits": 2, "Grade": "A",  "Category": "Elective", "Semester": "Y2-S1"},
        ])

    cat_options = ["Core", "Major", "Minor", "Elective", "Gen Ed", "Lab", "In Progress"]
    sem_options = [f"Y{y}-S{s}" for y in range(1, 5) for s in range(1, 3)]

    edited = st.data_editor(
        st.session_state["cht_courses"],
        column_config={
            "Grade":    st.column_config.SelectboxColumn("Grade",    options=list(scale.keys()) + ["IP"], required=True),
            "Credits":  st.column_config.NumberColumn("Credits", min_value=1, max_value=6),
            "Category": st.column_config.SelectboxColumn("Category", options=cat_options),
            "Semester": st.column_config.SelectboxColumn("Semester", options=sem_options),
        },
        num_rows="dynamic",
        use_container_width=True,
        key="cht_table",
    )
    st.session_state["cht_courses"] = edited

    # ── Compute ───────────────────────────────────────────────────────────────
    total_qp = total_cr_graded = total_cr_all = 0.0
    for _, row in edited.iterrows():
        g = str(row.get("Grade", "F"))
        cr = float(row.get("Credits", 3))
        total_cr_all += cr
        if g != "IP":
            gp = scale.get(g, 0.0)
            total_qp += gp * cr
            total_cr_graded += cr

    gpa = total_qp / total_cr_graded if total_cr_graded else 0.0
    progress_pct = min(total_cr_all / degree_credits * 100, 100.0)
    label, color = _gpa_classification(gpa, scale_max)

    _section("Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(_big_result(f"{gpa:.3f}", f"GPA / {scale_max}", color), unsafe_allow_html=True)
    s2.markdown(_big_result(f"{total_cr_all:.0f}/{degree_credits:.0f}", "Credits Done/Required", "#8B5CF6"), unsafe_allow_html=True)
    s3.markdown(_big_result(f"{progress_pct:.1f}%", "Degree Progress", "#6366F1"), unsafe_allow_html=True)
    s4.markdown(_big_result(label, "Standing", color), unsafe_allow_html=True)

    st.progress(progress_pct / 100)

    # Category breakdown
    if not edited.empty:
        with st.expander("📊  Credits by Category"):
            cat_summary = edited.groupby("Category")["Credits"].sum().reset_index()
            cat_summary.columns = ["Category", "Credits Completed"]
            st.dataframe(cat_summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  4 — GRADE CALCULATOR  (required score on final)
# ══════════════════════════════════════════════════════════════════════════════

def render_grade_calculator() -> None:
    st.markdown(
        "**Find out what score you need on your final exam (or remaining assignments) "
        "to achieve your target grade.**"
    )

    _section("Your Current Standing")
    g1, g2 = st.columns(2)
    current_grade  = g1.number_input("Current grade (%)",    0.0, 100.0, 72.0, 0.5, key="gc_curr")
    current_weight = g2.number_input("Weight of work so far (%)", 1.0, 100.0, 60.0, 1.0, key="gc_wt")
    target_grade   = g1.number_input("Target final grade (%)", 0.0, 100.0, 80.0, 0.5, key="gc_target")
    final_weight   = g2.number_input("Final exam weight (%)", 1.0, 100.0, 40.0, 1.0, key="gc_fw")

    # Formula: target = current_grade × (current_weight/100) + needed × (final_weight/100)
    # → needed = (target - current_grade × current_weight/100) / (final_weight/100)
    needed = (target_grade - current_grade * (current_weight / 100)) / (final_weight / 100)

    _section("Result")
    r1, r2, r3 = st.columns(3)

    if needed <= 100:
        n_color = "#10B981" if needed <= 70 else ("#F59E0B" if needed <= 85 else "#EF4444")
        verdict = "Achievable ✓" if needed <= 100 else "Not possible"
        r1.markdown(_big_result(f"{needed:.1f}%", "Needed on Final", n_color), unsafe_allow_html=True)
    else:
        r1.markdown(_big_result("Impossible", "Even 100% won't reach target", "#EF4444"), unsafe_allow_html=True)

    # Maximum possible final grade
    max_possible = current_grade * (current_weight / 100) + 100 * (final_weight / 100)
    r2.markdown(_big_result(f"{max_possible:.1f}%", "Max Achievable Grade", "#8B5CF6"), unsafe_allow_html=True)

    # If they score 0 on the final
    min_possible = current_grade * (current_weight / 100)
    r3.markdown(_big_result(f"{min_possible:.1f}%", "Grade if Final = 0%", "#6366F1"), unsafe_allow_html=True)

    # Score range table
    with st.expander("📋  Score → Final Grade Table"):
        scores = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
        rows = []
        for s in scores:
            fg = current_grade * (current_weight / 100) + s * (final_weight / 100)
            rows.append({"Final Exam Score": f"{s}%", "Final Grade": f"{fg:.1f}%"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Multi-assessment mode
    with st.expander("🎯  Multiple Remaining Assessments"):
        st.caption("Add remaining assessments to see what average you need across all of them.")
        if "gc_assessments" not in st.session_state:
            st.session_state["gc_assessments"] = pd.DataFrame([
                {"Assessment": "Final Exam", "Weight (%)": 40},
                {"Assessment": "Lab Report", "Weight (%)": 10},
            ])
        ae = st.data_editor(
            st.session_state["gc_assessments"],
            column_config={"Weight (%)": st.column_config.NumberColumn(min_value=1, max_value=100)},
            num_rows="dynamic", use_container_width=True, key="gc_ae",
        )
        st.session_state["gc_assessments"] = ae
        total_remaining_w = float(ae["Weight (%)"].sum()) if len(ae) > 0 else 0
        if total_remaining_w > 0:
            needed_avg = (target_grade - current_grade * (current_weight / 100)) / (total_remaining_w / 100)
            a_color = "#10B981" if 0 <= needed_avg <= 100 else "#EF4444"
            st.markdown(
                _big_result(
                    f"{needed_avg:.1f}%" if 0 <= needed_avg <= 100 else "Impossible",
                    f"Avg needed across {len(ae)} remaining assessments", a_color
                ),
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  5 — ATTENDANCE CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def render_attendance_calculator() -> None:
    st.markdown("**Track your attendance and find out how many classes you can miss — or need to attend.**")

    _section("Attendance Input")
    a1, a2, a3 = st.columns(3)
    attended  = a1.number_input("Classes attended",      0, 500, 35, 1, key="att_att")
    total     = a2.number_input("Total classes held",    1, 500, 45, 1, key="att_tot")
    required  = a3.selectbox("Required minimum (%)", ["75%","80%","85%","90%","Custom"], key="att_req")

    if required == "Custom":
        req_pct = st.slider("Custom required %", 50, 100, 75, key="att_cust") / 100
    else:
        req_pct = int(required.replace("%","")) / 100

    current_pct = attended / total if total > 0 else 0.0

    # Classes can still miss: solve (attended/(total+x)) >= req → x <= attended/req - total
    can_miss = max(0, int(attended / req_pct - total)) if req_pct > 0 else 999

    # Classes need to attend to recover: solve ((attended+x)/(total+x)) >= req
    # → attended + x >= req*(total+x) → x(1-req) >= req*total - attended → x >= (req*total-attended)/(1-req)
    if current_pct >= req_pct:
        need_attend = 0
    else:
        denom = 1 - req_pct
        need_attend = max(0, int((req_pct * total - attended) / denom) + 1) if denom > 0 else 999

    _section("Status")
    color = "#10B981" if current_pct >= req_pct + 0.05 else (
        "#F59E0B" if current_pct >= req_pct else "#EF4444"
    )
    status = "SAFE ✓" if current_pct >= req_pct + 0.05 else (
        "WARNING ⚠" if current_pct >= req_pct else "DEFICIT ✗"
    )

    s1, s2, s3, s4 = st.columns(4)
    s1.markdown(_big_result(f"{current_pct*100:.1f}%", "Current Attendance", color), unsafe_allow_html=True)
    s2.markdown(_big_result(status, f"vs {req_pct*100:.0f}% required", color), unsafe_allow_html=True)
    s3.markdown(_big_result(str(can_miss), "Classes You Can Still Miss", "#F59E0B"), unsafe_allow_html=True)
    s4.markdown(_big_result(
        str(need_attend) if need_attend > 0 else "—",
        "Consecutive Classes Needed to Recover",
        "#EF4444" if need_attend > 0 else "#10B981",
    ), unsafe_allow_html=True)

    st.progress(min(current_pct, 1.0))
    st.caption(f"{attended} attended / {total} held  ·  Requirement: {req_pct*100:.0f}%")

    # Multi-subject tracker
    with st.expander("📚  Subject-Wise Tracker"):
        if "att_subjects" not in st.session_state:
            st.session_state["att_subjects"] = pd.DataFrame([
                {"Subject": "Mathematics",  "Attended": 20, "Total": 24},
                {"Subject": "Programming",  "Attended": 18, "Total": 24},
                {"Subject": "Physics",      "Attended": 22, "Total": 24},
            ])
        se = st.data_editor(
            st.session_state["att_subjects"],
            column_config={
                "Attended": st.column_config.NumberColumn(min_value=0, max_value=200),
                "Total":    st.column_config.NumberColumn(min_value=1, max_value=200),
            },
            num_rows="dynamic", use_container_width=True, key="att_sub_table",
        )
        st.session_state["att_subjects"] = se
        if not se.empty and "Attended" in se.columns and "Total" in se.columns:
            se = se.copy()
            se["Attendance %"] = (se["Attended"] / se["Total"].replace(0, 1) * 100).round(1)
            se["Status"] = se["Attendance %"].apply(
                lambda x: "✓ Safe" if x >= req_pct*100+5 else ("⚠ Warning" if x >= req_pct*100 else "✗ Deficit")
            )
            st.dataframe(se[["Subject","Attended","Total","Attendance %","Status"]],
                        use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  6 — TASK PRIORITY MATRIX  (Eisenhower)
# ══════════════════════════════════════════════════════════════════════════════

QUADRANTS = {
    "do_first":  {"label": "DO FIRST",  "icon": "🔴", "desc": "Urgent + Important",      "color": "#EF4444"},
    "schedule":  {"label": "SCHEDULE",  "icon": "🟡", "desc": "Important, Not Urgent",    "color": "#F59E0B"},
    "delegate":  {"label": "DELEGATE",  "icon": "🔵", "desc": "Urgent, Not Important",    "color": "#3B82F6"},
    "eliminate": {"label": "ELIMINATE", "icon": "⚪", "desc": "Not Urgent + Not Important","color": "#6B7280"},
}

def render_task_matrix() -> None:
    st.markdown(
        "**Organise tasks by urgency × importance. "
        "Each quadrant tells you exactly what to do with the task.**"
    )

    # ── Add task ─────────────────────────────────────────────────────────────
    _section("Add a Task")
    t1, t2, t3 = st.columns([3, 2, 1])
    new_task     = t1.text_input("Task name", placeholder="e.g. Study for midterm", key="tpm_name")
    new_quadrant = t2.selectbox(
        "Quadrant",
        options=list(QUADRANTS.keys()),
        format_func=lambda k: f"{QUADRANTS[k]['icon']} {QUADRANTS[k]['label']} — {QUADRANTS[k]['desc']}",
        key="tpm_quad",
    )
    add_btn = t3.button("Add ➕", use_container_width=True, key="tpm_add")

    if add_btn and new_task.strip():
        key = f"tpm_tasks_{new_quadrant}"
        if key not in st.session_state:
            st.session_state[key] = []
        st.session_state[key].append(new_task.strip())

    # ── Matrix display ────────────────────────────────────────────────────────
    _section("Priority Matrix")
    q_keys = list(QUADRANTS.keys())
    row1 = st.columns(2)
    row2 = st.columns(2)
    cols = [row1[0], row1[1], row2[0], row2[1]]

    for col, qk in zip(cols, q_keys):
        q     = QUADRANTS[qk]
        tasks = st.session_state.get(f"tpm_tasks_{qk}", [])
        c     = q["color"]

        task_html = ""
        for t in tasks:
            task_html += (
                f'<div style="background:rgba(255,255,255,.04);border-left:3px solid {c};'
                f'border-radius:0 8px 8px 0;padding:6px 10px;margin:4px 0;'
                f'font-size:.8rem;color:#EDE9FE">{t}</div>'
            )
        if not task_html:
            task_html = '<div style="font-size:.74rem;color:#3D3060;padding:6px 0">No tasks yet</div>'

        with col:
            st.markdown(
                f'<div style="background:rgba(255,255,255,.02);border:1px solid {c}44;'
                f'border-top:3px solid {c};border-radius:14px;padding:14px 16px;min-height:160px">'
                f'<div style="font-size:.65rem;font-weight:700;color:{c};'
                f'text-transform:uppercase;letter-spacing:.1em">'
                f'{q["icon"]} {q["label"]}</div>'
                f'<div style="font-size:.68rem;color:#5B4D8A;margin-bottom:10px">{q["desc"]}</div>'
                f'{task_html}'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Clear button per quadrant
            if tasks:
                if st.button(f"Clear {q['label']}", key=f"tpm_clear_{qk}", use_container_width=True):
                    st.session_state[f"tpm_tasks_{qk}"] = []
                    st.rerun()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_tasks = sum(len(st.session_state.get(f"tpm_tasks_{k}", [])) for k in q_keys)
    if total_tasks > 0:
        _section("Summary")
        sm_cols = st.columns(4)
        for i, (qk, sm_col) in enumerate(zip(q_keys, sm_cols)):
            q     = QUADRANTS[qk]
            count = len(st.session_state.get(f"tpm_tasks_{qk}", []))
            sm_col.markdown(
                _big_result(str(count), f"{q['icon']} {q['label']}", q["color"]),
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def render_study_tools() -> None:
    """Renders all 6 tools in named sub-tabs."""
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🧮 GPA Calculator",
        "📊 CGPA → %",
        "🎓 Credit Hours",
        "📝 Grade Calculator",
        "📅 Attendance",
        "✅ Task Matrix",
    ])
    with t1: render_gpa_calculator()
    with t2: render_cgpa_converter()
    with t3: render_credit_tracker()
    with t4: render_grade_calculator()
    with t5: render_attendance_calculator()
    with t6: render_task_matrix()
