"""
admin_dashboard.py — Zewail City Campus Assistant · Admin Panel
═══════════════════════════════════════════════════════════════════════
Password-protected Streamlit admin dashboard for:
  • Monitoring query logs and feedback (Feature 5.6)
  • Running AI-powered gap analysis
  • Managing system-prompt patches (fine-tuning via rules)
  • Downloading full query log

Run separately from the main app:
    streamlit run admin_dashboard.py --server.port 8502

Set ADMIN_PASSWORD in your .env file.  Default: "zc-admin-2025"
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Allow same project-root imports as main app
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_D_PACKAGES = r"D:\py311_packages"
if _D_PACKAGES not in sys.path:
    sys.path.insert(0, _D_PACKAGES)

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ZC Assistant — Admin",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Auth ───────────────────────────────────────────────────────────────────────
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "zc-admin-2025")

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.admin-header {
    background: linear-gradient(135deg, #1a0e35 0%, #0f1e35 100%);
    border-bottom: 2px solid #7C3AED;
    padding: 18px 28px; border-radius: 12px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
}
.admin-title { color: #C4B5FD; font-size: 1.3rem; font-weight: 700; }
.admin-sub   { color: #5B4D8A; font-size: .78rem; }
.stat-box {
    background: #120d24; border: 1px solid #2d1f5e;
    border-radius: 12px; padding: 18px 20px; text-align: center;
}
.stat-val { font-size: 2rem; font-weight: 700; color: #A78BFA; }
.stat-lbl { font-size: .72rem; color: #5B4D8A; text-transform: uppercase;
            letter-spacing: .1em; margin-top: 4px; }
.flag-row {
    background: rgba(239,68,68,.06); border: 1px solid rgba(239,68,68,.2);
    border-left: 3px solid #EF4444; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0; font-size: .82rem;
}
.patch-row {
    background: rgba(16,185,129,.06); border: 1px solid rgba(16,185,129,.2);
    border-left: 3px solid #10B981; border-radius: 8px;
    padding: 10px 14px; margin: 6px 0; font-size: .82rem; color: #D1FAE5;
}
.patch-id  { font-size:.65rem; color:#6EE7B7; font-weight:700; }
.patch-lbl { font-weight:600; color:#34D399; }
.q-text { color: #DDD6FE; font-weight: 600; }
.q-meta { color: #5B4D8A; font-size: .72rem; }
.score-ok  { color: #34D399; font-weight:700; }
.score-bad { color: #F87171; font-weight:700; }
.thumb-up  { color: #34D399; }
.thumb-dn  { color: #F87171; }
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


def _login_gate() -> bool:
    if st.session_state.get("admin_authed"):
        return True
    st.markdown("""
    <div class="admin-header">
        <div style="font-size:2rem">⚙️</div>
        <div>
            <div class="admin-title">Campus Assistant — Admin Panel</div>
            <div class="admin-sub">Zewail City of Science and Technology</div>
        </div>
    </div>""", unsafe_allow_html=True)
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("### 🔐 Admin Login")
        pwd = st.text_input("Password", type="password", placeholder="Enter admin password")
        if st.button("Login", use_container_width=True, type="primary"):
            if pwd == ADMIN_PASSWORD:
                st.session_state["admin_authed"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False


if not _login_gate():
    st.stop()

# ── Imports (only after auth) ──────────────────────────────────────────────────
from feedback_logger import (
    get_all_entries, get_flagged_entries, get_stats,
    update_thumb, run_ai_gap_analysis,
    load_prompt_patches, save_prompt_patch, delete_prompt_patch,
    LOG_FILE,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# ── Header ─────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([5, 1])
with col_h1:
    st.markdown("""
    <div class="admin-header">
        <div style="font-size:2rem">⚙️</div>
        <div>
            <div class="admin-title">Campus Assistant — Admin Panel</div>
            <div class="admin-sub">Zewail City of Science and Technology · Real-time feedback &amp; knowledge management</div>
        </div>
    </div>""", unsafe_allow_html=True)
with col_h2:
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["admin_authed"] = False
        st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_overview, tab_flagged, tab_all, tab_analysis, tab_patches = st.tabs([
    "📊  Overview",
    "🚩  Flagged Queries",
    "📋  All Queries",
    "🤖  AI Gap Analysis",
    "⚙️  Prompt Patches",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Overview
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    stats = get_stats()
    entries = get_all_entries()

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in [
        (c1, stats["total"],   "Total Queries"),
        (c2, stats["flagged"], "🚩 Flagged"),
        (c3, stats["up"],      "👍 Positive"),
        (c4, stats["down"],    "👎 Negative"),
        (c5, f'{stats["avg_score"]:.0%}', "Avg Confidence"),
    ]:
        col.markdown(
            f'<div class="stat-box"><div class="stat-val">{val}</div>'
            f'<div class="stat-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    if not entries:
        st.info("No queries logged yet. Queries are logged automatically when students use the chatbot.")
        st.stop()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Queries over time chart ────────────────────────────────────────────────
    import pandas as pd

    df_all = pd.DataFrame(entries)
    df_all["date"] = pd.to_datetime(df_all["timestamp"]).dt.date

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown(
            "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>"
            "📈 Queries per day</div>",
            unsafe_allow_html=True,
        )
        by_day = df_all.groupby("date").size().reset_index(name="Queries")
        st.bar_chart(by_day.set_index("date"), color="#8B5CF6")

    with col_chart2:
        st.markdown(
            "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>"
            "🎯 Confidence score distribution</div>",
            unsafe_allow_html=True,
        )
        if "max_score" in df_all.columns:
            bins = pd.cut(df_all["max_score"], bins=[0, 0.3, 0.45, 0.6, 0.75, 1.01],
                          labels=["<30%", "30-45%", "45-60%", "60-75%", "75%+"])
            dist = bins.value_counts().sort_index()
            st.bar_chart(dist, color="#EC4899")

    # ── Intent distribution ────────────────────────────────────────────────────
    if "intent" in df_all.columns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px'>"
            "🏷️ Query intent distribution</div>",
            unsafe_allow_html=True,
        )
        intent_counts = df_all["intent"].value_counts()
        st.bar_chart(intent_counts, color="#F59E0B")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Flagged Queries
# ══════════════════════════════════════════════════════════════════════════════
with tab_flagged:
    flagged = get_flagged_entries()
    if not flagged:
        st.success("✅ No flagged queries. The system is handling all student questions confidently.")
    else:
        st.markdown(
            f"<div style='color:#F87171;font-size:.85rem;margin-bottom:12px'>"
            f"⚠️ {len(flagged)} queries need attention "
            f"(low confidence &lt; 45% or thumbs-down)</div>",
            unsafe_allow_html=True,
        )

        # Filter controls
        fcol1, fcol2 = st.columns([2, 1])
        with fcol1:
            search_q = st.text_input("🔍 Filter by keyword", placeholder="e.g. scholarship, office hours…")
        with fcol2:
            thumb_filter = st.selectbox("Thumb filter", ["All", "👎 Down", "Not rated"])

        filtered = flagged
        if search_q:
            filtered = [e for e in filtered if search_q.lower() in e["question"].lower()]
        if thumb_filter == "👎 Down":
            filtered = [e for e in filtered if e.get("thumb") == "down"]
        elif thumb_filter == "Not rated":
            filtered = [e for e in filtered if e.get("thumb") is None]

        st.markdown(f"**{len(filtered)} entries shown**")

        for e in reversed(filtered[-50:]):
            thumb_icon = {"up": "👍", "down": "👎"}.get(e.get("thumb"), "—")
            score_css  = "score-ok" if e["max_score"] >= 0.45 else "score-bad"
            with st.expander(
                f"[{e['timestamp'][:10]}] {e['question'][:80]}{'…' if len(e['question'])>80 else ''}",
                expanded=False,
            ):
                c1e, c2e, c3e = st.columns(3)
                c1e.markdown(f"**Confidence:** <span class='{score_css}'>{e['max_score']:.0%}</span>",
                             unsafe_allow_html=True)
                c2e.markdown(f"**Thumb:** {thumb_icon}")
                c3e.markdown(f"**Intent:** `{e.get('intent','—')}`")

                st.markdown(f"**Question:** {e['question']}")
                st.markdown(f"**Answer preview:**")
                st.markdown(
                    f"<div style='background:#120d24;border-radius:8px;padding:10px;"
                    f"font-size:.8rem;color:#9D77F5;margin:4px 0'>{e['answer_preview']}</div>",
                    unsafe_allow_html=True,
                )

                # Manual thumb override from admin
                col_up, col_dn, _ = st.columns([1, 1, 5])
                if col_up.button("Mark 👍", key=f"adm_up_{e['entry_id']}"):
                    update_thumb(e["entry_id"], "up")
                    st.success("Marked as positive.")
                    st.rerun()
                if col_dn.button("Mark 👎", key=f"adm_dn_{e['entry_id']}"):
                    update_thumb(e["entry_id"], "down")
                    st.success("Marked as negative.")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: All Queries
# ══════════════════════════════════════════════════════════════════════════════
with tab_all:
    entries = get_all_entries()
    if not entries:
        st.info("No queries logged yet.")
    else:
        import pandas as pd

        df = pd.DataFrame(entries)[
            ["timestamp", "question", "intent", "max_score", "thumb", "flagged", "contact_key"]
        ].copy()
        df["timestamp"] = df["timestamp"].str[:16]
        df["max_score"] = (df["max_score"] * 100).round(0).astype(int).astype(str) + "%"
        df["thumb"]     = df["thumb"].fillna("—")
        df["flagged"]   = df["flagged"].map({True: "🚩", False: ""})
        df.rename(columns={
            "timestamp":   "Time",
            "question":    "Question",
            "intent":      "Intent",
            "max_score":   "Confidence",
            "thumb":       "Thumb",
            "flagged":     "Flag",
            "contact_key": "Routed to",
        }, inplace=True)

        # Search
        search = st.text_input("🔍 Search questions", placeholder="Type to filter…")
        if search:
            df = df[df["Question"].str.contains(search, case=False, na=False)]

        st.dataframe(df[::-1], use_container_width=True, height=500)

        # Download
        csv = pd.DataFrame(entries).to_csv(index=False)
        st.download_button(
            "⬇️ Download full log (CSV)",
            data=csv,
            file_name=f"campus_assistant_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: AI Gap Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    st.markdown(
        "Run a GPT-4o analysis of all flagged/thumbs-down queries. "
        "The AI will identify knowledge gaps, suggest content to add, "
        "and propose new system prompt rules.",
    )

    flagged_count = len(get_flagged_entries())
    st.markdown(f"**{flagged_count} flagged queries** will be analysed.")

    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not found in environment. Cannot run analysis.")
    elif flagged_count == 0:
        st.success("No flagged queries to analyse yet.")
    else:
        if st.button("🤖 Run AI Gap Analysis", type="primary", use_container_width=False):
            with st.spinner("GPT-4o is analysing knowledge gaps…"):
                report = run_ai_gap_analysis(OPENAI_API_KEY)
            st.session_state["gap_report"] = report
            st.success("Analysis complete!")

    if "gap_report" in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Analysis Report")
        st.markdown(st.session_state["gap_report"])

        st.markdown("---")
        st.markdown("### ➕ Apply a Rule from this Report")
        st.markdown(
            "Copy a suggested rule from the report above and add it to the "
            "assistant's system prompt. It will be applied immediately on next query."
        )
        new_rule_label = st.text_input(
            "Rule label (short description)",
            placeholder="e.g. Handle office-hours questions",
        )
        new_rule_text = st.text_area(
            "Rule text",
            height=120,
            placeholder=(
                "e.g. When a student asks about faculty office hours, "
                "always provide the Guide & Info doc link if exact hours are not found."
            ),
        )
        if st.button("✅ Add Rule to System Prompt", type="primary"):
            if new_rule_text.strip():
                pid = save_prompt_patch(new_rule_text, label=new_rule_label or new_rule_text[:60])
                st.success(f"Rule added (ID: {pid}). It will apply to all new queries immediately.")
                st.rerun()
            else:
                st.warning("Please enter a rule before adding.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: Prompt Patches
# ══════════════════════════════════════════════════════════════════════════════
with tab_patches:
    patches = load_prompt_patches()

    st.markdown(
        "These rules are appended to the assistant's system prompt on every query. "
        "They are applied in real-time — no restart required.",
    )

    if not patches:
        st.info("No prompt patches yet. Run the AI Gap Analysis and add rules from there.")
    else:
        st.markdown(f"**{len(patches)} active rule(s)**")
        for p in patches:
            with st.expander(
                f"[{p['patch_id']}] {p.get('label', p['rule'][:60])}",
                expanded=False,
            ):
                st.markdown(f"**Created:** {p.get('created','—')}")
                st.markdown(f"**Rule:**")
                st.code(p["rule"], language=None)
                if st.button("🗑️ Delete this rule", key=f"del_{p['patch_id']}"):
                    delete_prompt_patch(p["patch_id"])
                    st.success("Rule deleted.")
                    st.rerun()

    st.markdown("---")
    st.markdown("### ➕ Add a Rule Manually")
    m_label = st.text_input("Label", placeholder="Short description of the rule")
    m_rule  = st.text_area("Rule text", height=100,
                           placeholder="Type a new system prompt instruction…")
    if st.button("Add Rule", type="primary"):
        if m_rule.strip():
            pid = save_prompt_patch(m_rule, label=m_label or m_rule[:60])
            st.success(f"Rule added (ID: {pid}).")
            st.rerun()
        else:
            st.warning("Enter a rule first.")

    # Re-ingest trigger
    st.markdown("---")
    st.markdown("### 🔄 Re-run LinkTree Ingest")
    st.markdown(
        "After adding new documents to the knowledge base, re-run the ingestion "
        "pipeline to embed them into ChromaDB."
    )
    if st.button("▶️ Run linktree_ingest.py now"):
        import subprocess
        py = str(Path(sys.executable))
        ingest = str(PROJECT_ROOT / "linktree_ingest.py")
        with st.spinner("Running ingest… (this may take 1–2 minutes)"):
            result = subprocess.run(
                [py, ingest],
                capture_output=True, text=True,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                timeout=180,
            )
        if result.returncode == 0:
            st.success("Ingest completed successfully.")
            st.code(result.stdout[-3000:] if result.stdout else "(no output)")
        else:
            st.error("Ingest failed.")
            st.code(result.stderr[-2000:] if result.stderr else "(no error output)")
