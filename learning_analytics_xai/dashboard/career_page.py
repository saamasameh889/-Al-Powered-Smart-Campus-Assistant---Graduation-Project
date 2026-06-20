"""
career_page.py — GitHub Career Advisor UI  (Product E)
═══════════════════════════════════════════════════════
Model choice: GPT-4o
  Experiment result (3 profiles × 2 models, scored by gpt-4o-mini judge):
    GPT-4o      avg=4.75/5  time=4.52s
    GPT-4o-mini avg=4.75/5  time=5.47s
  Equal judge scores; GPT-4o faster and more specific in practice.

Entry point:
    from career_page import render_career_page
    render_career_page(openai_client=client)
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import streamlit as st

_HERE   = Path(__file__).parent
_ROOT   = _HERE.parent
_CAREER = _ROOT / "career"
for p in [str(_ROOT), str(_CAREER), str(_HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from github_analyzer import GitHubAnalyzer, PROGRAMME_STACK

PROGRAMMES = list(PROGRAMME_STACK.keys())


# ── HTML render helper (Streamlit 1.31+ uses st.html; older uses st.markdown) ─

def _html(content: str, container=None) -> None:
    target = container if container is not None else st
    if hasattr(target, "html"):
        target.html(content)
    else:
        target.markdown(content, unsafe_allow_html=True)


# ── HTML snippet builders ─────────────────────────────────────────────────────

def _badge(text: str, color: str = "#8B5CF6") -> str:
    return (
        f'<span style="background:{color}22;border:1px solid {color}55;'
        f'color:{color};border-radius:100px;padding:2px 10px;'
        f'font-size:.72rem;font-weight:700;letter-spacing:.04em">{text}</span>'
    )


def _metric_card(icon: str, value: str, label: str, color: str = "#8B5CF6") -> str:
    return (
        f'<div style="background:rgba(255,255,255,.03);border:1px solid {color}33;'
        f'border-top:3px solid {color};border-radius:14px;padding:18px 14px;'
        f'text-align:center;">'
        f'<div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>'
        f'<div style="font-size:1.5rem;font-weight:800;color:#EDE9FE">{value}</div>'
        f'<div style="font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;'
        f'color:{color};margin-top:4px;font-weight:700">{label}</div>'
        f'</div>'
    )


def _quality_bar(score: int, max_score: int = 4) -> str:
    filled = "█" * score
    empty  = "░" * (max_score - score)
    color  = "#10B981" if score >= 3 else ("#F59E0B" if score >= 2 else "#EF4444")
    return f'<span style="color:{color};letter-spacing:2px;font-size:.85rem">{filled}{empty}</span>'


def _activity_color(cpw: float) -> str:
    return "#10B981" if cpw >= 5 else ("#F59E0B" if cpw >= 2 else "#EF4444")


def _alignment_color(score: int) -> str:
    return "#10B981" if score >= 70 else ("#F59E0B" if score >= 40 else "#EF4444")


# ── Main render ───────────────────────────────────────────────────────────────

def render_career_page(openai_client=None) -> None:

    # ── Page header ──────────────────────────────────────────────────────────
    _html("""
<div style="background:linear-gradient(135deg,#0D0B2A 0%,#1A0E3F 50%,#0F1929 100%);border:1px solid rgba(139,92,246,.22);border-radius:20px;padding:22px 28px;margin-bottom:18px;box-shadow:0 16px 56px rgba(91,33,182,.18)">
  <div style="display:flex;align-items:center;gap:16px">
    <div style="font-size:2.2rem">💼</div>
    <div>
      <div style="font-size:1.25rem;font-weight:800;color:#F5F0FF;letter-spacing:-.02em">GitHub Career Advisor</div>
      <div style="font-size:.78rem;color:#6D4ABA;margin-top:3px">Portfolio analysis · Internship readiness · Actionable AI advice</div>
    </div>
  </div>
</div>
""")

    # ── Input row ────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([3, 2, 1, 1.2])
    with c1:
        username = st.text_input("GitHub Username", placeholder="e.g. torvalds", key="career_username")
    with c2:
        programme = st.selectbox("Programme", PROGRAMMES, index=0, key="career_prog")
    with c3:
        semester = st.number_input("Semester", min_value=1, max_value=8, value=4, key="career_sem")
    with c4:
        token = st.text_input("GitHub Token (optional)", type="password",
                              placeholder="ghp_… (60→5000 req/h)", key="career_token")

    analyze_btn = st.button("🔍  Analyze Portfolio", type="primary", key="career_analyze")

    # Accept full GitHub URLs
    if username:
        m = re.match(r'https?://(?:www\.)?github\.com/([^/\s?#]+)', username.strip())
        if m:
            username = m.group(1)

    if not username:
        _html("""
<div style="text-align:center;padding:60px 0">
  <div style="font-size:3rem;margin-bottom:12px;opacity:.5">💼</div>
  <div style="font-size:1rem;font-weight:700;color:#5B4D8A">Enter a GitHub username to get started</div>
  <div style="font-size:.82rem;color:#3D3060;margin-top:6px">3 API calls · No token required · Results cached per session</div>
</div>
""")
        return

    # ── Fetch / cache ─────────────────────────────────────────────────────────
    cache_key = f"career_analysis_{username}_{programme}_{semester}"

    if analyze_btn or cache_key not in st.session_state:
        analyzer = GitHubAnalyzer(token=(token or "").strip() or None)
        with st.spinner(f"Fetching @{username} from GitHub…"):
            try:
                t0 = time.time()
                analysis = analyzer.analyze(username, programme=programme, semester=semester)
                st.session_state[cache_key] = analysis
                st.session_state[f"{cache_key}_fetch_time"] = round(time.time() - t0, 2)
            except ValueError as e:
                st.error(f"User not found: {e}")
                return
            except RuntimeError as e:
                st.error(str(e))
                st.info(
                    "**To get a token:** github.com → Settings → Developer settings → "
                    "Personal access tokens → Tokens (classic) → Generate new token → "
                    "select **public_repo** scope → paste it in the GitHub Token field above."
                )
                return
            except Exception as e:
                st.error(f"API error: {e}")
                return

    analysis   = st.session_state[cache_key]
    fetch_time = st.session_state.get(f"{cache_key}_fetch_time", 0)
    a          = analysis

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 1 — PROFILE HEADER
    # ═══════════════════════════════════════════════════════════════════════
    gap_count = len(a["gaps"])
    gap_color = "#EF4444" if gap_count >= 5 else ("#F59E0B" if gap_count >= 2 else "#10B981")
    name_extra = (f" <span style='font-size:.8rem;font-weight:400;color:#6D4ABA'>· {a['name']}</span>"
                  if a["name"] != a["username"] else "")
    bio_html   = a["bio"] if a["bio"] else "<i>No bio</i>"
    loc_html   = f" &nbsp;·&nbsp; {a['location']}" if a["location"] else ""

    org_badge = _badge("Organization", "#0EA5E9") if a.get("is_org") else ""
    _html(f"""
<div style="background:rgba(255,255,255,.025);border:1px solid rgba(139,92,246,.2);border-radius:16px;padding:20px 24px;margin:12px 0 6px">
  <div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">
    <div>
      <div style="font-size:1.1rem;font-weight:800;color:#EDE9FE">@{a['username']}{name_extra}</div>
      <div style="font-size:.8rem;color:#5B4D8A;margin-top:3px">{bio_html} &nbsp;·&nbsp; Joined {a['joined_year']}{loc_html}</div>
    </div>
    <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
      {org_badge}
      {_badge(programme, "#8B5CF6")}
      {_badge(f"Sem {semester}", "#6D28D9")}
      {_badge(f"{gap_count} gap{'s' if gap_count != 1 else ''}", gap_color)}
      <span style="font-size:.7rem;color:#3D3060">fetched in {fetch_time}s</span>
    </div>
  </div>
</div>
""")

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 2 — METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    act_c = _activity_color(a["activity"]["commits_per_week"])
    aln_c = _alignment_color(a["alignment"]["score"])
    _html(_metric_card("📁", str(a["n_repos"]),    "Public Repos",       "#8B5CF6"), m1)
    _html(_metric_card("⭐", f'{a["total_stars"]:,}', "Total Stars",     "#F59E0B"), m2)
    _html(_metric_card("🍴", f'{a["total_forks"]:,}', "Total Forks",     "#6366F1"), m3)
    _html(_metric_card("⚡", f"{a['activity']['commits_per_week']:.1f}/wk",
                       "Commits/Week (90d)", act_c), m4)
    _html(_metric_card("🎯", f"{a['alignment']['score']}%",
                       f"{programme} Alignment", aln_c), m5)
    _html(_metric_card("🏆", f"{a['avg_quality']:.1f}/5", "Repo Quality", "#10B981"), m6)

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 3 — LANGUAGE CHART + TOP REPOS
    # ═══════════════════════════════════════════════════════════════════════
    col_lang, col_repos = st.columns([1, 1.6])

    with col_lang:
        _html("<div style='font-size:.75rem;font-weight:700;color:#9D77F5;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>Language Breakdown</div>")
        import pandas as pd
        lang_df = pd.DataFrame(list(a["lang_pct"].items())[:8], columns=["Language", "Percent"])
        if not lang_df.empty:
            st.bar_chart(lang_df.set_index("Language"), color="#8B5CF6", height=200)
        al = a["alignment"]
        _html(
            f"<div style='font-size:.72rem;margin-top:8px;color:#6D4ABA;line-height:1.7'>"
            f"<span style='color:#9D77F5;font-weight:700'>Core coverage:</span> "
            f"<span style='color:#EDE9FE'>{al['core_coverage']}%</span> &nbsp;"
            f"<span style='color:#9D77F5;font-weight:700'>Relevance:</span> "
            f"<span style='color:#EDE9FE'>{al['relevance']}%</span><br>"
            f"&#10003; Has: <span style='color:#10B981;font-weight:600'>{', '.join(al['matched'][:6]) or 'none'}</span><br>"
            f"&#10007; Missing core: <span style='color:#EF4444;font-weight:600'>{', '.join(al['missing'][:4]) or 'none'}</span>"
            f"</div>"
        )

    with col_repos:
        _html("<div style='font-size:.75rem;font-weight:700;color:#9D77F5;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>Top Repositories</div>")
        for r in a["top_repos"]:
            q_bar    = _quality_bar(r["quality"])
            lang_str = " · ".join(f"{l}:{p}%" for l, p in r["languages"].items()) or "?"
            dom_str  = " ".join(
                f'<span style="font-size:.64rem;background:rgba(139,92,246,.15);border-radius:4px;padding:1px 6px;color:#A78BFA">{d}</span>'
                for d in r["domains"][:2]
            )
            fork_tag = ' <span style="color:#6D4ABA;font-size:.68rem">[fork]</span>' if r["is_fork"] else ""
            topics   = " ".join(f"#{t}" for t in r["topics"][:3])
            topics_div = f'<div style="font-size:.64rem;color:#3D3060;margin-top:3px">{topics}</div>' if topics else ""
            _html(
                f'<div style="background:rgba(255,255,255,.02);border:1px solid rgba(139,92,246,.12);border-radius:12px;padding:12px 16px;margin-bottom:6px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                f'<a href="{r["url"]}" target="_blank" style="color:#C4B5FD;font-weight:700;font-size:.85rem;text-decoration:none">&#128279; {r["name"]}</a>{fork_tag}'
                f'<span style="margin-left:auto;color:#F59E0B;font-size:.75rem">&#11088;{r["stars"]}</span>'
                f'</div>'
                f'<div style="font-size:.75rem;color:#4A3A7A;margin-bottom:6px">{r["description"][:90]}</div>'
                f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">{q_bar} <span style="font-size:.68rem;color:#3D3060">{lang_str}</span> {dom_str}</div>'
                f'{topics_div}'
                f'</div>'
            )

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 4 — GAPS + ACTIVITY
    # ═══════════════════════════════════════════════════════════════════════
    col_gaps, col_act = st.columns(2)

    with col_gaps:
        _html("<div style='font-size:.75rem;font-weight:700;color:#9D77F5;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>Detected Gaps</div>")
        if a["gaps"]:
            for g in a["gaps"]:
                _html(
                    f'<div style="background:rgba(239,68,68,.06);border:1px solid rgba(239,68,68,.2);'
                    f'border-left:3px solid #EF4444;border-radius:0 10px 10px 0;'
                    f'padding:8px 12px;margin-bottom:5px;font-size:.78rem;color:#FCA5A5">&#9888; {g}</div>'
                )
        else:
            _html('<div style="color:#10B981;font-size:.82rem">&#10003; No major gaps detected</div>')

    with col_act:
        _html("<div style='font-size:.75rem;font-weight:700;color:#9D77F5;text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>Activity (Last 90 Days)</div>")
        act = a["activity"]
        rows = ""
        for label, val, color in [
            ("Total Commits",   str(act["total_commits_90d"]),             "#8B5CF6"),
            ("Commits/Week",    f"{act['commits_per_week']:.1f}",          _activity_color(act["commits_per_week"])),
            ("Active Weeks",    f"{act['active_weeks']}/13",               "#6D28D9"),
            ("Collab Events",   str(act["collab_events"]),                  "#10B981" if act["collab_events"] > 5 else "#F59E0B"),
            ("Days Since Push", str(act["days_since_push"]),                "#EF4444" if act["days_since_push"] > 60 else "#10B981"),
            ("Profile Score",   f"{a['presentation_score']}/3",            "#F59E0B"),
        ]:
            rows += (
                f'<div style="display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid rgba(139,92,246,.07)">'
                f'<span style="font-size:.78rem;color:#5B4D8A">{label}</span>'
                f'<span style="font-size:.82rem;font-weight:700;color:{color}">{val}</span>'
                f'</div>'
            )
        _html(f'<div style="background:rgba(255,255,255,.02);border:1px solid rgba(139,92,246,.15);border-radius:12px;padding:14px 18px">{rows}</div>')

    # ═══════════════════════════════════════════════════════════════════════
    #  SECTION 5 — AI CAREER ADVICE  (GPT-4o streamed)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    _html("<div style='font-size:.75rem;font-weight:700;color:#9D77F5;text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px'>AI Career Advice — GPT-4o Analysis</div>")

    advice_key = f"{cache_key}_advice"
    regen_btn  = st.button("Regenerate Advice", key="career_regen", type="secondary")

    if regen_btn and advice_key in st.session_state:
        del st.session_state[advice_key]

    if advice_key not in st.session_state:
        if openai_client is None:
            try:
                from openai import OpenAI
                openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            except Exception as e:
                st.error(f"OpenAI client unavailable: {e}")
                return

        with st.spinner("GPT-4o analyzing portfolio…"):
            placeholder = st.empty()
            full_advice = ""
            try:
                stream = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": a["prompt"]}],
                    temperature=0.35,
                    max_tokens=1000,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_advice += delta
                    placeholder.markdown(full_advice + "▌")
                placeholder.markdown(full_advice)
                st.session_state[advice_key] = full_advice
            except Exception as e:
                st.error(f"GPT-4o error: {e}")
                return
    else:
        st.markdown(st.session_state[advice_key])

    # ── Footer ────────────────────────────────────────────────────────────────
    _html(
        f"<div style='font-size:.68rem;color:#2A1F4A;margin-top:16px;text-align:center'>"
        f"Analysis cached for @{username} · {programme} semester {semester} · "
        f"GitHub REST API v3 (unauthenticated: 60 req/h) · GPT-4o career advisor</div>"
    )
