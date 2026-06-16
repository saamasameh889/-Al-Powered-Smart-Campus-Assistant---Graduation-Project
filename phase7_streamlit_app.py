#!/usr/bin/env python3
"""
phase7_streamlit_app.py  —  Zewail City Campus Assistant
Phase 7: Premium Streamlit UI — Academic Advisor AI integrated.
"""
from __future__ import annotations

import re
import sys
import time

# Packages installed on D: due to C: disk space constraints
_D_PACKAGES = r"D:\py311_packages"
if _D_PACKAGES not in sys.path:
    sys.path.insert(0, _D_PACKAGES)
from pathlib import Path

_CITE_RE = re.compile(
    r"\[Source\s*\d+[^\]]*\]"
    r"|\(Source\s*\d+[^)]*\)"
    r"|\([A-Z]{2,}\s+\w[^)]{2,40}\)"
    r"|Source\s*\d+\s*\|[^.\n]*[.\n]",
)

def _clean_answer(text: str) -> str:
    return _CITE_RE.sub("", text).strip()

_ADVISOR_HEADERS = re.compile(
    r'^#{1,3}\s*(Student Summary|Eligibility Analysis|Recommended Plans|'
    r'Risk Analysis|Graduation Impact|Academic Notes|Safe Plan|Balanced Plan|'
    r'Fast Graduation Plan)',
    re.MULTILINE,
)

def _is_advisor_response(text: str) -> bool:
    return bool(_ADVISOR_HEADERS.search(text))

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

_XAI_DASHBOARD = str(PROJECT_ROOT / "learning_analytics_xai" / "dashboard")
if _XAI_DASHBOARD not in sys.path:
    sys.path.insert(0, _XAI_DASHBOARD)
try:
    import importlib
    if "analytics_page" in sys.modules:
        importlib.reload(sys.modules["analytics_page"])
    from analytics_page import render_learning_analytics_page as _render_xai_page
    _XAI_IMPORT_OK  = True
    _XAI_IMPORT_ERR = ""
except Exception as _e:
    _XAI_IMPORT_OK  = False
    _XAI_IMPORT_ERR = str(_e)

# ── Product D: clustering page ─────────────────────────────────────────────────
try:
    from clustering_page import render_clustering_page as _render_clustering_page
    _CLUST_IMPORT_OK  = True
    _CLUST_IMPORT_ERR = ""
except Exception as _e2:
    _CLUST_IMPORT_OK  = False
    _CLUST_IMPORT_ERR = str(_e2)

# ── Product C: forecasting page ────────────────────────────────────────────────
_FCAST_DASHBOARD = str(PROJECT_ROOT / "learning_analytics_xai" / "dashboard")
if _FCAST_DASHBOARD not in sys.path:
    sys.path.insert(0, _FCAST_DASHBOARD)
try:
    from forecasting_page import render_forecasting_page as _render_forecasting_page
    _FCAST_IMPORT_OK  = True
    _FCAST_IMPORT_ERR = ""
except Exception as _e3:
    _FCAST_IMPORT_OK  = False
    _FCAST_IMPORT_ERR = str(_e3)

st.set_page_config(
    page_title="Zewail City Campus Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS  —  Professional dark-purple design system
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: #07010F !important;
    color: #EDE9FE !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    padding-top: .75rem !important;
    padding-bottom: 3rem !important;
    max-width: 1320px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(139,92,246,.55); }

/* ════════════════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #0B0120 0%, #0F0527 55%, #130830 100%) !important;
    border-right: 1px solid rgba(139,92,246,.13) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not([class*="badge"]),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #C4B5FD !important; }

[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid rgba(139,92,246,.12) !important;
    margin: 12px 0 !important;
}

[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(139,92,246,.07) !important;
    border: 1px solid rgba(139,92,246,.18) !important;
    border-radius: 14px !important;
    padding: 12px 14px !important;
    transition: border-color .2s !important;
}
[data-testid="stSidebar"] [data-testid="stMetric"]:hover {
    border-color: rgba(139,92,246,.35) !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] p {
    font-size: .65rem !important;
    text-transform: uppercase !important;
    letter-spacing: .12em !important;
    color: #7C3AED !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #EDE9FE !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #6D28D9 0%, #7C3AED 50%, #8B5CF6 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    letter-spacing: .01em !important;
    padding: 11px 0 !important;
    transition: all .25s ease !important;
    box-shadow: 0 4px 18px rgba(109,40,217,.35) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(109,40,217,.5) !important;
    opacity: .95 !important;
}

/* ════════════════════════════════════════════════════════
   HEADER
════════════════════════════════════════════════════════ */
.zc-header {
    background: linear-gradient(135deg, #110330 0%, #1C0850 35%, #2D1080 65%, #1A0648 100%);
    border: 1px solid rgba(139,92,246,.22);
    border-radius: 20px;
    padding: 20px 28px;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 18px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 56px rgba(91,33,182,.22), inset 0 1px 0 rgba(255,255,255,.05);
}
.zc-header::before {
    content: '';
    position: absolute; top: -70px; right: -30px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(236,72,153,.14) 0%, transparent 68%);
    border-radius: 50%; pointer-events: none;
}
.zc-header::after {
    content: '';
    position: absolute; bottom: -50px; left: 25%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(139,92,246,.1) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.zc-logo {
    width: 50px; height: 50px; flex-shrink: 0; z-index: 1;
    background: linear-gradient(135deg, #6D28D9 0%, #8B5CF6 50%, #EC4899 100%);
    border-radius: 15px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 6px 22px rgba(109,40,217,.55);
}
.zc-htxt { flex: 1; z-index: 1; min-width: 0; }
.zc-htxt h1 {
    margin: 0;
    font-size: 1.32rem; font-weight: 800;
    color: #F5F0FF; letter-spacing: -.025em;
    text-shadow: 0 2px 16px rgba(139,92,246,.25);
}
.zc-htxt p {
    margin: 4px 0 0;
    font-size: .76rem; color: #9D77F5; font-weight: 400; letter-spacing: .01em;
}
.zc-badge {
    z-index: 1; flex-shrink: 0;
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(139,92,246,.18), rgba(236,72,153,.12));
    border: 1px solid rgba(139,92,246,.32);
    color: #C4B5FD;
    border-radius: 100px; padding: 6px 16px;
    font-size: .68rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
    box-shadow: 0 4px 14px rgba(109,40,217,.2);
}

/* ════════════════════════════════════════════════════════
   TABS
════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid rgba(139,92,246,.14) !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #5B4D8A !important;
    font-weight: 600 !important;
    font-size: .85rem !important;
    border: none !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 22px !important;
    transition: color .2s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover { color: #A78BFA !important; }
[data-testid="stTabs"] [aria-selected="true"] {
    color: #EDE9FE !important;
    background: rgba(139,92,246,.1) !important;
    border-bottom: 2px solid #8B5CF6 !important;
}

/* ════════════════════════════════════════════════════════
   CHAT MESSAGES  (st.chat_message)
════════════════════════════════════════════════════════ */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 3px 0 !important;
    gap: 14px !important;
    align-items: flex-start !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    background: rgba(255,255,255,.025) !important;
    border: 1px solid rgba(139,92,246,.16) !important;
    border-left: 3px solid #7C3AED !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 16px 20px !important;
    line-height: 1 !important;
}

/* User bubble override using :has() */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
    background: linear-gradient(135deg, rgba(76,29,149,.55), rgba(109,40,217,.4)) !important;
    border: 1px solid rgba(139,92,246,.3) !important;
    border-left: none !important;
    border-right: 3px solid #A78BFA !important;
    border-radius: 18px 4px 18px 18px !important;
}

/* Text inside messages */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
    color: #EDE9FE !important;
    font-size: .91rem !important;
    line-height: 1.78 !important;
    margin-bottom: .55rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
    margin-bottom: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
    color: #DDD6FE !important;
    font-size: .91rem !important;
    line-height: 1.76 !important;
    margin-bottom: .28rem !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
    color: #C4B5FD !important;
    font-size: .98rem !important;
    font-weight: 700 !important;
    margin: .9rem 0 .4rem !important;
    padding-bottom: .35rem !important;
    border-bottom: 1px solid rgba(139,92,246,.15) !important;
    letter-spacing: -.01em !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2:first-child,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3:first-child {
    margin-top: 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
    color: #C4B5FD !important; font-weight: 600 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em {
    color: #A78BFA !important; font-style: italic !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    background: rgba(139,92,246,.18) !important;
    color: #F0ABFC !important;
    border-radius: 5px !important;
    padding: 2px 7px !important;
    font-size: .83rem !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
    border-left: 3px solid rgba(139,92,246,.4) !important;
    padding-left: 12px !important;
    margin: .5rem 0 !important;
    color: #9D77F5 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: .84rem !important;
    margin: .75rem 0 !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
    background: rgba(109,40,217,.25) !important;
    color: #C4B5FD !important;
    padding: 9px 13px !important;
    font-weight: 700 !important;
    font-size: .8rem !important;
    letter-spacing: .03em !important;
    text-align: left !important;
    border: 1px solid rgba(139,92,246,.2) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
    padding: 8px 13px !important;
    border: 1px solid rgba(139,92,246,.12) !important;
    color: #DDD6FE !important;
    vertical-align: top !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] tr:nth-child(even) td {
    background: rgba(139,92,246,.04) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] tr:hover td {
    background: rgba(139,92,246,.08) !important;
}
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] hr {
    border: none !important;
    border-top: 1px solid rgba(139,92,246,.15) !important;
    margin: .75rem 0 !important;
}

/* User bubble text */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] li {
    color: #F5F0FF !important;
}

/* ════════════════════════════════════════════════════════
   CHAT INPUT
════════════════════════════════════════════════════════ */
[data-testid="stChatInput"] {
    background: rgba(15,5,35,.95) !important;
    border: 1.5px solid rgba(139,92,246,.28) !important;
    border-radius: 18px !important;
    color: #EDE9FE !important;
    box-shadow: 0 4px 28px rgba(0,0,0,.35) !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,.12), 0 4px 28px rgba(0,0,0,.35) !important;
}
[data-testid="stChatInput"] textarea {
    color: #EDE9FE !important;
    font-size: .9rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #4A3A7A !important; }

/* ════════════════════════════════════════════════════════
   SUGGESTION BUTTONS
════════════════════════════════════════════════════════ */
.sug-btn-wrap .stButton > button,
div[data-testid="column"] .stButton > button {
    background: rgba(139,92,246,.08) !important;
    border: 1px solid rgba(139,92,246,.2) !important;
    color: #B8A4F0 !important;
    border-radius: 100px !important;
    font-size: .77rem !important;
    font-weight: 500 !important;
    padding: 7px 15px !important;
    width: 100% !important;
    transition: all .22s ease !important;
    letter-spacing: .01em !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(139,92,246,.22) !important;
    border-color: #8B5CF6 !important;
    color: #EDE9FE !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(139,92,246,.2) !important;
}

/* ════════════════════════════════════════════════════════
   EXPANDER  (source citations)
════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: rgba(139,92,246,.03) !important;
    border: 1px solid rgba(139,92,246,.12) !important;
    border-radius: 12px !important;
    margin: 2px 0 10px 46px !important;
}
[data-testid="stExpander"] summary {
    color: #5B4D8A !important;
    font-size: .75rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    letter-spacing: .01em !important;
}
[data-testid="stExpander"] summary:hover { color: #A78BFA !important; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    padding: 0 14px 14px !important;
}

/* ════════════════════════════════════════════════════════
   SOURCE CARDS
════════════════════════════════════════════════════════ */
.src-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 8px;
}
.src-card {
    background: rgba(11,4,28,.85);
    border: 1px solid rgba(139,92,246,.14);
    border-radius: 12px; padding: 12px 14px;
    font-size: .77rem; position: relative; overflow: hidden;
    transition: border-color .2s, transform .2s;
    cursor: default;
}
.src-card:hover { border-color: rgba(139,92,246,.38); transform: translateY(-2px); }
.src-card::before {
    content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
}
.sc-courses::before   { background: #8B5CF6; }
.sc-policy::before    { background: #EC4899; }
.sc-admissions::before{ background: #06B6D4; }
.sc-research::before  { background: #10B981; }
.sc-faculty::before   { background: #F59E0B; }
.sc-facilities::before{ background: #EF4444; }
.sc-deadlines::before { background: #F97316; }
.sc-general::before   { background: #8B7EC8; }
.sc-name  { font-weight: 600; color: #C4B5FD; margin-bottom: 3px; font-size: .79rem; }
.sc-meta  { color: #3D3060; font-size: .69rem; margin-bottom: 5px; }
.sc-score {
    background: rgba(139,92,246,.16); color: #A78BFA;
    border-radius: 100px; padding: 2px 9px;
    font-size: .67rem; font-weight: 700; display: inline-block;
}
.sc-exc {
    color: #5B4D8A; font-size: .71rem; line-height: 1.45; margin-top: 7px;
    border-top: 1px solid rgba(139,92,246,.1); padding-top: 6px;
}

/* ════════════════════════════════════════════════════════
   ADVISOR BADGE
════════════════════════════════════════════════════════ */
.advisor-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: linear-gradient(135deg, rgba(109,40,217,.2), rgba(236,72,153,.1));
    border: 1px solid rgba(139,92,246,.28);
    color: #A78BFA;
    border-radius: 100px; padding: 4px 14px;
    font-size: .66rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; margin-bottom: 12px;
}

/* ════════════════════════════════════════════════════════
   PROFILE PILLS  (sidebar)
════════════════════════════════════════════════════════ */
.ppill {
    background: rgba(109,40,217,.07);
    border: 1px solid rgba(139,92,246,.18);
    border-left: 3px solid #7C3AED;
    border-radius: 10px; padding: 9px 13px; margin: 5px 0;
    font-size: .8rem; line-height: 1.45;
}
.ppill .plbl {
    font-size: .62rem; text-transform: uppercase; letter-spacing: .12em;
    color: #7C3AED !important; display: block; margin-bottom: 3px; font-weight: 700;
}
.ppill .pval { color: #DDD6FE !important; font-weight: 600; }
.ppill-warn {
    background: rgba(239,68,68,.06);
    border: 1px solid rgba(239,68,68,.2);
    border-left: 3px solid #EF4444;
    border-radius: 10px; padding: 9px 13px; margin: 5px 0;
    font-size: .8rem; line-height: 1.45;
}
.ppill-warn .plbl {
    font-size: .62rem; text-transform: uppercase; letter-spacing: .12em;
    color: #EF4444 !important; display: block; margin-bottom: 3px; font-weight: 700;
}
.ppill-warn .pval { color: #FCA5A5 !important; font-weight: 600; }

/* ════════════════════════════════════════════════════════
   ANALYTICS STAT CARDS
════════════════════════════════════════════════════════ */
.stat-card {
    background: rgba(255,255,255,.025);
    border: 1px solid rgba(139,92,246,.17);
    border-radius: 18px; padding: 24px 18px; text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,.2);
    transition: border-color .2s, transform .2s;
}
.stat-card:hover { border-color: rgba(139,92,246,.38); transform: translateY(-3px); }
.sc-icon { font-size: 1.65rem; margin-bottom: 10px; }
.sc-val  { font-size: 2rem; font-weight: 800; color: #EDE9FE; line-height: 1; }
.sc-lbl  {
    font-size: .67rem; color: #5B4D8A; margin-top: 8px;
    text-transform: uppercase; letter-spacing: .1em; font-weight: 700;
}

/* ════════════════════════════════════════════════════════
   ALERTS / SPINNERS
════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border: 1px solid rgba(139,92,246,.2) !important;
    background: rgba(139,92,246,.05) !important;
    font-size: .87rem !important;
}
[data-testid="stStatusWidget"] { color: #9D77F5 !important; }

/* ════════════════════════════════════════════════════════
   FEATURE CARDS  (welcome screen)
════════════════════════════════════════════════════════ */
.feat-card {
    background: rgba(139,92,246,.05);
    border: 1px solid rgba(139,92,246,.16);
    border-radius: 18px; padding: 22px 18px; text-align: center;
    transition: border-color .2s, transform .2s, background .2s;
    height: 100%;
}
.feat-card:hover {
    background: rgba(139,92,246,.1);
    border-color: rgba(139,92,246,.32);
    transform: translateY(-3px);
}
.feat-icon { font-size: 2rem; margin-bottom: 12px; }
.feat-title {
    font-size: .9rem; font-weight: 700; color: #C4B5FD;
    margin-bottom: 8px; letter-spacing: -.01em;
}
.feat-desc { font-size: .76rem; color: #4A3A7A; line-height: 1.58; }
</style>
""", unsafe_allow_html=True)


# ── Category helpers ───────────────────────────────────────────────────────────
CAT_CSS  = {"courses":"sc-courses","policy":"sc-policy","admissions":"sc-admissions",
            "research":"sc-research","faculty":"sc-faculty","facilities":"sc-facilities",
            "deadlines":"sc-deadlines","general":"sc-general"}
CAT_ICON = {"courses":"📘","policy":"📋","admissions":"📝","research":"🔬",
            "faculty":"👩‍🏫","facilities":"🏛️","deadlines":"📅","general":"ℹ️"}

SUGGESTIONS = [
    "Plan my next semester (CSAI, sem 5, GPA 2.8, failed Signals)",
    "When will I graduate? (CSAI CS, semester 4, GPA 3.1)",
    "Prerequisite for Machine Learning (CSAI 253)?",
    "What courses can I take after CSAI 101 and CSAI 201?",
    "What scholarships are available for undergrad students?",
    "What is the academic probation policy?",
    "Electives for SCI Biomedical Sciences?",
    "BUS graduation requirements?",
]


# ── Resource loader ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_assistant():
    from phase6_conversational_memory import ConversationalAssistant
    return ConversationalAssistant()


# ── Session state ──────────────────────────────────────────────────────────────
def init_state() -> None:
    if "session" not in st.session_state:
        from phase6_conversational_memory import ConversationSession
        st.session_state["session"] = ConversationSession()
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    if "_pending" not in st.session_state:
        st.session_state["_pending"] = None
    if "_last_audio_hash" not in st.session_state:
        st.session_state["_last_audio_hash"] = None


# ── Header ─────────────────────────────────────────────────────────────────────
def render_header() -> None:
    st.markdown("""
    <div class="zc-header">
        <div class="zc-logo">🎓</div>
        <div class="zc-htxt">
            <h1>Zewail City Campus Assistant</h1>
            <p>Academic Advisor AI &nbsp;·&nbsp; Powered by official Zewail City documents &amp; GPT-4o</p>
        </div>
        <div class="zc-badge">✦ &nbsp;Academic Advisor AI</div>
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> None:
    session = st.session_state["session"]
    profile = session.get_profile()

    with st.sidebar:
        # Branding block
        st.markdown("""
        <div style="text-align:center;padding:22px 0 16px">
            <div style="font-size:2.8rem;filter:drop-shadow(0 0 14px rgba(139,92,246,.55));line-height:1">
                🏛️
            </div>
            <div style="font-weight:900;font-size:1rem;color:#C4B5FD;
                        letter-spacing:.12em;margin-top:10px;text-transform:uppercase">
                Zewail City
            </div>
            <div style="font-size:.66rem;color:#3D3060;margin-top:3px;letter-spacing:.05em">
                University of Science &amp; Technology
            </div>
            <div style="margin:12px auto 0;width:36px;height:2px;
                        background:linear-gradient(90deg,transparent,rgba(139,92,246,.5),transparent)">
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Student Profile ────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:.63rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.14em;color:#6D28D9;padding:2px 0 8px;display:flex;"
            "align-items:center;gap:6px'>🪪 &nbsp;Student Profile</div>",
            unsafe_allow_html=True,
        )

        if profile.has_academic_context():
            for label, value in profile.sidebar_items():
                pill_cls = "ppill-warn" if label == "Must Retake" else "ppill"
                st.markdown(
                    f"<div class='{pill_cls}'>"
                    f"<span class='plbl'>{label}</span>"
                    f"<span class='pval'>{value}</span></div>",
                    unsafe_allow_html=True,
                )

            if profile.gpa and profile.gpa < 2.0:
                st.markdown(
                    "<div style='background:rgba(239,68,68,.09);border:1px solid rgba(239,68,68,.25);"
                    "border-radius:10px;padding:9px 12px;font-size:.74rem;color:#FCA5A5;margin-top:8px;"
                    "line-height:1.5'>"
                    "⚠️ <strong>GPA below probation threshold (2.0).</strong><br>"
                    "Max 12 credits recommended.</div>",
                    unsafe_allow_html=True,
                )
            elif profile.gpa and profile.gpa < 2.5:
                st.markdown(
                    "<div style='background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.22);"
                    "border-radius:10px;padding:9px 12px;font-size:.74rem;color:#FCD34D;margin-top:8px;"
                    "line-height:1.5'>"
                    "⚠️ <strong>GPA in warning zone.</strong><br>"
                    "Consider a balanced load (15 cr).</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='font-size:.78rem;color:#3D3060;padding:4px 0 10px;line-height:1.6'>"
                "Share your program, semester, GPA, and completed or failed courses "
                "for personalised academic advice.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='background:rgba(109,40,217,.07);border:1px solid rgba(139,92,246,.18);"
                "border-left:3px solid rgba(139,92,246,.5);border-radius:0 12px 12px 0;"
                "padding:11px 14px;font-size:.75rem;color:#9D77F5;line-height:1.6'>"
                "<strong style='color:#C4B5FD'>Example prompt:</strong><br>"
                "\"I'm CSAI, semester 5, GPA 2.8. I completed CSAI 101, CSAI 201. "
                "I failed CSAI 261. What should I take next?\"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Session Stats ──────────────────────────────────────────────────────
        st.markdown(
            "<div style='font-size:.63rem;font-weight:700;text-transform:uppercase;"
            "letter-spacing:.14em;color:#6D28D9;padding:2px 0 8px'>📊 &nbsp;Session Stats</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Queries", session.query_count)
        avg = session.total_response_time / max(session.query_count, 1)
        c2.metric("Avg Time", f"{avg:.1f}s")

        if session.topic_counts:
            top = max(session.topic_counts, key=lambda k: session.topic_counts[k])
            st.markdown(
                f"<div style='font-size:.73rem;color:#3D3060;margin-top:6px;padding:0 2px'>"
                f"Most asked: <span style='color:#9D77F5;font-weight:600'>{top}</span></div>",
                unsafe_allow_html=True,
            )

        st.divider()

        if st.button("＋  New Conversation", use_container_width=True):
            from phase6_conversational_memory import ConversationSession
            st.session_state["session"]       = ConversationSession()
            st.session_state["chat_messages"] = []
            st.rerun()

        st.markdown(
            f"<div style='text-align:center;font-size:.62rem;color:#2A1F4A;margin-top:10px'>"
            f"Session · {session.session_id[:8]}…</div>",
            unsafe_allow_html=True,
        )


# ── Welcome / Suggestion screen ────────────────────────────────────────────────
def render_suggestions(compact: bool = False) -> None:
    if not compact:
        # Full welcome screen (only shown when no messages yet)
        st.markdown("""
        <div style="text-align:center;padding:36px 0 28px">
            <div style="font-size:3.4rem;margin-bottom:16px;
                        filter:drop-shadow(0 0 24px rgba(139,92,246,.55))">🎓</div>
            <div style="font-size:1.55rem;font-weight:800;color:#F5F0FF;
                        letter-spacing:-.03em;margin-bottom:8px;line-height:1.2">
                Welcome to Zewail City<br>Campus Assistant
            </div>
            <div style="font-size:.87rem;color:#4A3A7A;max-width:500px;
                        margin:0 auto;line-height:1.65">
                Your AI-powered academic advisor — ask about courses, graduation plans,
                scholarships, faculty, and any Zewail City policy.
            </div>
        </div>
        """, unsafe_allow_html=True)

        features = [
            ("📚", "Course Planning",
             "Personalized course recommendations based on your semester, GPA, and academic history."),
            ("🎯", "Graduation Tracker",
             "Find your graduation roadmap with Safe, Balanced, and Fast plan options."),
            ("📋", "Policies & Resources",
             "Instant answers on academic policies, scholarships, faculty, and campus services."),
        ]
        c1, c2, c3 = st.columns(3)
        for col, (icon, title, desc) in zip([c1, c2, c3], features):
            col.markdown(
                f'<div class="feat-card">'
                f'<div class="feat-icon">{icon}</div>'
                f'<div class="feat-title">{title}</div>'
                f'<div class="feat-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Divider + label (always shown)
    label = "✦ &nbsp; Try one of these" if not compact else "✦ &nbsp; Quick questions"
    st.markdown(f"""
    <div style="margin:{('24px' if not compact else '8px')} 0 10px;
                display:flex;align-items:center;gap:14px">
        <div style="flex:1;height:1px;
                    background:linear-gradient(90deg,transparent,rgba(139,92,246,.18))"></div>
        <div style="font-size:.65rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:.15em;color:#3D3060;white-space:nowrap">{label}</div>
        <div style="flex:1;height:1px;
                    background:linear-gradient(90deg,rgba(139,92,246,.18),transparent)"></div>
    </div>
    """, unsafe_allow_html=True)

    # 2 rows × 4 suggestion pills (always shown)
    for row_start in range(0, len(SUGGESTIONS), 4):
        row  = SUGGESTIONS[row_start : row_start + 4]
        cols = st.columns(len(row))
        for i, (col, sug) in enumerate(zip(cols, row)):
            if col.button(sug, key=f"sug_{row_start + i}"):
                st.session_state["_pending"] = sug
                st.rerun()
        if row_start == 0:
            st.markdown("<div style='height:5px'></div>", unsafe_allow_html=True)

    st.markdown(
        "<hr style='border:none;border-top:1px solid rgba(139,92,246,.1);"
        "margin:14px 0 6px'>",
        unsafe_allow_html=True,
    )


# ── Source cards ───────────────────────────────────────────────────────────────
def render_sources(sources: list[dict], elapsed: float) -> None:
    if not sources:
        return
    cards = ""
    for s in sources:
        cat   = s.get("category", "general")
        icon  = CAT_ICON.get(cat, "ℹ️")
        css   = CAT_CSS.get(cat, "sc-general")
        label = s["source"] + (f" · p.{s['page']}" if s.get("page") else "")
        score = int(s["score"] * 100)
        exc   = s["text"][:200].replace("<", "&lt;").replace(">", "&gt;")
        if len(s["text"]) > 200:
            exc += "…"
        cards += (
            f'<div class="src-card {css}">'
            f'<div class="sc-name">{icon} {cat.title()}</div>'
            f'<div class="sc-meta">{label}</div>'
            f'<span class="sc-score">▲ {score}% match</span>'
            f'<div class="sc-exc">{exc}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="src-grid" style="padding:0 2px">{cards}</div>',
                unsafe_allow_html=True)


# ── Voice transcription helper ─────────────────────────────────────────────────
def _transcribe(audio_file) -> str:
    """Send recorded audio to OpenAI Whisper and return transcript text."""
    import os
    from openai import OpenAI as _OAI
    client = _OAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=("question.wav", audio_file, "audio/wav"),
    )
    return transcript.text.strip()


# ── Chat renderer ──────────────────────────────────────────────────────────────
def render_chat(assistant) -> None:
    session = st.session_state["session"]
    msgs    = st.session_state["chat_messages"]

    render_suggestions(compact=bool(msgs))

    # ── Render history ─────────────────────────────────────────────────────────
    for msg in msgs:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            content    = _clean_answer(msg["content"])
            is_advisor = msg.get("is_advisor", False) or _is_advisor_response(content)
            with st.chat_message("assistant", avatar="🎓"):
                if is_advisor:
                    st.markdown(
                        '<span class="advisor-badge">🎓 &nbsp;Academic Advisor Response</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(content)

            srcs = msg.get("sources", [])
            if srcs:
                with st.expander(
                    f"📚 {len(srcs)} source{'s' if len(srcs) > 1 else ''}"
                    f"  ·  {msg.get('elapsed', 0):.1f}s",
                    expanded=False,
                ):
                    render_sources(srcs, msg.get("elapsed", 0))

            followups = msg.get("followups", [])
            if followups:
                st.markdown(
                    "<div style='font-size:.68rem;color:#3D3060;margin:6px 0 4px 46px;"
                    "text-transform:uppercase;letter-spacing:.1em'>✦ Follow-up</div>",
                    unsafe_allow_html=True,
                )
                fu_cols = st.columns(len(followups))
                for fi, (fcol, fsug) in enumerate(zip(fu_cols, followups)):
                    if fcol.button(fsug, key=f"fu_{id(msg)}_{fi}"):
                        st.session_state["_pending"] = fsug
                        st.rerun()

    # ── Voice input ────────────────────────────────────────────────────────────
    with st.expander("🎤 Voice Input", expanded=False):
        st.markdown(
            "<div style='font-size:.76rem;color:#4A3A7A;margin-bottom:8px'>"
            "Record your question — it will be transcribed automatically.</div>",
            unsafe_allow_html=True,
        )
        try:
            audio_val = st.audio_input("Record", label_visibility="collapsed", key="voice_rec")
            if audio_val is not None:
                raw = audio_val.read()
                ahash = hash(raw)
                if st.session_state.get("_last_audio_hash") != ahash:
                    st.session_state["_last_audio_hash"] = ahash
                    with st.spinner("Transcribing…"):
                        try:
                            import io
                            text = _transcribe(io.BytesIO(raw))
                            if text:
                                st.caption(f"Heard: *{text}*")
                                st.session_state["_pending"] = text
                                st.rerun()
                        except Exception as e:
                            st.warning(f"Transcription failed: {e}")
        except Exception:
            st.caption("Voice input not available in this environment.")

    # ── Text input ─────────────────────────────────────────────────────────────
    pending    = st.session_state.pop("_pending", None)
    user_input = st.chat_input(
        "Ask anything — courses, prerequisites, graduation plan, scholarships, policies…"
    )
    question = pending or user_input

    if not question:
        return

    # ── Render new exchange live (streaming) ───────────────────────────────────
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    elapsed = 0.0
    chunks  = []
    answer  = ""
    with st.chat_message("assistant", avatar="🎓"):
        try:
            t0 = time.time()
            with st.spinner("Retrieving…"):
                result_chunks, content, is_streamed = assistant.ask_stream(question, session)
            chunks = result_chunks

            if is_streamed:
                answer = st.write_stream(content)
            else:
                clean  = _clean_answer(content)
                is_adv = _is_advisor_response(clean)
                if is_adv:
                    st.markdown(
                        '<span class="advisor-badge">🎓 &nbsp;Academic Advisor Response</span>',
                        unsafe_allow_html=True,
                    )
                st.markdown(clean)
                answer = clean

            elapsed = time.time() - t0

        except Exception as exc:
            answer = f"**Error:** {exc}"
            st.markdown(answer)

    # Sources (outside chat bubble)
    srcs = [
        {"source": c.source, "page": c.page, "category": c.category,
         "score": c.score, "text": c.text}
        for c in chunks
    ]
    if srcs:
        with st.expander(
            f"📚 {len(srcs)} source{'s' if len(srcs) > 1 else ''}  ·  {elapsed:.1f}s",
            expanded=False,
        ):
            render_sources(srcs, elapsed)

    # Follow-up suggestions
    followups: list[str] = []
    try:
        followups = assistant._rag.suggest_followups(question, _clean_answer(answer), n=3)
    except Exception:
        pass
    if followups:
        st.markdown(
            "<div style='font-size:.68rem;color:#3D3060;margin:8px 0 4px;"
            "text-transform:uppercase;letter-spacing:.1em'>✦ Follow-up questions</div>",
            unsafe_allow_html=True,
        )
        fu_cols = st.columns(len(followups))
        for fi, (fcol, fsug) in enumerate(zip(fu_cols, followups)):
            if fcol.button(fsug, key=f"new_fu_{fi}"):
                st.session_state["_pending"] = fsug
                st.rerun()

    # Persist to display history
    clean_answer = _clean_answer(answer)
    msgs.append({"role": "user", "content": question})
    msgs.append({
        "role":       "assistant",
        "content":    clean_answer,
        "is_advisor": _is_advisor_response(clean_answer),
        "sources":    srcs,
        "elapsed":    elapsed,
        "followups":  followups,
    })
    st.rerun()


# ── Analytics tab ──────────────────────────────────────────────────────────────
def render_analytics() -> None:
    session = st.session_state["session"]
    profile = session.get_profile()

    if session.query_count == 0:
        st.markdown("""
        <div style="text-align:center;padding:72px 0;color:#3D3060">
            <div style="font-size:3.2rem;margin-bottom:14px;opacity:.6">📊</div>
            <div style="font-size:1.1rem;font-weight:700;color:#5B4D8A;margin-bottom:6px">
                No session data yet
            </div>
            <div style="font-size:.84rem;color:#3D3060">
                Start a conversation in the Chat tab to generate analytics.
            </div>
        </div>""", unsafe_allow_html=True)
        return

    avg = session.total_response_time / max(session.query_count, 1)
    cards_data = [
        ("🗨️", session.query_count,             "Total Queries"),
        ("⚡", f"{avg:.1f}s",                    "Avg Response"),
        ("🏷️", len(session.topic_counts),         "Topics"),
        ("💬", len(session.conversation_history), "History Turns"),
    ]
    cols = st.columns(4)
    for col, (icon, val, lbl) in zip(cols, cards_data):
        col.markdown(
            f'<div class="stat-card">'
            f'<div class="sc-icon">{icon}</div>'
            f'<div class="sc-val">{val}</div>'
            f'<div class="sc-lbl">{lbl}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if session.topic_counts:
        st.markdown(
            "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
            "text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>"
            "🏷️ &nbsp;Topics discussed</div>",
            unsafe_allow_html=True,
        )
        import pandas as pd
        df = (pd.DataFrame(list(session.topic_counts.items()), columns=["Topic", "Count"])
              .sort_values("Count", ascending=False))
        st.bar_chart(df.set_index("Topic"), color="#8B5CF6")

    if profile.has_academic_context():
        st.markdown(
            "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
            "text-transform:uppercase;letter-spacing:.1em;margin:18px 0 10px'>"
            "🪪 &nbsp;Student Profile</div>",
            unsafe_allow_html=True,
        )
        for label, value in profile.sidebar_items():
            st.markdown(f"- **{label}**: {value}")
        if profile.completed_courses:
            with st.expander(f"Completed courses ({len(profile.completed_courses)})", expanded=False):
                for c in profile.completed_courses:
                    st.markdown(f"- {c}")
        if profile.failed_courses:
            with st.expander(f"Courses to retake ({len(profile.failed_courses)})", expanded=False):
                for c in profile.failed_courses:
                    st.markdown(f"- ⚠️ {c}")

    st.markdown(
        "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
        "text-transform:uppercase;letter-spacing:.1em;margin:18px 0 10px'>"
        "💬 &nbsp;Conversation History</div>",
        unsafe_allow_html=True,
    )
    for msg in session.conversation_history:
        is_user = msg["role"] == "user"
        border  = "#8B5CF6" if is_user else "#EC4899"
        who     = "You" if is_user else "Assistant"
        txt     = msg["content"][:220] + ("…" if len(msg["content"]) > 220 else "")
        st.markdown(
            f"<div style='border-left:3px solid {border};padding:8px 14px;margin:5px 0;"
            f"background:rgba(255,255,255,.02);border-radius:0 12px 12px 0;font-size:.8rem'>"
            f"<span style='font-weight:700;font-size:.68rem;text-transform:uppercase;"
            f"letter-spacing:.08em;color:{border}'>{who}</span><br>"
            f"<span style='color:#4A3A7A;line-height:1.55'>{txt}</span></div>",
            unsafe_allow_html=True,
        )


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    init_state()
    render_header()

    with st.spinner("Loading AI engine…"):
        try:
            assistant = get_assistant()
            rag_ok    = True
        except Exception as exc:
            rag_ok = False
            err    = str(exc)

    tab_chat, tab_analytics, tab_xai, tab_eval, tab_clust, tab_fcast = st.tabs([
        "💬  Chat",
        "📊  Analytics",
        "🧠  Learning Analytics & XAI",
        "🔬  RAG Evaluation",
        "🧩  Student Archetypes",
        "📈  GPA Forecast",
    ])

    with tab_chat:
        render_sidebar()
        if not rag_ok:
            st.error(
                f"**Pipeline not ready:** {err}  \n"
                "Run phases 1–4 first and set `OPENAI_API_KEY` in `.env`."
            )
        else:
            render_chat(assistant)

    with tab_analytics:
        render_analytics()

    with tab_xai:
        if not _XAI_IMPORT_OK:
            st.error(f"**Learning Analytics import failed:** {_XAI_IMPORT_ERR}")
        else:
            try:
                _render_xai_page()
            except Exception as _xai_err:
                import traceback as _tb
                st.error(f"**Learning Analytics error:** {_xai_err}")
                st.code(_tb.format_exc(), language="python")

        # ── XAI → RAG Advisor Bridge ───────────────────────────────────────────
        if st.session_state.get("xai_analysed"):
            st.divider()
            st.markdown(
                "<div style='font-size:.75rem;font-weight:700;color:#9D77F5;"
                "text-transform:uppercase;letter-spacing:.1em;margin-bottom:10px'>"
                "🎓 &nbsp;Consult the Academic Advisor</div>",
                unsafe_allow_html=True,
            )
            flow    = st.session_state.get("xai_flow", {})
            profile = st.session_state.get("xai_profile", {})
            result  = st.session_state.get("xai_result", {})

            prog   = flow.get("programme", "")
            sem    = flow.get("semester", "")
            risk   = getattr(result, "risk_label", "unknown risk") if result else "unknown risk"
            mid    = profile.get("avg_midterm",  "?")
            fin    = profile.get("avg_final",    "?")
            att    = profile.get("avg_attendance","?")
            failed = profile.get("failed_courses", 0)

            default_q = (
                f"I'm a {prog} student in semester {sem}. "
                f"My XAI risk assessment is '{risk}'. "
                f"Avg midterm {mid}%, final {fin}%, attendance {att}%, "
                f"failed courses {failed}. "
                f"What should I do to improve my academic standing?"
            )

            bridge_q = st.text_area(
                "Pre-filled from your XAI profile — edit if needed:",
                value=default_q,
                height=90,
                key="xai_bridge_q",
            )
            if st.button("Send to Academic Advisor →", key="xai_bridge_btn"):
                if rag_ok and bridge_q.strip():
                    with st.spinner("Getting advisor response…"):
                        try:
                            adv_answer, adv_chunks = assistant.ask(bridge_q.strip(), st.session_state["session"])
                        except Exception as _be:
                            adv_answer  = f"**Error:** {_be}"
                            adv_chunks  = []
                    st.markdown("**Advisor says:**")
                    st.markdown(_clean_answer(adv_answer))
                    if adv_chunks:
                        with st.expander(f"📚 {len(adv_chunks)} sources"):
                            render_sources(
                                [{"source": c.source, "page": c.page, "category": c.category,
                                  "score": c.score, "text": c.text} for c in adv_chunks],
                                0,
                            )
                    # Also save to chat history so it appears in Chat tab
                    st.session_state["chat_messages"].append(
                        {"role": "user", "content": bridge_q.strip()}
                    )
                    st.session_state["chat_messages"].append({
                        "role":       "assistant",
                        "content":    _clean_answer(adv_answer),
                        "is_advisor": _is_advisor_response(adv_answer),
                        "sources":    [],
                        "elapsed":    0,
                        "followups":  [],
                    })

    with tab_eval:
        from rag_evaluator import render_evaluation_tab
        if rag_ok:
            render_evaluation_tab(assistant)
        else:
            st.warning("Load the RAG pipeline first (set OPENAI_API_KEY and ensure the vector DB exists).")

    with tab_clust:
        if not _CLUST_IMPORT_OK:
            st.error(f"**Clustering import failed:** {_CLUST_IMPORT_ERR}")
        else:
            try:
                _oai = assistant._rag._oai_chat if rag_ok and assistant else None
                _render_clustering_page(openai_client=_oai)
            except Exception as _ce:
                import traceback as _tb
                st.error(f"**Clustering error:** {_ce}")
                st.code(_tb.format_exc(), language="python")

    with tab_fcast:
        if not _FCAST_IMPORT_OK:
            st.error(f"**Forecasting import failed:** {_FCAST_IMPORT_ERR}")
        else:
            try:
                _render_forecasting_page()
            except Exception as _fe:
                import traceback as _tb
                st.error(f"**Forecasting error:** {_fe}")
                st.code(_tb.format_exc(), language="python")


if __name__ == "__main__":
    main()
