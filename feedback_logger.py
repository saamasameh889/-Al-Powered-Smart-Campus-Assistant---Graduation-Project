"""
feedback_logger.py — Query logging, thumbs feedback, gap analysis, prompt patches.
Used by Features 5.3 (escalation logging), 5.6 (feedback loop), and the admin dashboard.

Log file  : db/query_log.jsonl   (one JSON object per line, append-only)
Patches   : db/prompt_patches.json  (list of admin-approved system prompt rules)
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

_BASE = Path(__file__).parent / "db"
LOG_FILE     = _BASE / "query_log.jsonl"
PATCHES_FILE = _BASE / "prompt_patches.json"

# Confidence threshold below which a query is auto-flagged
LOW_CONFIDENCE_THRESHOLD = 0.45


def _ensure_dirs() -> None:
    _BASE.mkdir(parents=True, exist_ok=True)


# ── Query logging ──────────────────────────────────────────────────────────────

def log_query(
    *,
    question:       str,
    intent:         str,
    max_score:      float,
    answer_preview: str,
    session_id:     str = "",
    contact_key:    str = "",
) -> str:
    """
    Append a new query entry. Returns the entry_id.
    Queries are auto-flagged when max_score < LOW_CONFIDENCE_THRESHOLD.
    """
    _ensure_dirs()
    entry_id = str(uuid.uuid4())[:8]
    entry = {
        "entry_id":       entry_id,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "session_id":     session_id,
        "question":       question,
        "intent":         intent,
        "max_score":      round(max_score, 4),
        "answer_preview": answer_preview[:400],
        "contact_key":    contact_key,
        "flagged":        max_score < LOW_CONFIDENCE_THRESHOLD,
        "thumb":          None,   # "up" / "down" / None
        "ai_note":        None,   # filled by gap analysis
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry_id


def update_thumb(entry_id: str, thumb: str) -> None:
    """Set thumb to 'up' or 'down' for a given entry_id. Re-flags down-votes."""
    if not LOG_FILE.exists():
        return
    lines   = LOG_FILE.read_text(encoding="utf-8").splitlines()
    updated = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            if entry.get("entry_id") == entry_id:
                entry["thumb"] = thumb
                if thumb == "down":
                    entry["flagged"] = True
        except json.JSONDecodeError:
            pass
        updated.append(json.dumps(entry, ensure_ascii=False))
    LOG_FILE.write_text("\n".join(updated) + "\n", encoding="utf-8")


# ── Query retrieval ────────────────────────────────────────────────────────────

def get_all_entries() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    entries = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def get_flagged_entries() -> list[dict]:
    """Return entries that are auto-flagged (low score) or thumbed down."""
    return [
        e for e in get_all_entries()
        if e.get("flagged") or e.get("thumb") == "down"
    ]


def get_stats() -> dict:
    entries  = get_all_entries()
    total    = len(entries)
    thumbs   = [e for e in entries if e.get("thumb")]
    up_count = sum(1 for e in thumbs if e["thumb"] == "up")
    dn_count = sum(1 for e in thumbs if e["thumb"] == "down")
    flagged  = sum(1 for e in entries if e.get("flagged"))
    scores   = [e["max_score"] for e in entries if "max_score" in e]
    avg_sc   = round(sum(scores) / len(scores), 3) if scores else 0.0
    return {
        "total":    total,
        "flagged":  flagged,
        "up":       up_count,
        "down":     dn_count,
        "avg_score": avg_sc,
    }


# ── AI gap analysis ────────────────────────────────────────────────────────────

def run_ai_gap_analysis(openai_api_key: str) -> str:
    """
    Send all flagged / thumbs-down queries to GPT-4o for analysis.
    Returns a markdown report string.
    """
    from openai import OpenAI

    flagged = get_flagged_entries()
    if not flagged:
        return "✅ No flagged queries to analyse — the knowledge base appears to be covering student needs well."

    query_list = "\n".join(
        f"{i+1}. [{e['timestamp'][:10]}] score={e['max_score']:.2f} "
        f"thumb={e.get('thumb','—')}  \"{e['question']}\""
        for i, e in enumerate(flagged[:60])
    )

    prompt = f"""You are an AI knowledge-base curator for the Zewail City Campus Assistant — a RAG chatbot for Zewail City of Science and Technology students.

Below are student queries that the system struggled to answer (retrieval confidence < 0.45 or student gave thumbs-down):

{query_list}

Analyse these queries and produce a structured improvement report with exactly these sections:

## 1. Knowledge Gap Summary
What topics or information is missing from the knowledge base? Group related queries.

## 2. Specific Content to Add
For each gap, describe the exact content that should be ingested (document type, section, specific facts).

## 3. System Prompt Rule Suggestions
Suggest specific new rules to add to the assistant's system prompt (e.g. "When asked about X, always Y").
Format each suggestion as a ready-to-paste rule starting with a number.

## 4. Priority Ranking
Rank the top 5 most impactful gaps with a one-sentence justification each.

## 5. Pattern Observations
Are there recurring phrasing patterns, languages (Arabic?), or student confusion points?

Be concrete and actionable. Avoid vague advice."""

    client = OpenAI(api_key=openai_api_key)
    resp   = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.25,
    )
    return resp.choices[0].message.content


# ── Prompt patches ─────────────────────────────────────────────────────────────

def load_prompt_patches() -> list[dict]:
    """Return the list of admin-approved system-prompt patches."""
    if not PATCHES_FILE.exists():
        return []
    try:
        data = json.loads(PATCHES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_prompt_patch(rule_text: str, label: str = "") -> str:
    """Append a new patch. Returns the patch id."""
    _ensure_dirs()
    patches  = load_prompt_patches()
    patch_id = str(uuid.uuid4())[:8]
    patches.append({
        "patch_id":  patch_id,
        "created":   datetime.now().isoformat(timespec="seconds"),
        "label":     label or rule_text[:60],
        "rule":      rule_text.strip(),
        "active":    True,
    })
    PATCHES_FILE.write_text(
        json.dumps(patches, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return patch_id


def delete_prompt_patch(patch_id: str) -> None:
    patches = [p for p in load_prompt_patches() if p.get("patch_id") != patch_id]
    PATCHES_FILE.write_text(
        json.dumps(patches, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_patched_prompt(base_prompt: str) -> str:
    """Return base_prompt with all active patches appended."""
    patches = [p for p in load_prompt_patches() if p.get("active", True)]
    if not patches:
        return base_prompt
    rules = "\n".join(
        f"{i+1}. {p['rule']}"
        for i, p in enumerate(patches)
    )
    return (
        base_prompt
        + f"\n\n# ADMIN-APPROVED IMPROVEMENTS (applied {datetime.now().strftime('%Y-%m-%d')})\n"
        + rules
    )
