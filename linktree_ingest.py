#!/usr/bin/env python3
"""
linktree_ingest.py  —  Zewail City Campus Assistant
═══════════════════════════════════════════════════════════════════════════════
Fetches content from the Zewail City Student LinkTree Google Docs and adds
it to the existing ChromaDB vector store.

Usage:
    python linktree_ingest.py              # fetch all registered docs
    python linktree_ingest.py --list       # show registered docs and exit
    python linktree_ingest.py --dry-run    # fetch + chunk but don't embed/store

HOW TO ADD A NEW DOCUMENT:
  1. Open the Google Doc
  2. Share → "Anyone with the link" → Viewer
  3. Copy the document ID from the URL:
     https://docs.google.com/document/d/<DOC_ID>/edit
  4. Add an entry to LINKTREE_DOCS below
  5. Re-run this script
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Iterator

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT    = Path(__file__).parent
CHROMA_DIR      = PROJECT_ROOT / "db" / "chroma_db"
COLLECTION_NAME = "zewail_campus"
EMBED_MODEL     = "text-embedding-3-small"
EMBED_BATCH     = 50
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
MIN_CHUNK_LEN   = 60

# ══════════════════════════════════════════════════════════════════════════════
#  REGISTERED LINKTREE DOCUMENTS
#  ─────────────────────────────────────────────────────────────────────────────
#  Add each sub-document here after making it publicly accessible
#  (Share → Anyone with the link → Viewer).
#
#  Fields:
#    doc_id   : Google Doc ID from the URL
#    label    : human-readable name (used as source in ChromaDB)
#    category : one of: courses | tools | guide | socials | extra | linktree
#    section  : the LinkTree section this belongs to
# ══════════════════════════════════════════════════════════════════════════════

LINKTREE_DOCS = [
    # ── Main index ────────────────────────────────────────────────────────────
    {
        "doc_id":   "1x61s5Y3Xz36GB7Tz7B0R2gfUF6VXXMs8NjbH5z9ka6E",
        "label":    "Zewail City Student LinkTree (Main Index)",
        "category": "linktree",
        "section":  "root",
    },
    # ── Courses ───────────────────────────────────────────────────────────────
    {
        "doc_id":   "17uIMIfaCAiazIK53pHY33GH_JldF_24-qLr_MtyFE-c",
        "label":    "LinkTree — Computer Science Courses (CSAI, DSAI, SW, HCI, IT)",
        "category": "courses",
        "section":  "Courses",
    },
    {
        "doc_id":   "1DpmNY24lBGnJKbUWiTSWcCEMXVctAHJUBvRER8oubYU",
        "label":    "LinkTree — Mathematics Courses (MATH 101–307)",
        "category": "courses",
        "section":  "Courses",
    },
    {
        "doc_id":   "1kYSafBd9ii4Whck6D1hNm9p6XXJGQOpf995cJ6pleiE",
        "label":    "LinkTree — Engineering Courses (CIE, ENGR, SPC, REE)",
        "category": "courses",
        "section":  "Courses",
    },
    {
        "doc_id":   "1PD7PZ4pPVRHcHVelXQxLpn5M1HJrxwsmRrBjBDNqL8s",
        "label":    "LinkTree — General Education Courses (English ENGL, Humanities SCH)",
        "category": "courses",
        "section":  "Courses",
    },
    {
        "doc_id":   "1KnQ5HCPVxjhntHtXQZ8EHeS_mHG2Lfzdx-u0KvDJOSs",
        "label":    "LinkTree — Courses Index (All Majors)",
        "category": "courses",
        "section":  "Courses",
    },
    {
        "doc_id":   "1UDKznWf3Yro2Rw8H8sHHMmHwTxu78Lg8MhX1EPoFeck",
        "label":    "LinkTree — Course Materials and Books (Math, CSAI, DSAI, SWD, SCH)",
        "category": "courses",
        "section":  "Courses",
    },
    {
        "doc_id":   "1j8qSKShjGkVAr-CbeHhOffphRLmQZgB4RXOAdWzjJ34",
        "label":    "LinkTree — Student Drive Collections (Day One, Foundation, CIE, BMS, ENV)",
        "category": "courses",
        "section":  "Courses",
    },
    # ── Tools & Resources ─────────────────────────────────────────────────────
    {
        "doc_id":   "1ACSuOBTeFa6vcJuPP3RJRTrhCZRbA2L06Fq3M72KCF8",
        "label":    "LinkTree — Coding Practice Platforms (HackerRank, LeetCode, CodeChef)",
        "category": "tools",
        "section":  "Tools",
    },
    {
        "doc_id":   "1E0TPoMprNY1WwO7YPfZG2HwSa-OZLPiqwExbq8fRm4c",
        "label":    "LinkTree — Tools & Services (Self-Service Portal, VPN, LTS, Office Hours)",
        "category": "tools",
        "section":  "Tools",
    },
    {
        "doc_id":   "10Y2Oo-JrDApqCJkCFdFkcoXvygclBUB4vyltAjauWVY",
        "label":    "LinkTree — Guide & Info (Academic Regulations, Handbooks)",
        "category": "guide",
        "section":  "Guide & info",
    },
    {
        "doc_id":   "1eFvIFMbwjOHDsKnyE2vF17no_KGrzf3AIdtJok14BGI",
        "label":    "LinkTree — Guide & Info (Office Hours, Staff Info)",
        "category": "guide",
        "section":  "Guide & info",
    },
    {
        "doc_id":   "1bt_Lo4bNkzWmiPJM22xeqxjP_haftxZcHtTgwDFvK6g",
        "label":    "LinkTree — Guide & Info (Campus Navigation)",
        "category": "guide",
        "section":  "Guide & info",
    },
    # ── Socials ───────────────────────────────────────────────────────────────
    {
        "doc_id":   "16NQC9RIQ203dMnZcasaY6laKYU3YRx_krbI_I_LyJZ8",
        "label":    "LinkTree — Socials (WhatsApp Groups, Clubs, Academic Helper Groups)",
        "category": "socials",
        "section":  "Socials",
    },
    # ── Extra ─────────────────────────────────────────────────────────────────
    {
        "doc_id":   "1cAowixGXc8QoGx_Gf11GabldvSJOlB4ACx3gKgx9b4g",
        "label":    "LinkTree — Extra Material",
        "category": "extra",
        "section":  "Extra",
    },
]

# ══════════════════════════════════════════════════════════════════════════════
#  MANUAL DOCUMENTS (paste content directly when docs can't be made public)
# ══════════════════════════════════════════════════════════════════════════════

MANUAL_DOCS: list[dict] = [
    # Example:
    # {
    #     "id":       "linktree_manual_offices_001",
    #     "label":    "Staff Offices and Room Numbers",
    #     "category": "guide",
    #     "section":  "Guide & info",
    #     "text":     """
    #         Dr. Ahmed Hassan — Office B204 — Office Hours: Sun/Tue 10-12
    #         Dr. Sara Mahmoud — Office A108 — Office Hours: Mon/Wed 2-4
    #         ...
    #     """,
    # },
]


# ══════════════════════════════════════════════════════════════════════════════
#  Google Docs fetcher
# ══════════════════════════════════════════════════════════════════════════════

def _gdoc_text(doc_id: str) -> str | None:
    """
    Fetch plain text from a Google Doc via the export URL.
    Returns None if the doc is private / not accessible.
    """
    url = f"https://docs.google.com/document/d/{doc_id}/export?format=txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace").lstrip("﻿")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return None      # private doc — not an error we can fix here
        print(f"    HTTP {e.code} fetching {doc_id}")
        return None
    except Exception as exc:
        print(f"    Error fetching {doc_id}: {exc}")
        return None


def _gdoc_links(doc_id: str) -> list[str]:
    """
    Download DOCX export and extract all hyperlinks from the relationships XML.
    Used to discover nested documents.
    """
    import io
    import zipfile

    url = f"https://docs.google.com/document/d/{doc_id}/export?format=docx"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
    except Exception:
        return []

    links = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            rels_path = "word/_rels/document.xml.rels"
            if rels_path in z.namelist():
                xml = z.read(rels_path).decode("utf-8", errors="replace")
                links = re.findall(r'Target="(https?://[^"]+)"', xml)
    except Exception:
        pass
    return [l for l in links if "google.com" in l]


# ══════════════════════════════════════════════════════════════════════════════
#  Chunker (same algorithm as phase4_chunk_and_embed.py)
# ══════════════════════════════════════════════════════════════════════════════

def _sliding_window(text: str, size: int, overlap: int) -> Iterator[str]:
    start = 0
    while start < len(text):
        chunk = text[start: start + size].strip()
        if chunk:
            yield chunk
        if start + size >= len(text):
            break
        start += size - overlap


def _chunk(text: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    raw: list[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_SIZE:
            raw.append(para)
        else:
            raw.extend(_sliding_window(para, CHUNK_SIZE, CHUNK_OVERLAP))

    merged: list[str] = []
    buf = ""
    for ch in raw:
        if buf:
            candidate = buf + "\n\n" + ch
            if len(candidate) <= CHUNK_SIZE + CHUNK_OVERLAP:
                buf = candidate
                continue
            merged.append(buf)
            buf = ch
        else:
            buf = ch
    if buf:
        merged.append(buf)

    return [c for c in merged if len(c) >= MIN_CHUNK_LEN]


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(dry_run: bool = False) -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not dry_run:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print("LinkTree Ingest — Zewail City Campus Assistant")
    print("=" * 62)

    all_chunks: list[dict] = []
    fetched_ok  = 0
    fetched_err = 0
    discovered_nested: list[str] = []

    # ── 1. Registered Google Docs ──────────────────────────────────────────────
    for entry in LINKTREE_DOCS:
        doc_id  = entry["doc_id"]
        label   = entry["label"]
        cat     = entry["category"]
        section = entry["section"]
        print(f"\n  [{cat.upper()}] {label[:55]}")
        print(f"    Doc ID : {doc_id}")

        text = _gdoc_text(doc_id)
        if text is None:
            print("    [!] Not accessible (private). Make it public or add content manually.")
            fetched_err += 1

            # Still try to extract nested links via DOCX
            nested = _gdoc_links(doc_id)
            if nested:
                print(f"    -> Found {len(nested)} nested link(s) inside this doc:")
                for lnk in nested:
                    print(f"      {lnk}")
                discovered_nested.extend(nested)
            continue

        text = text.strip()
        if len(text) < MIN_CHUNK_LEN:
            print(f"    [!] Content too short ({len(text)} chars) — likely image-only doc.")
            fetched_err += 1
            continue

        chunks = _chunk(text)
        print(f"    OK Fetched {len(text):,} chars -> {len(chunks)} chunks")
        fetched_ok += 1

        for i, ch in enumerate(chunks):
            all_chunks.append({
                "chunk_id":    f"linktree_{doc_id[:12]}_c{i:04d}",
                "text":        ch,
                "doc_id":      f"linktree_{doc_id[:12]}",
                "source_type": "linktree",
                "source":      label,
                "page":        "",
                "category":    cat,
                "section":     section,
            })

        # Extract nested links from this doc too
        nested = _gdoc_links(doc_id)
        for lnk in nested:
            if lnk not in discovered_nested:
                discovered_nested.append(lnk)

    # ── 2. Manual documents ────────────────────────────────────────────────────
    for entry in MANUAL_DOCS:
        label = entry["label"]
        cat   = entry.get("category", "guide")
        print(f"\n  [MANUAL] {label[:55]}")
        chunks = _chunk(entry["text"])
        print(f"    OK {len(chunks)} chunks from manual content")
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "chunk_id":    f"{entry['id']}_c{i:04d}",
                "text":        ch,
                "doc_id":      entry["id"],
                "source_type": "linktree_manual",
                "source":      label,
                "page":        "",
                "category":    cat,
                "section":     entry.get("section", ""),
            })

    # ── 3. Report nested links discovered ─────────────────────────────────────
    if discovered_nested:
        print(f"\n  Discovered {len(discovered_nested)} nested link(s) across all fetched docs:")
        for lnk in discovered_nested:
            m = re.search(r"/document/d/([^/\?]+)", lnk)
            if m:
                nested_id = m.group(1)
                print(f"    -> https://docs.google.com/document/d/{nested_id}/edit")
            else:
                print(f"    -> {lnk}")
        print("  Add these IDs to LINKTREE_DOCS or MANUAL_DOCS to ingest them.")

    # ── 4. Summary before embedding ────────────────────────────────────────────
    print(f"\n  Docs fetched OK   : {fetched_ok}")
    print(f"  Docs failed/private: {fetched_err}")
    print(f"  Manual docs       : {len(MANUAL_DOCS)}")
    print(f"  Total chunks      : {len(all_chunks)}")

    if not all_chunks:
        print("\n  Nothing to embed. Make sub-docs public or add MANUAL_DOCS entries.")
        return

    if dry_run:
        print("\n  [DRY RUN] Skipping embed/store step.")
        for ch in all_chunks[:3]:
            print(f"\n  --- Sample chunk ({ch['source']}) ---")
            print(ch["text"][:300])
        return

    # ── 5. Embed and upsert into existing ChromaDB collection ─────────────────
    print(f"\n  Embedding {len(all_chunks)} chunks into ChromaDB ...")
    import chromadb
    from openai import OpenAI

    oai = OpenAI(api_key=api_key)
    db  = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        col = db.get_collection(COLLECTION_NAME)
    except Exception:
        col = db.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    done = 0
    for batch_start in range(0, len(all_chunks), EMBED_BATCH):
        batch = all_chunks[batch_start: batch_start + EMBED_BATCH]
        texts = [c["text"] for c in batch]
        ids   = [c["chunk_id"] for c in batch]
        metas = [{k: v for k, v in c.items() if k not in ("text", "chunk_id")} for c in batch]

        try:
            resp = oai.embeddings.create(model=EMBED_MODEL, input=texts)
            embeddings = [item.embedding for item in resp.data]
            col.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
            done += len(batch)
            print(f"  Upserted {done}/{len(all_chunks)} ...", end="\r")
            time.sleep(0.2)
        except Exception as exc:
            print(f"\n  ERROR: {exc}")

    print(f"\n  Done. ChromaDB collection '{COLLECTION_NAME}' now has {col.count()} vectors total.")


# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",  action="store_true", help="Fetch + chunk but skip embedding")
    ap.add_argument("--list",     action="store_true", help="List registered docs and exit")
    args = ap.parse_args()

    if args.list:
        print("Registered LinkTree documents:")
        for d in LINKTREE_DOCS:
            print(f"  [{d['category']}] {d['label']}")
            print(f"    https://docs.google.com/document/d/{d['doc_id']}/edit")
        sys.exit(0)

    run(dry_run=args.dry_run)
