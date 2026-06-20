#!/usr/bin/env python3
"""
compare_career_advisor.py
==========================
2-way comparison between:

  Model A — Fine-tuned Seq2Seq (from scratch, trained on 2K GH Archive pairs)
             Loaded from: seq2seq_github_finetuned.pt

  Model B — GPT-4o zero-shot (production model, same prompt, no fine-tuning)

Pipeline:
  Step 1 — Generate test portfolio prompts (4 synthetic profiles)
  Step 2 — Run fine-tuned Seq2Seq on all prompts
  Step 3 — Run GPT-4o zero-shot on all prompts
  Step 4 — Score with BLEU-1, Token F1, Section Coverage, Keyword Grounding
  Step 5 — Print report + save comparison plot

Run:
    python compare_career_advisor.py --api-key sk-...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

PAIRS_PATH      = Path("github_training_pairs.jsonl")
CHECKPOINT_PATH = Path("seq2seq_github_20k_finetuned.pt")
FT_DATA_PATH    = Path("gpt_finetune_data.jsonl")
PLOT_PATH       = Path("career_advisor_comparison.png")

SEQ2SEQ_COLOR = "#55A868"   # green
GPTFT_COLOR   = "#DD8452"   # orange


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — CONVERT PAIRS TO OPENAI FORMAT & UPLOAD
# ══════════════════════════════════════════════════════════════════════════════

def convert_to_openai_format(pairs_path: Path, out_path: Path) -> None:
    """Convert github_training_pairs.jsonl → OpenAI fine-tuning JSONL format."""
    pairs = [json.loads(l) for l in pairs_path.read_text().splitlines() if l.strip()]

    with open(out_path, "w") as f:
        for p in pairs:
            record = {
                "messages": [
                    {"role": "system",
                     "content": "You are an expert GitHub portfolio auditor and tech industry HR advisor."},
                    {"role": "user",    "content": p["prompt"]},
                    {"role": "assistant", "content": p["advisory"]},
                ]
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Converted {len(pairs):,} pairs → {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")


def upload_and_finetune(client, out_path: Path) -> str:
    """Upload training file and start a GPT-4o-mini fine-tuning job."""
    print("  Uploading training file to OpenAI …")
    with open(out_path, "rb") as f:
        file_resp = client.files.create(file=f, purpose="fine-tune")
    file_id = file_resp.id
    print(f"  File uploaded: {file_id}")

    print("  Starting fine-tuning job (gpt-4o-mini) …")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model="gpt-4o-mini-2024-07-18",
        hyperparameters={"n_epochs": 3},
    )
    job_id = job.id
    print(f"  Job started: {job_id}")
    print("  Waiting for fine-tuning to complete (this takes 15–45 minutes) …")

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        print(f"  Status: {status}  [{time.strftime('%H:%M:%S')}]")
        if status == "succeeded":
            ft_model = job.fine_tuned_model
            print(f"  Fine-tuned model ready: {ft_model}")
            return ft_model
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Fine-tuning job {status}: {job}")
        time.sleep(60)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TEST PORTFOLIO PROMPTS  (synthetic profiles, no API calls needed)
# ══════════════════════════════════════════════════════════════════════════════

TEST_PROFILES = [
    {
        "username": "alex_dev",
        "programme": "CSAI", "semester": 5,
        "n_repos": 14, "commits_90d": 187, "commits_per_week": 4.2,
        "active_weeks": 11, "collab_events": 8, "days_since_push": 3,
        "languages": {"Python": 62.0, "JavaScript": 21.0, "Shell": 10.0, "C++": 7.0},
        "domains": ["ML/AI", "Web", "DevOps"],
        "alignment": {"score": 81, "core_coverage": 100, "relevance": 75,
                      "matched_core": ["Python", "C++"], "missing_core": []},
        "top_repos": ["ml-sentiment-api", "react-dashboard", "docker-utils",
                      "cnn-classifier", "fastapi-starter", "data-pipeline"],
        "gaps": ["3/14 repos have no description",
                 "No collaborative activity detected"],
    },
    {
        "username": "sara_ml",
        "programme": "DSAI", "semester": 3,
        "n_repos": 6, "commits_90d": 34, "commits_per_week": 0.8,
        "active_weeks": 4, "collab_events": 0, "days_since_push": 41,
        "languages": {"Python": 88.0, "Jupyter Notebook": 12.0},
        "domains": ["Data", "ML/AI"],
        "alignment": {"score": 64, "core_coverage": 50, "relevance": 80,
                      "matched_core": ["Python"], "missing_core": ["R"]},
        "top_repos": ["pandas-analysis", "sklearn-experiments", "numpy-practice",
                      "housing-prediction", "nlp-basics", "mnist-cnn"],
        "gaps": ["Last push was 41 days ago — profile appears inactive",
                 "Low commit frequency (0.8/week)",
                 "No collaborative activity",
                 "Missing core DSAI languages: R"],
    },
    {
        "username": "omar_swe",
        "programme": "SWE", "semester": 6,
        "n_repos": 22, "commits_90d": 312, "commits_per_week": 7.1,
        "active_weeks": 13, "collab_events": 31, "days_since_push": 1,
        "languages": {"TypeScript": 45.0, "JavaScript": 30.0, "Python": 15.0, "CSS": 10.0},
        "domains": ["Web", "DevOps", "Mobile"],
        "alignment": {"score": 92, "core_coverage": 100, "relevance": 90,
                      "matched_core": ["JavaScript", "TypeScript", "Java"],
                      "missing_core": []},
        "top_repos": ["nextjs-ecommerce", "graphql-api", "ci-template",
                      "react-native-app", "node-microservices", "ts-utils"],
        "gaps": ["14/22 repos missing a license"],
    },
    {
        "username": "hana_beginner",
        "programme": "CSAI", "semester": 2,
        "n_repos": 3, "commits_90d": 12, "commits_per_week": 0.3,
        "active_weeks": 2, "collab_events": 0, "days_since_push": 78,
        "languages": {"Python": 100.0},
        "domains": ["General"],
        "alignment": {"score": 38, "core_coverage": 50, "relevance": 40,
                      "matched_core": ["Python"], "missing_core": ["C++"]},
        "top_repos": ["hello-world", "python-calculator", "guess-number"],
        "gaps": ["Last push was 78 days ago",
                 "Low commit frequency (0.3/week)",
                 "No collaborative activity",
                 "Missing core CSAI languages: C++",
                 "Only 3 public repos"],
    },
]


def _build_test_prompt(p: dict) -> str:
    langs     = p["languages"]
    al        = p["alignment"]
    lang_str  = ", ".join(f"{l} {v:.0f}%" for l, v in list(langs.items())[:5])
    all_langs = ", ".join(langs.keys())
    repos     = "\n".join(f"  • {r}" for r in p["top_repos"])
    gaps      = "\n".join(f"  ⚠ {g}" for g in p["gaps"]) or "  ✓ No major gaps"
    matched   = ", ".join(al["matched_core"]) or "none"
    missing   = ", ".join(al["missing_core"]) or "none"

    return (
        f"You are an expert GitHub portfolio auditor and tech industry HR advisor.\n"
        f"Evaluate this developer portfolio honestly, like a senior hiring manager.\n\n"
        f"═══════════════════════════════════════════════════\n"
        f"PROFILE: @{p['username']}  |  {p['programme']} — Semester {p['semester']} of 8\n"
        f"Public repos: {p['n_repos']}  |  Domain focus: {', '.join(p['domains'])}\n"
        f"═══════════════════════════════════════════════════\n\n"
        f"LANGUAGE DISTRIBUTION:\n"
        f"  All languages: {all_langs}\n"
        f"  Breakdown: {lang_str}\n\n"
        f"ACTIVITY — LAST 90 DAYS:\n"
        f"  Commits        : {p['commits_90d']:,}\n"
        f"  Commits/week   : {p['commits_per_week']:.1f}\n"
        f"  Active weeks   : {p['active_weeks']} / 13\n"
        f"  Collab events  : {p['collab_events']}\n"
        f"  Days since push: {p['days_since_push']}\n\n"
        f"{p['programme']} STACK ALIGNMENT:\n"
        f"  Score: {al['score']}%  "
        f"(core coverage {al['core_coverage']}%  +  relevance {al['relevance']}%)\n"
        f"  Core matched: [{matched}]   Missing: [{missing}]\n\n"
        f"TOP REPOSITORIES:\n{repos}\n\n"
        f"DETECTED GAPS:\n{gaps}\n\n"
        f"═══════════════════════════════════════════════════\n"
        f"TASK: Write a career advisory report for this {p['programme']} student.\n"
        f"Sections: Portfolio Verdict | Strengths | Critical Gaps | "
        f"Projects to Build | 30-Day Plan\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RUN BOTH MODELS
# ══════════════════════════════════════════════════════════════════════════════

def run_seq2seq(prompts: list[str]) -> list[str]:
    print("\n[Model A] Fine-tuned Seq2Seq (from scratch) …")
    import torch
    from models_from_scratch.seq2seq_transformer import Seq2SeqTransformer, Seq2SeqConfig
    from transformers import BartTokenizer

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt      = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    cfg       = ckpt.get("config", Seq2SeqConfig())
    model     = Seq2SeqTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    answers = []
    for i, prompt in enumerate(prompts):
        enc = tokenizer(prompt, max_length=512, truncation=True, return_tensors="pt")
        src_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            out_ids = model.generate(
                src_ids,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new=300,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
            )

        text = tokenizer.decode(out_ids[0, 1:], skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
        answers.append(text.strip() or "(no output)")
        print(f"  Profile {i+1}/{len(prompts)} done.")

    return answers


def run_gpt4o_zeroshot(client, prompts: list[str]) -> list[str]:
    print(f"\n[Model B] GPT-4o zero-shot (production model) …")
    answers = []
    for i, prompt in enumerate(prompts):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": "You are an expert GitHub portfolio auditor and tech industry HR advisor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        text = resp.choices[0].message.content.strip()
        answers.append(text)
        print(f"  Profile {i+1}/{len(prompts)} done.")
    return answers


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — METRICS
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_SECTIONS = [
    "portfolio verdict", "strengths", "critical gaps",
    "projects to build", "30-day",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z0-9%.]+\b', text.lower())


def bleu1(pred: str, ref: str) -> float:
    p, r = _tokenize(pred), _tokenize(ref)
    if not p:
        return 0.0
    pc, rc = Counter(p), Counter(r)
    clipped = sum(min(c, rc[w]) for w, c in pc.items())
    prec    = clipped / len(p)
    bp      = 1.0 if len(p) >= len(r) else math.exp(1 - len(r) / len(p))
    return bp * prec


def token_f1(pred: str, ref: str) -> float:
    p, r = _tokenize(pred), _tokenize(ref)
    if not p or not r:
        return 0.0
    pc, rc = Counter(p), Counter(r)
    common = sum((pc & rc).values())
    if not common:
        return 0.0
    prec = common / sum(pc.values())
    rec  = common / sum(rc.values())
    return 2 * prec * rec / (prec + rec)


def section_coverage(pred: str) -> float:
    text = pred.lower()
    hits = sum(1 for s in REQUIRED_SECTIONS if s in text)
    return hits / len(REQUIRED_SECTIONS)


def keyword_grounding(pred: str, profile: dict) -> float:
    """Did the model mention actual repo names and key numbers from the profile?"""
    text  = pred.lower()
    keys  = (
        [r.lower() for r in profile["top_repos"]]
        + [str(profile["commits_90d"]), str(profile["n_repos"])]
        + [str(profile["alignment"]["score"]) + "%"]
    )
    hits = sum(1 for k in keys if k in text)
    return hits / max(len(keys), 1)


def score_pair(pred: str, ref: str, profile: dict) -> dict:
    return {
        "bleu1":            bleu1(pred, ref),
        "token_f1":         token_f1(pred, ref),
        "section_coverage": section_coverage(pred),
        "keyword_grounding": keyword_grounding(pred, profile),
        "overall":          (bleu1(pred, ref) + token_f1(pred, ref)
                             + section_coverage(pred) + keyword_grounding(pred, profile)) / 4,
    }


METRICS      = ["bleu1", "token_f1", "section_coverage", "keyword_grounding", "overall"]
METRIC_NAMES = ["BLEU-1", "Token F1", "Section\nCoverage", "Keyword\nGrounding", "Overall"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — PRINT REPORT + PLOT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(profiles, seq_scores, gpt_scores):
    SEP = "═" * 76
    print(f"\n{SEP}")
    print("  CAREER ADVISOR COMPARISON  —  Fine-tuned Seq2Seq  vs  GPT-4o (zero-shot)")
    print(f"  Seq2Seq trained on 2,000 GH Archive pairs  |  GPT-4o: zero-shot reference")
    print(SEP)
    print(f"\n  {'Metric':<22}  {'Seq2Seq (scratch)':>20}  {'GPT-4o-mini FT':>20}")
    print(f"  {'─'*22}  {'─'*20}  {'─'*20}")

    s_avgs = {m: sum(s[m] for s in seq_scores) / len(seq_scores) for m in METRICS}
    g_avgs = {m: sum(s[m] for s in gpt_scores) / len(gpt_scores) for m in METRICS}

    for m, label in zip(METRICS, METRIC_NAMES):
        sv = s_avgs[m]
        gv = g_avgs[m]
        label_flat = label.replace("\n", " ")
        winner = "◀ Seq2Seq wins" if sv > gv else ("tie" if abs(sv-gv) < 0.01 else "")
        print(f"  {label_flat:<22}  {sv*100:>8.1f}%             {gv*100:>8.1f}%    {winner}")

    print(f"\n{SEP}")
    for i, (p, ss, gs) in enumerate(zip(profiles, seq_scores, gpt_scores), 1):
        print(f"\n  Profile {i}: @{p['username']}  ({p['programme']}, sem {p['semester']})")
        print(f"  {'─'*72}")
        print(f"  {'Metric':<22}  {'Seq2Seq':>10}  {'GPT-4o-mini':>12}")
        for m, label in zip(METRICS, METRIC_NAMES):
            label_flat = label.replace("\n", " ")
            print(f"  {label_flat:<22}  {ss[m]*100:>8.1f}%  {gs[m]*100:>10.1f}%")
    print(f"\n{SEP}")


def make_plot(seq_scores: list[dict], gpt_scores: list[dict]) -> None:
    s_avg = np.array([sum(s[m] for s in seq_scores) / len(seq_scores) for m in METRICS])
    g_avg = np.array([sum(s[m] for s in gpt_scores) / len(gpt_scores) for m in METRICS])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Career Advisor Comparison: Fine-tuned Seq2Seq (from scratch)  vs  GPT-4o (zero-shot)\n"
        "Seq2Seq trained on 20,000 GH Archive pairs  |  4 test profiles",
        fontsize=13, fontweight="bold",
    )

    BG = "#F8F9FA"
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(True, color="#DEE2E6", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=SEQ2SEQ_COLOR, label="Seq2Seq fine-tuned (from scratch, 139 M)"),
        mpatches.Patch(color=GPTFT_COLOR,   label="GPT-4o zero-shot (~200 B)"),
    ]

    # ── Panel 1: Average per metric ───────────────────────────────────────────
    x, w = np.arange(len(METRICS)), 0.35
    b1 = axes[0].bar(x - w/2, s_avg * 100, w, color=SEQ2SEQ_COLOR, alpha=0.88, zorder=3)
    b2 = axes[0].bar(x + w/2, g_avg * 100, w, color=GPTFT_COLOR,   alpha=0.88, zorder=3)
    for bars, color in [(b1, SEQ2SEQ_COLOR), (b2, GPTFT_COLOR)]:
        for bar in bars:
            v = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 1,
                         f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5,
                         color=color, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(METRIC_NAMES, fontsize=9)
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Average Score per Metric", fontweight="bold")
    axes[0].legend(handles=legend_patches, fontsize=8)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── Panel 2: Overall score per profile ────────────────────────────────────
    n_p = len(seq_scores)
    xp  = np.arange(n_p)
    s_ov = np.array([s["overall"] for s in seq_scores])
    g_ov = np.array([s["overall"] for s in gpt_scores])
    p_labels = [f"Profile {i+1}" for i in range(n_p)]

    b3 = axes[1].bar(xp - w/2, s_ov * 100, w, color=SEQ2SEQ_COLOR, alpha=0.88, zorder=3)
    b4 = axes[1].bar(xp + w/2, g_ov * 100, w, color=GPTFT_COLOR,   alpha=0.88, zorder=3)
    for bars, color in [(b3, SEQ2SEQ_COLOR), (b4, GPTFT_COLOR)]:
        for bar in bars:
            v = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, v + 1,
                         f"{v:.0f}%", ha="center", va="bottom", fontsize=9,
                         color=color, fontweight="bold")
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels(p_labels, fontsize=9)
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Overall Score (%)")
    axes[1].set_title("Overall Score per Profile", fontweight="bold")
    axes[1].legend(handles=legend_patches, fontsize=8)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # ── Panel 3: Radar chart ──────────────────────────────────────────────────
    ax_r   = axes[2]
    ax_r.remove()
    ax_r   = fig.add_subplot(1, 3, 3, polar=True)
    ax_r.set_facecolor(BG)

    n_r    = len(METRICS)
    angles = np.linspace(0, 2 * np.pi, n_r, endpoint=False).tolist()
    angles += angles[:1]
    ax_r.set_theta_offset(np.pi / 2)
    ax_r.set_theta_direction(-1)

    for avg, color, label in [
        (s_avg, SEQ2SEQ_COLOR, "Seq2Seq"),
        (g_avg, GPTFT_COLOR,   "GPT-4o-mini FT"),
    ]:
        vals = (avg * 100).tolist() + [avg[0] * 100]
        ax_r.plot(angles, vals, "o-", lw=2, color=color, label=label, zorder=3)
        ax_r.fill(angles, vals, alpha=0.12, color=color, zorder=2)

    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(METRIC_NAMES, fontsize=9)
    ax_r.set_ylim(0, 100)
    ax_r.set_yticks([20, 40, 60, 80, 100])
    ax_r.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=7, color="#666")
    ax_r.set_title("Radar: All Metrics", fontweight="bold", pad=18)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=8.5)

    fig.text(0.5, 0.01,
             "Seq2Seq trained on 20,000 GH Archive portfolio→advisory pairs  |  "
             "GPT-4o is zero-shot (no fine-tuning)  |  "
             "Reference = GPT-4o output (used as gold standard)",
             ha="center", fontsize=8, color="#666")

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {PLOT_PATH}")
    print(f"Open with:  open {PLOT_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.api_key)

    print("═" * 76)
    print("  Career Advisor Comparison")
    print("  Fine-tuned Seq2Seq (from scratch)  vs  GPT-4o (zero-shot)")
    print("  Seq2Seq trained on 2,000 GH Archive pairs")
    print("═" * 76)

    # ── Step 1: Build test prompts ──────────────────────────────────────────
    print("\n[Step 1] Building test prompts …")
    prompts = [_build_test_prompt(p) for p in TEST_PROFILES]
    print(f"  {len(prompts)} test profiles ready.")

    # ── Steps 2–3: Run both models ──────────────────────────────────────────
    seq_answers = run_seq2seq(prompts)
    gpt_answers = run_gpt4o_zeroshot(client, prompts)

    # ── Step 4: Score  (GPT-4o output is the gold-standard reference) ───────
    print("\n[Step 4] Scoring …")
    seq_scores = [score_pair(s, g, p)
                  for s, g, p in zip(seq_answers, gpt_answers, TEST_PROFILES)]
    gpt_scores = [score_pair(g, g, p)
                  for g, p in zip(gpt_answers, TEST_PROFILES)]

    # ── Step 5: Report + plot ───────────────────────────────────────────────
    print_report(TEST_PROFILES, seq_scores, gpt_scores)
    make_plot(seq_scores, gpt_scores)

    print("\n" + "═" * 76)
    print("  SAMPLE OUTPUTS — Profile 1 (@alex_dev, CSAI sem 5)")
    print("═" * 76)
    print(f"\n── Seq2Seq (fine-tuned) output ──\n{seq_answers[0][:600]}")
    print(f"\n── GPT-4o (zero-shot) output ──\n{gpt_answers[0][:600]}")


if __name__ == "__main__":
    main()
