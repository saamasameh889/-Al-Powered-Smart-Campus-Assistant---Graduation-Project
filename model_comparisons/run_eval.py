import os as _os
_os.chdir(r"C:/Users/ahmed/OneDrive/Desktop/Graduation-/model_comparisons")
import matplotlib
matplotlib.use("Agg")
# Install dependencies (run once)

import os, json, time, re, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from openai import OpenAI
from rouge_score import rouge_scorer

# ── Set your OpenAI key ────────────────────────────────────────────────────────
# Option A: paste key directly
# os.environ["OPENAI_API_KEY"] = "sk-..."

# Option B: load from .env in project root
try:
    from dotenv import load_dotenv
    load_dotenv(Path("../") / ".env")
except Exception:
    pass

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
print("OpenAI client ready.")

# ── Load CampusRAG (add project root to path) ─────────────────────────────────
project_root = Path("../").resolve()
sys.path.insert(0, str(project_root))

from phase5_rag_pipeline import CampusRAG

rag = CampusRAG()
print(f"RAG pipeline ready — model: {rag._chat_model}")

TEST_SET = [
    # ── Scholarships & Financial ──────────────────────────────────────────────
    {
        "id": 1, "category": "scholarships",
        "question": "What is the minimum cGPA a student must maintain to keep a full 100% scholarship at Zewail City?",
        "gold_answer": "Students must maintain a minimum cumulative GPA of 3.00 to keep their full 100% scholarship."
    },
    {
        "id": 2, "category": "scholarships",
        "question": "What tuition discount does a Thanaweya Amma student with a score of 85% or more receive at Zewail City?",
        "gold_answer": "A Thanaweya Amma student with a score of 85% or more receives a 50% discount on tuition fees."
    },
    {
        "id": 3, "category": "scholarships",
        "question": "What percentage discount do siblings of current UST students receive on tuition?",
        "gold_answer": "Siblings of UST students receive a 10% discount on tuition fees."
    },
    {
        "id": 4, "category": "scholarships",
        "question": "What is the scholarship discount for a returning student who achieves a cGPA between 3.80 and 4.00?",
        "gold_answer": "Returning students with a cGPA from 3.80 to less than 4.00 receive a 50% discount."
    },
    {
        "id": 5, "category": "scholarships",
        "question": "What discount is offered to students applying to the School of Business in their first year?",
        "gold_answer": "Students applying to the School of Business receive a 25% discount in their first year."
    },
    {
        "id": 6, "category": "scholarships",
        "question": "What tuition discount do IGCSE students with scores of 93% or above receive at Zewail City?",
        "gold_answer": "IGCSE students with scores of 93% or above receive a 50% discount on tuition fees."
    },
    {
        "id": 7, "category": "scholarships",
        "question": "What discount do children of Egyptian university faculty members receive at Zewail City?",
        "gold_answer": "Children of faculty members working in Egyptian universities or Egyptian research centers receive a 20% discount in their first year."
    },

    # ── Credit Hours & Graduation Requirements ────────────────────────────────
    {
        "id": 8, "category": "graduation",
        "question": "How many total credit hours are required to graduate from the School of CSAI at Zewail City?",
        "gold_answer": "132 credit hours are required to graduate from the School of CSAI."
    },
    {
        "id": 9, "category": "graduation",
        "question": "How many credit hours are required to graduate from the School of Business at Zewail City?",
        "gold_answer": "114 credit hours are required to graduate from the School of Business."
    },
    {
        "id": 10, "category": "graduation",
        "question": "How many credit hours are required to graduate from the School of Engineering at Zewail City?",
        "gold_answer": "140 credit hours are required to graduate from the School of Engineering."
    },

    # ── Schools & Programs ────────────────────────────────────────────────────
    {
        "id": 11, "category": "programs",
        "question": "How many undergraduate schools does Zewail City have?",
        "gold_answer": "Zewail City has four undergraduate schools: Engineering, CSAI, Science, and Business."
    },
    {
        "id": 12, "category": "programs",
        "question": "What programs are offered by the School of Science at Zewail City?",
        "gold_answer": "The School of Science offers Biomedical Sciences, Nanoscience, and Physics of the Universe programs."
    },
    {
        "id": 13, "category": "programs",
        "question": "What engineering programs are offered in the School of Engineering at Zewail City?",
        "gold_answer": "The School of Engineering offers Aerospace Engineering, Nanotechnology and Nanoelectronics, Environmental Engineering, Communications and Information Engineering, and Renewable Energy Engineering."
    },
    {
        "id": 14, "category": "programs",
        "question": "What programs are offered in the School of CSAI at Zewail City?",
        "gold_answer": "The School of CSAI offers Computer Science, DSAI, HCI, and Computer Engineering programs."
    },
    {
        "id": 15, "category": "programs",
        "question": "Can a UST student join a Minor offered by a different school than their Major?",
        "gold_answer": "Yes, a student can join a Minor offered by a different school, as long as the Minor is not in the same field as their Major."
    },

    # ── Faculty ───────────────────────────────────────────────────────────────
    {
        "id": 16, "category": "faculty",
        "question": "Who is the director of the Academic Advising Unit at Zewail City?",
        "gold_answer": "Dr. Ahmed Sayed Abdelsamea is the Director of Academic Advising Unit and an Associate Professor in the Math Department."
    },
    {
        "id": 17, "category": "faculty",
        "question": "Who is the director of the Biomedical Sciences Program at Zewail City?",
        "gold_answer": "Dr. Nagwa El-Badri is the Director of the Biomedical Sciences Program and Director of the Center of Excellence for Stem Cells and Regenerative Medicine."
    },
    {
        "id": 18, "category": "faculty",
        "question": "Who is the director of the Renewable Energy Engineering program at Zewail City?",
        "gold_answer": "Dr. Amgad A. El-Deib is the Director of Renewable Energy Engineering and Director of the Center of Renewable Energy and Energy Efficiencies."
    },
    {
        "id": 19, "category": "faculty",
        "question": "Who is the director of the Nanoscience program at Zewail City?",
        "gold_answer": "Dr. Ibrahim El Sherbiny is the Director of the Nanoscience Program and Co-Director of Center for Materials Science."
    },
    {
        "id": 20, "category": "faculty",
        "question": "Who is the director of the Communications and Information Engineering Program at Zewail City?",
        "gold_answer": "Dr. Samy Soliman is the Director of Communications and Information Engineering Program."
    },
    {
        "id": 21, "category": "faculty",
        "question": "Who is the Director General of Research at Zewail City?",
        "gold_answer": "Dr. Salah Obayya is the Director General of Research Institutes and Founding Director of the Center for Photonics and Smart Materials."
    },
    {
        "id": 22, "category": "faculty",
        "question": "Who directs the Software Development Program at Zewail City?",
        "gold_answer": "Dr. Doaa Shawky is the Director of the Software Development Program."
    },
    {
        "id": 23, "category": "faculty",
        "question": "Who directs the Data Science Program at Zewail City?",
        "gold_answer": "Dr. Khaled Mostafa El Sayed is the Director of the Data Science Program."
    },

    # ── Library & Facilities ──────────────────────────────────────────────────
    {
        "id": 24, "category": "facilities",
        "question": "What are the opening hours of the Zewail City library?",
        "gold_answer": "The Zewail City library is open from 08:30 AM to 03:45 PM for Zewail City members."
    },
    {
        "id": 25, "category": "facilities",
        "question": "Approximately how many items does the Zewail City library collection contain?",
        "gold_answer": "The library collection contains approximately 3,500 items."
    },
    {
        "id": 26, "category": "facilities",
        "question": "What plagiarism checking tools are available to Zewail City students?",
        "gold_answer": "Zewail City students have free access to Turnitin and iThenticate for plagiarism checking."
    },
    {
        "id": 27, "category": "facilities",
        "question": "How many users can Hall 2 of the Zewail City library accommodate?",
        "gold_answer": "Hall 2 of the Zewail City library can accommodate 105 users."
    },

    # ── Research Centers ──────────────────────────────────────────────────────
    {
        "id": 28, "category": "research",
        "question": "What research institute at Zewail City focuses on medical sciences?",
        "gold_answer": "The Helmy Institute for Medical Sciences (HIMS) focuses on medical sciences, encompassing centers for aging, genomics, microbiology, stem cells, and food research."
    },
    {
        "id": 29, "category": "research",
        "question": "What is the name of Zewail City's robotics team and what award did they win in 2025?",
        "gold_answer": "Aquila ZC is Zewail City's robotics team. They won the VEX U World Championship Judges Award 2025 in Dallas, Texas."
    },
    {
        "id": 30, "category": "research",
        "question": "Who is the director of the Center for Fundamental Physics at Zewail City?",
        "gold_answer": "Dr. Shaaban Khalil is the Director of the Center for Fundamental Physics."
    },

    # ── Academic Calendar & Registration ──────────────────────────────────────
    {
        "id": 31, "category": "academic",
        "question": "When does online registration start according to the Zewail City academic calendar?",
        "gold_answer": "Online registration starts on Sunday, June 28, 2026."
    },
    {
        "id": 32, "category": "academic",
        "question": "What email should students contact for scholarship or financial support inquiries at Zewail City?",
        "gold_answer": "Students should contact finsupport@zewailcity.edu.eg for scholarship or financial support inquiries."
    },

    # ── Admissions ────────────────────────────────────────────────────────────
    {
        "id": 33, "category": "admissions",
        "question": "When was the first cohort of students accepted at Zewail City's University of Science and Technology?",
        "gold_answer": "The first cohort was accepted in Fall 2013, and classes have been graduating since Summer 2017."
    },
    {
        "id": 34, "category": "admissions",
        "question": "What discount do Palestinian, Syrian, Sudanese, or Yemeni resident students receive at Zewail City?",
        "gold_answer": "Resident students from Syria, Palestine, Sudan, and Yemen receive a 20% discount on tuition in their first year."
    },
    {
        "id": 35, "category": "admissions",
        "question": "What scholarship percentage does Zewail City award to students who maintain a cGPA of exactly 4.00?",
        "gold_answer": "Students who maintain a cGPA of 4.00 receive a 100% scholarship discount."
    },

    # ── General & Policies ────────────────────────────────────────────────────
    {
        "id": 36, "category": "policy",
        "question": "Can a student receive both a sibling discount and a merit scholarship at Zewail City at the same time?",
        "gold_answer": "No. Only one discount can be applied per individual. Discounts cannot be combined."
    },
    {
        "id": 37, "category": "programs",
        "question": "What Business programs are offered by the School of Business at Zewail City?",
        "gold_answer": "The School of Business offers Finance, Business Analytics, Actuarial Analysis and Risk Management, Operations Management, and Entrepreneurship and Innovation Management programs."
    },
    {
        "id": 38, "category": "faculty",
        "question": "Who is the Director of the Physics of the Universe program at Zewail City?",
        "gold_answer": "Dr. Tarek Ibrahim is the Director of the Physics of Universe Program."
    },
    {
        "id": 39, "category": "faculty",
        "question": "Who is the Acting Dean of Academic Affairs at Zewail City?",
        "gold_answer": "Dr. Tamer Samir Ahmed is the Acting Dean of Academic Affairs."
    },
    {
        "id": 40, "category": "facilities",
        "question": "How long can a student hold a checked-out library book at Zewail City before the hold is cancelled?",
        "gold_answer": "A student has 2 days to visit the library to check out a book on hold before the hold is cancelled."
    },
]

print(f"Test set: {len(TEST_SET)} questions across {len(set(q['category'] for q in TEST_SET))} categories.")

VANILLA_SYSTEM = (
    "You are a helpful academic advisor. Answer the student's question as accurately as possible. "
    "If you do not know the answer, say so clearly."
)

def ask_vanilla(question: str) -> str:
    """GPT-4o-mini with no context — just the question."""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": VANILLA_SYSTEM},
            {"role": "user",   "content": question},
        ],
        temperature=0,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def ask_rag(question: str) -> str:
    """GPT-4o-mini with RAG — question + retrieved Zewail chunks."""
    answer, _ = rag.answer(question, top_k=6)
    return answer


print("Running evaluation — this will take a few minutes...")
results = []

for i, item in enumerate(TEST_SET):
    print(f"  [{i+1:02d}/{len(TEST_SET)}] {item['question'][:70]}...")
    vanilla = ask_vanilla(item["question"])
    time.sleep(0.5)
    rag_ans  = ask_rag(item["question"])
    time.sleep(0.5)
    results.append({
        "id":           item["id"],
        "category":     item["category"],
        "question":     item["question"],
        "gold_answer":  item["gold_answer"],
        "vanilla_ans":  vanilla,
        "rag_ans":      rag_ans,
    })

print("\nDone. All answers collected.")

# ── ROUGE-L ──────────────────────────────────────────────────────────────────
scorer_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l(hypothesis: str, reference: str) -> float:
    return scorer_rouge.score(reference, hypothesis)["rougeL"].fmeasure


# ── Refusal detection ─────────────────────────────────────────────────────────
REFUSAL_PATTERNS = re.compile(
    r"(i don'?t know|i do not know|i'?m not sure|i cannot|i can'?t|no information|not aware|"
    r"don'?t have (specific|that|this|the|enough)|not in my (knowledge|training)|unable to|"
    r"my knowledge (base|cutoff)|not familiar with|cannot provide|not certain)",
    re.IGNORECASE
)

def is_refusal(text: str) -> int:
    return int(bool(REFUSAL_PATTERNS.search(text)))


# ── GPT-4o judge (factual accuracy + hallucination) ───────────────────────────
JUDGE_PROMPT = """\
You are a strict factual evaluator for Zewail City of Science and Technology academic questions.

Gold Answer: {gold}

Model Answer: {answer}

Evaluate the Model Answer on TWO criteria:

1. FACTUAL_ACCURACY (0 or 1): Is the core factual claim in the Model Answer consistent with the Gold Answer?
   - 1 = correct or mostly correct (minor wording differences OK)
   - 0 = incorrect, contradicts gold, or does not address the question

2. HALLUCINATION (0 or 1): Does the Model Answer contain specific institutional facts (numbers, names, 
   policies, dates, percentages) that are NOT in the Gold Answer and are likely invented?
   - 1 = yes, contains hallucinated Zewail-specific facts
   - 0 = no hallucinations detected

Reply with ONLY this JSON (no explanation):
{{"factual_accuracy": <0 or 1>, "hallucination": <0 or 1>}}
"""

def judge(gold: str, answer: str) -> dict:
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(gold=gold, answer=answer)}],
            temperature=0,
            max_tokens=60,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{.*?\}', raw, re.DOTALL)
        return json.loads(m.group()) if m else {"factual_accuracy": 0, "hallucination": 0}
    except Exception:
        return {"factual_accuracy": 0, "hallucination": 0}


print("Scoring all answers with GPT-4o judge...")
for i, r in enumerate(results):
    print(f"  [{i+1:02d}/{len(results)}] judging...")
    
    # ROUGE-L
    r["vanilla_rouge"]  = rouge_l(r["vanilla_ans"], r["gold_answer"])
    r["rag_rouge"]      = rouge_l(r["rag_ans"],     r["gold_answer"])
    
    # Refusal
    r["vanilla_refusal"] = is_refusal(r["vanilla_ans"])
    r["rag_refusal"]     = is_refusal(r["rag_ans"])
    
    # GPT-4o judge
    v_judge = judge(r["gold_answer"], r["vanilla_ans"])
    time.sleep(0.3)
    g_judge = judge(r["gold_answer"], r["rag_ans"])
    time.sleep(0.3)
    
    r["vanilla_accuracy"]     = v_judge["factual_accuracy"]
    r["vanilla_hallucination"]= v_judge["hallucination"]
    r["rag_accuracy"]         = g_judge["factual_accuracy"]
    r["rag_hallucination"]    = g_judge["hallucination"]

print("\nScoring complete.")

df = pd.DataFrame(results)
n  = len(df)

summary = {
    "Model": ["GPT-4o-mini (Vanilla)", "GPT-4o-mini + RAG"],
    "Factual Accuracy": [
        f"{df['vanilla_accuracy'].sum()}/{n}  ({df['vanilla_accuracy'].mean()*100:.1f}%)",
        f"{df['rag_accuracy'].sum()}/{n}  ({df['rag_accuracy'].mean()*100:.1f}%)",
    ],
    "Hallucination Rate": [
        f"{df['vanilla_hallucination'].sum()}/{n}  ({df['vanilla_hallucination'].mean()*100:.1f}%)",
        f"{df['rag_hallucination'].sum()}/{n}  ({df['rag_hallucination'].mean()*100:.1f}%)",
    ],
    "Refusal Rate": [
        f"{df['vanilla_refusal'].sum()}/{n}  ({df['vanilla_refusal'].mean()*100:.1f}%)",
        f"{df['rag_refusal'].sum()}/{n}  ({df['rag_refusal'].mean()*100:.1f}%)",
    ],
    "ROUGE-L (avg)": [
        f"{df['vanilla_rouge'].mean():.4f}",
        f"{df['rag_rouge'].mean():.4f}",
    ],
}

summary_df = pd.DataFrame(summary)
print("=" * 75)
print("EVALUATION SUMMARY — GPT-4o-mini Vanilla vs. RAG")
print("=" * 75)
print(summary_df.to_string(index=False))
print("=" * 75)

# ── Per-category breakdown ────────────────────────────────────────────────────
cat_df = df.groupby("category").agg(
    n              = ("id",                    "count"),
    vanilla_acc    = ("vanilla_accuracy",       "mean"),
    rag_acc        = ("rag_accuracy",           "mean"),
    vanilla_hall   = ("vanilla_hallucination",  "mean"),
    rag_hall       = ("rag_hallucination",      "mean"),
    vanilla_rouge  = ("vanilla_rouge",          "mean"),
    rag_rouge      = ("rag_rouge",              "mean"),
).round(3).reset_index()

print("\nPer-category accuracy:")
print(cat_df[["category", "n", "vanilla_acc", "rag_acc"]].to_string(index=False))

# ── Figures ───────────────────────────────────────────────────────────────────
VANILLA_COLOR = "#6366F1"
RAG_COLOR     = "#10B981"

metrics_vanilla = [
    df["vanilla_accuracy"].mean(),
    1 - df["vanilla_hallucination"].mean(),   # inverted: lower hall = better
    df["vanilla_rouge"].mean(),
]
metrics_rag = [
    df["rag_accuracy"].mean(),
    1 - df["rag_hallucination"].mean(),
    df["rag_rouge"].mean(),
]
metric_labels = ["Factual Accuracy", "No-Hallucination Rate", "ROUGE-L"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0F0A1E")

# ── Panel 1: grouped bar — overall metrics ────────────────────────────────────
ax = axes[0]
ax.set_facecolor("#1A1035")
x = np.arange(len(metric_labels))
w = 0.35
bars1 = ax.bar(x - w/2, metrics_vanilla, w, label="Vanilla", color=VANILLA_COLOR, alpha=0.85)
bars2 = ax.bar(x + w/2, metrics_rag,     w, label="RAG",     color=RAG_COLOR,     alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, color="#C4B5FD", fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", color="#C4B5FD")
ax.set_title("Overall Metrics Comparison", color="#EDE9FE", fontweight="bold")
ax.tick_params(colors="#C4B5FD")
ax.spines[:].set_color("#3D3060")
for bar in bars1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{bar.get_height():.2f}", ha="center", color="#C4B5FD", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            f"{bar.get_height():.2f}", ha="center", color="#10B981", fontsize=8)
ax.legend(facecolor="#1A1035", labelcolor="#EDE9FE")

# ── Panel 2: per-category accuracy ───────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#1A1035")
cats = cat_df["category"].tolist()
xi   = np.arange(len(cats))
ax2.bar(xi - w/2, cat_df["vanilla_acc"], w, label="Vanilla", color=VANILLA_COLOR, alpha=0.85)
ax2.bar(xi + w/2, cat_df["rag_acc"],     w, label="RAG",     color=RAG_COLOR,     alpha=0.85)
ax2.set_xticks(xi)
ax2.set_xticklabels(cats, rotation=35, ha="right", color="#C4B5FD", fontsize=8)
ax2.set_ylim(0, 1.3)
ax2.set_ylabel("Accuracy", color="#C4B5FD")
ax2.set_title("Accuracy by Category", color="#EDE9FE", fontweight="bold")
ax2.tick_params(colors="#C4B5FD")
ax2.spines[:].set_color("#3D3060")
ax2.legend(facecolor="#1A1035", labelcolor="#EDE9FE")

# ── Panel 3: per-question ROUGE-L scatter ─────────────────────────────────────
ax3 = axes[2]
ax3.set_facecolor("#1A1035")
ax3.scatter(df["vanilla_rouge"], df["rag_rouge"], alpha=0.7, color=RAG_COLOR, edgecolors="#8B5CF6", s=50)
lim_max = max(df["vanilla_rouge"].max(), df["rag_rouge"].max()) + 0.05
ax3.plot([0, lim_max], [0, lim_max], "--", color="#6B7280", linewidth=1, label="y = x (tied)")
ax3.set_xlabel("Vanilla ROUGE-L", color="#C4B5FD")
ax3.set_ylabel("RAG ROUGE-L", color="#C4B5FD")
ax3.set_title("Per-question ROUGE-L\n(above diagonal = RAG wins)", color="#EDE9FE", fontweight="bold")
ax3.tick_params(colors="#C4B5FD")
ax3.spines[:].set_color("#3D3060")
ax3.legend(facecolor="#1A1035", labelcolor="#EDE9FE", fontsize=8)

plt.tight_layout(pad=2)
plt.savefig("rag_vs_vanilla_comparison.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("Chart saved to rag_vs_vanilla_comparison.png")

# ── Save full results to CSV ──────────────────────────────────────────────────
df.to_csv("rag_vs_vanilla_results.csv", index=False)
print("Full results saved to rag_vs_vanilla_results.csv")

# ── Print per-question detail ─────────────────────────────────────────────────
print("\nPer-question accuracy:")
detail = df[["id", "category", "vanilla_accuracy", "rag_accuracy", 
             "vanilla_hallucination", "rag_hallucination",
             "vanilla_rouge", "rag_rouge"]].copy()
detail.columns = ["ID", "Category", "V_Acc", "R_Acc", "V_Hall", "R_Hall", "V_ROUGE", "R_ROUGE"]
detail["V_ROUGE"] = detail["V_ROUGE"].round(3)
detail["R_ROUGE"] = detail["R_ROUGE"].round(3)
print(detail.to_string(index=False))

# ── Sample qualitative comparison — show 3 cases where vanilla fails ──────────
failures = df[(df["vanilla_accuracy"] == 0) & (df["rag_accuracy"] == 1)].head(3)

print("=" * 75)
print("QUALITATIVE EXAMPLES — RAG correct, Vanilla wrong")
print("=" * 75)
for _, row in failures.iterrows():
    print(f"\nQ{row['id']} [{row['category']}]: {row['question']}")
    print(f"  Gold:    {row['gold_answer']}")
    print(f"  Vanilla: {row['vanilla_ans'][:200]}")
    print(f"  RAG:     {row['rag_ans'][:200]}")
    print("-" * 50)
