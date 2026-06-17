"""
compare_career_models.py
Compares GPT-4o vs GPT-4o-mini vs Claude claude-sonnet-4-6 for the GitHub Career Advisor task.
Run with: python compare_career_models.py
Requires OPENAI_API_KEY and ANTHROPIC_API_KEY in environment.
"""
import os, sys, time, json, statistics
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from openai    import OpenAI
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("anthropic package not found — skipping Claude comparison")

oai     = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",""))
anth    = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY","")) if HAS_ANTHROPIC else None

# ── 3 representative synthetic student profiles ───────────────────────────────
PROFILES = [
    {
        "label":     "CSAI Year-2 (sparse portfolio)",
        "programme": "CSAI",
        "semester":  3,
        "prompt": """You are a senior tech career advisor reviewing a GitHub portfolio for CSAI internship readiness.

Student context:
  Programme : CSAI  |  Semester 3 of 8  |  @student_csai2 (joined 2023)
  Bio       : (empty)
  Public repos: 8  |  Total stars: 2  |  Followers: 4

Portfolio metrics:
  Languages        : Python 61%, C++ 28%, C 11%
  Activity (90d)   : 0.8 commits/week, 10 total, last push 45d ago
  Collaborative    : 0 collab events (forks of others, PRs, issue contributions)
  Domain coverage  : General, ML/AI
  Repo quality avg : 1.2/4 (has description / topics / license / non-empty)
  Programme stack  : 33% aligned — matched [Python, C++] — expected [Python, C++, C, Jupyter Notebook, PyTorch, TensorFlow]
  Presentation     : 0/3 (profile README, bio, custom avatar)

Top 4 repositories (by stars + recency):
  • data-structures ⭐0 [quality 1/4] [C++:100%] (no topics)
    Implementation of data structures for CS course
  • hello-world ⭐0 [quality 0/4] [Python:100%] (no topics)
    (no description)
  • sorting-algorithms ⭐1 [quality 2/4] [C:80%, Python:20%] #algorithms
    Various sorting algorithms implemented in C
  • ml-homework ⭐1 [quality 1/4] [Jupyter Notebook:100%] (no topics)
    (no description)

Gaps detected:
  ⚠ 6/8 repos have no description (e.g. hello-world, ml-homework, lab1)
  ⚠ 7/8 repos have no topics/tags
  ⚠ Low commit frequency (0.8/week) — build a daily habit
  ⚠ No collaborative activity (no forks, PRs, or issue contributions)
  ⚠ Only 33% programme stack alignment — missing: Jupyter Notebook, PyTorch, TensorFlow
  ⚠ No profile README (/{username}/{username} repo missing)
  ⚠ Empty GitHub bio — 1-2 sentence bio increases profile views by ~30%

────────────────────────────────────────────────────────────
Provide specific, actionable career advice for this CSAI student applying for internships.
Respond in exactly this structure:

## Portfolio Snapshot
2-3 sentences on overall impression and readiness level.

## Fix First (Top 3 Immediate Actions)
The 3 highest-impact things to fix THIS WEEK. Reference specific repo names and exact numbers from above.

## Projects to Add
2-3 project ideas perfectly matched to CSAI expected stack and internship expectations. Be specific about tech stack, scope, and why this project stands out.

## Stand Out
1-2 ways to differentiate from other CSAI applicants at Zewail City applying to the same roles.

## 4-Week Action Plan
Week 1–4 concrete milestones.

Rules: Be specific — mention actual repo names from the list above. Reference exact percentages. Tailor all advice strictly to CSAI career paths."""
    },
    {
        "label":     "SWE Year-3 (decent portfolio, wrong stack)",
        "programme": "SWE",
        "semester":  6,
        "prompt": """You are a senior tech career advisor reviewing a GitHub portfolio for SWE internship readiness.

Student context:
  Programme : SWE  |  Semester 6 of 8  |  @student_swe3 (joined 2022)
  Bio       : Student at Zewail City | Python lover
  Public repos: 22  |  Total stars: 18  |  Followers: 31

Portfolio metrics:
  Languages        : Python 72%, Shell 14%, JavaScript 9%, HTML 5%
  Activity (90d)   : 2.4 commits/week, 31 total, last push 8d ago
  Collaborative    : 3 collab events (forks of others, PRs, issue contributions)
  Domain coverage  : ML/AI, Web, Data
  Repo quality avg : 2.1/4 (has description / topics / license / non-empty)
  Programme stack  : 28% aligned — matched [JavaScript, HTML] — expected [JavaScript, TypeScript, Java, HTML, CSS, Go]
  Presentation     : 1/3 (profile README, bio, custom avatar)

Top 6 repositories (by stars + recency):
  • flask-todo-api ⭐4 [quality 3/4] [Python:90%, HTML:10%] #flask #rest-api #python
    A simple REST API built with Flask for task management
  • sentiment-analysis ⭐5 [quality 3/4] [Python:100%] #nlp #machine-learning #bert
    Sentiment analysis using BERT fine-tuning on movie reviews
  • data-viz-dashboard ⭐3 [quality 2/4] [Python:85%, HTML:15%] #plotly
    Interactive dashboard built with Plotly
  • portfolio-site ⭐2 [quality 2/4] [HTML:60%, CSS:30%, JavaScript:10%] #portfolio
    My personal portfolio website
  • leetcode-solutions ⭐3 [quality 1/4] [Python:100%] (no topics)
    (no description)
  • movie-recommender [fork] ⭐1 [quality 1/4] [Python:100%] (no topics)
    (no description)

Gaps detected:
  ⚠ Only 28% programme stack alignment — missing: TypeScript, Java, CSS, Go
  ⚠ 72% of code is Python — SWE roles expect JavaScript/TypeScript dominance
  ⚠ No collaborative activity beyond 3 events
  ⚠ No profile README

────────────────────────────────────────────────────────────
Provide specific, actionable career advice for this SWE student applying for internships.
Respond in exactly this structure:

## Portfolio Snapshot
2-3 sentences on overall impression and readiness level.

## Fix First (Top 3 Immediate Actions)
The 3 highest-impact things to fix THIS WEEK. Reference specific repo names and exact numbers from above.

## Projects to Add
2-3 project ideas perfectly matched to SWE expected stack and internship expectations. Be specific about tech stack, scope, and why this project stands out.

## Stand Out
1-2 ways to differentiate from other SWE applicants at Zewail City applying to the same roles.

## 4-Week Action Plan
Week 1–4 concrete milestones.

Rules: Be specific — mention actual repo names from the list above. Reference exact percentages. Tailor all advice strictly to SWE career paths."""
    },
    {
        "label":     "DSAI Year-4 (strong portfolio, final polish)",
        "programme": "DSAI",
        "semester":  8,
        "prompt": """You are a senior tech career advisor reviewing a GitHub portfolio for DSAI internship readiness.

Student context:
  Programme : DSAI  |  Semester 8 of 8  |  @student_dsai4 (joined 2021)
  Bio       : Data Scientist | NLP | Kaggle Grandmaster (bronze) | @ZewailCity
  Public repos: 34  |  Total stars: 87  |  Followers: 142

Portfolio metrics:
  Languages        : Python 68%, Jupyter Notebook 19%, R 8%, SQL 5%
  Activity (90d)   : 5.1 commits/week, 66 total, last push 2d ago
  Collaborative    : 12 collab events (forks of others, PRs, issue contributions)
  Domain coverage  : ML/AI, Data, Research
  Repo quality avg : 2.9/4 (has description / topics / license / non-empty)
  Programme stack  : 80% aligned — matched [Python, R, Jupyter Notebook, SQL] — expected [Python, R, Jupyter Notebook, SQL, Scala]
  Presentation     : 2/3 (profile README, bio, custom avatar)

Top 6 repositories (by stars + recency):
  • arabic-sentiment-bert ⭐31 [quality 4/4] [Python:100%] #nlp #arabic #bert #transformers
    Fine-tuned AraBERT on 50k reviews; achieves 91% accuracy — published at ACL workshop 2024
  • kaggle-house-prices ⭐12 [quality 3/4] [Jupyter Notebook:100%] #kaggle #xgboost #feature-engineering
    Top 3% solution with detailed EDA and feature engineering notebook
  • time-series-anomaly ⭐18 [quality 4/4] [Python:80%, Jupyter Notebook:20%] #lstm #anomaly-detection
    LSTM autoencoder for industrial sensor anomaly detection
  • data-pipeline-etl ⭐8 [quality 3/4] [Python:70%, SQL:30%] #etl #airflow #postgres
    Airflow DAG pipeline ingesting 1M+ rows from multiple APIs
  • ml-from-scratch ⭐14 [quality 4/4] [Python:100%] #machine-learning #numpy #algorithms
    Linear regression, SVM, decision tree implemented from scratch with math derivations
  • zewail-research-assistant ⭐4 [quality 3/4] [Python:90%, HTML:10%] #rag #llm #langchain
    RAG-based chatbot for academic paper Q&A using LangChain + ChromaDB

Gaps detected:
  ⚠ No Scala in any repo — important for big-data roles (Spark)
  ⚠ Only 80% programme stack alignment — missing: Scala

────────────────────────────────────────────────────────────
Provide specific, actionable career advice for this DSAI student applying for internships.
Respond in exactly this structure:

## Portfolio Snapshot
2-3 sentences on overall impression and readiness level.

## Fix First (Top 3 Immediate Actions)
The 3 highest-impact things to fix THIS WEEK. Reference specific repo names and exact numbers from above.

## Projects to Add
2-3 project ideas perfectly matched to DSAI expected stack and internship expectations. Be specific about tech stack, scope, and why this project stands out.

## Stand Out
1-2 ways to differentiate from other DSAI applicants at Zewail City applying to the same roles.

## 4-Week Action Plan
Week 1–4 concrete milestones.

Rules: Be specific — mention actual repo names from the list above. Reference exact percentages. Tailor all advice strictly to DSAI career paths."""
    },
]

MODELS = {
    "gpt-4o":       ("openai", "gpt-4o"),
    "gpt-4o-mini":  ("openai", "gpt-4o-mini"),
}
if HAS_ANTHROPIC:
    MODELS["claude-sonnet-4-6"] = ("anthropic", "claude-sonnet-4-6")


def call_model(provider, model_id, prompt):
    t0 = time.time()
    if provider == "openai":
        resp = oai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=900,
        )
        text = resp.choices[0].message.content.strip()
    else:
        resp = anth.messages.create(
            model=model_id,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        text = resp.content[0].text.strip()
    return text, round(time.time() - t0, 2)


JUDGE_PROMPT = """You are evaluating GitHub career advice for a {programme} student.

Original prompt (student data):
{prompt_excerpt}

Generated advice:
{advice}

Rate this advice on four criteria (each 1-5):
- specificity: Does it reference specific repo names, exact percentages, and concrete data from the student's profile? (5=highly specific, 1=completely generic)
- actionability: Are the suggestions concrete and immediately doable this week? (5=crystal clear next steps, 1=vague platitudes)
- programme_fit: Is all advice tailored to {programme} career paths? (5=perfectly tailored, 1=generic advice that ignores the programme)
- structure: Does it follow the required 5-section structure (Snapshot/Fix First/Projects/Stand Out/Action Plan)? (5=perfect, 1=missing sections)

Reply ONLY as: specificity=X actionability=X programme_fit=X structure=X"""


def judge_advice(programme, prompt, advice):
    resp = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            programme=programme,
            prompt_excerpt=prompt[:600],
            advice=advice[:1200],
        )}],
        temperature=0, max_tokens=60,
    )
    text = resp.choices[0].message.content
    scores = {}
    for part in text.split():
        if "=" in part:
            k, v = part.split("=", 1)
            try: scores[k.strip()] = int(v.strip())
            except: pass
    return scores


# ── Run experiment ────────────────────────────────────────────────────────────
results = {m: [] for m in MODELS}

for p_idx, profile in enumerate(PROFILES):
    print(f"\n{'='*60}")
    print(f"Profile {p_idx+1}: {profile['label']}")
    print(f"{'='*60}")

    for model_name, (provider, model_id) in MODELS.items():
        try:
            advice, elapsed = call_model(provider, model_id, profile["prompt"])
            scores = judge_advice(profile["programme"], profile["prompt"], advice)
            avg = statistics.mean(scores.values()) if scores else 0.0
            results[model_name].append({
                "profile": profile["label"],
                "programme": profile["programme"],
                "advice": advice,
                "elapsed": elapsed,
                **scores,
                "avg": avg,
            })
            spec = scores.get("specificity", 0)
            act  = scores.get("actionability", 0)
            pfit = scores.get("programme_fit", 0)
            strc = scores.get("structure", 0)
            print(f"  {model_name:<20} avg={avg:.2f}/5  "
                  f"spec={spec} act={act} fit={pfit} struct={strc}  [{elapsed}s]")
        except Exception as e:
            print(f"  {model_name:<20} ERROR: {e}")
            results[model_name].append({"profile": profile["label"], "avg": 0, "elapsed": 0})


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n\n{'='*72}")
print(f"{'MODEL':<22} {'Specificity':>12} {'Actionability':>14} {'Prog.Fit':>10} {'Structure':>10} {'Overall':>9} {'Time':>7}")
print(f"{'-'*72}")

for model_name, model_results in results.items():
    def safe_mean(key):
        vals = [r.get(key, 0) for r in model_results if r.get(key, 0) > 0]
        return statistics.mean(vals) if vals else 0.0

    sp  = safe_mean("specificity")
    ac  = safe_mean("actionability")
    pf  = safe_mean("programme_fit")
    st  = safe_mean("structure")
    ov  = safe_mean("avg")
    tm  = safe_mean("elapsed")
    print(f"{model_name:<22} {sp:>12.2f} {ac:>14.2f} {pf:>10.2f} {st:>10.2f} {ov:>9.2f} {tm:>6.2f}s")

print(f"{'='*72}")


# ── Side-by-side for Profile 2 (SWE Year-3 — most revealing case) ────────────
print("\n\n--- Side-by-side: SWE Year-3 portfolio (Profile 2) ---")
for model_name, model_results in results.items():
    match = next((r for r in model_results if "SWE" in r.get("profile","")), None)
    if match:
        print(f"\n=== {model_name} ===")
        print(match.get("advice","")[:800])
        print("...")

print("\nDone. Use these results to select the production model in career_page.py")
