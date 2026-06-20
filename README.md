# AI-Powered Smart Campus Assistant

> An intelligent, unified AI platform for Zewail City students — academic advising, GPA forecasting, learning analytics, career readiness, and explainable AI in a single Streamlit application.

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.41-red)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.6-orange)](https://trychroma.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)](https://pytorch.org)

---

## Team Members

| Name | ID | Program |
|------|----|---------|
| Omar Mohamed | 202202184 | DSAI |
| Ahmed Mongi | 202201897 | DSAI |
| Joumana Mohamed | 202202100 | DSAI |
| Sama Samed | 202202118 | DSAI |

**Supervisor:** Dr. / Prof. Mohamed Fakhry Eldin Ghalwash

**School:** School of Computational Sciences and Artificial Intelligence (CSAI)  
**Institution:** Zewail City of Science and Technology  
**Submission Date:** June 2026

---

## Problem Statement

Students at Zewail City face significant challenges navigating their academic journey without personalized, intelligent support:

- **Complex curriculum** — 132 credit hours across 7+ programs with strict prerequisite chains
- **No personalized planning** — course selection relies on manual handbook reading and peer advice
- **No GPA forecasting** — students cannot predict academic risk before it becomes critical
- **No career readiness feedback** — no tool connects academic progress to industry expectations
- **Fragmented data** — academic history is spread across Google Classroom, handbooks, and portals with no unified view

This project builds the missing intelligent layer: a single AI platform that knows the Zewail curriculum, understands each student's situation, and provides actionable, explainable advice.

---

## Features

### Product A — Academic Advisor AI (RAG + Conversational Memory)
- Answers questions about courses, prerequisites, graduation requirements, policies, and campus life
- Retrieves from 4,841 indexed chunks across 225 Zewail sources (PDFs + website)
- 7-intent router: planning, graduation, prerequisite, risk, profile, general, course content
- Prerequisite graph traversal (137 courses) with Safe / Balanced / Fast planning tiers
- Program contamination protection — correct retrieval even after switching programs mid-session
- Transcript-first planning — never fabricates a semester plan without real course history

### Product B — Learning Analytics & XAI Dashboard
- Google Classroom OAuth 2.0 integration — imports real course history automatically
- 4-priority course matching: section code → alias → substring → fuzzy Jaccard
- Business rules: internship auto-pass (x399), GP Part 1 auto-pass when Part 2 exists (x498/499)
- SHAP explainability — per-feature GPA impact waterfall charts
- Plotly donut chart — 7-category course performance distribution
- Credit Audit Report — 5-metric dashboard (raw entries, dedup, verified credits, matched, unresolved)

### Product C — GPA Trajectory Forecasting (LSTM)
- Bidirectional LSTM with Temporal Attention and Static Encoder
- 3-quantile output: optimistic / realistic / pessimistic GPA trajectory
- R² = 0.9595 — outperforms Transformer (0.9602 comparable) and Prophet (−0.7836)
- Trained with Pinball/quantile loss + AdamW + CosineAnnealingLR

### Product D — Student Archetype Clustering (GMM)
- Gaussian Mixture Model clustering on 21 academic features
- Named archetypes: High Achiever, At-Risk, Steady Performer, etc.
- Interactive 2D cluster scatter with archetype labels

### Product E — GitHub Career Advisor
- 4-pillar evaluation: Documentation | Code Quality | Maintenance | Community
- GitHub API — repository analysis, language breakdown, star/fork metrics
- LLM-generated gap analysis vs CSAI programme requirements
- Actionable career recommendations per pillar

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STUDENT (Browser)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼─────────────────────────────────────┐
│              Streamlit App  (phase7_streamlit_app.py)            │
│   Tabs: Chat | Learning Analytics & XAI | Student Archetypes     │
│         GPA Forecast | Career Advisor                            │
└──┬──────────┬──────────────┬──────────────┬──────────────┬──────┘
   │          │              │              │              │
   ▼          ▼              ▼              ▼              ▼
┌──────┐ ┌────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐
│Intent│ │Google  │  │LSTM Model │  │   GMM    │  │  GitHub API  │
│Router│ │OAuth   │  │(PyTorch)  │  │Clustering│  │  Analyzer    │
│      │ │Classroom│ │Forecasting│  │(Product D│  │(Product E)   │
└──┬───┘ └───┬────┘  └───────────┘  └──────────┘  └──────────────┘
   │         │
   ▼         ▼
┌──────────────────────────────────────────────────────┐
│            RAG Pipeline  (phase5_rag_pipeline.py)     │
│  retrieve() → rerank() → source_diversity_cap()       │
│              → generate() via GPT-4o-mini             │
└──────────────────┬───────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │     ChromaDB         │
        │  4,841 chunks        │
        │  225 sources         │
        │  text-embedding-3-sm │
        └─────────────────────┘

DATA SOURCES INDEXED:
  • CSAI / BUS / SCI / ENGR Curricula PDFs (2022–2023)
  • Zewail City website (225 scraped pages)
  • Admissions, scholarships, academic regulations
  • course_catalog.json | prerequisites_graph.json | study_plans.json
```

---

## Technologies Used

### Frontend
- **Streamlit 1.41** — multi-tab single-page web application
- **Plotly** — interactive donut charts, scatter plots, GPA trajectory forecasts
- Custom dark CSS theme (#0F172A background, #7C3AED accent)

### Backend
- **Python 3.12**
- **OpenAI GPT-4o-mini** — intent classification and answer generation
- **OpenAI text-embedding-3-small** — document embeddings
- **LangChain** — document chunking and retrieval utilities

### Database
- **ChromaDB** — local vector database (4,841 chunks, 225 sources, 1536-dim embeddings)
- **JSON files** — course catalog, prerequisite graph, study plans, academic regulations
- **Session JSON** — per-student conversational state (data/sessions/)

### AI / ML Frameworks
- **PyTorch 2.6** — LSTM GPA forecasting model (Product C)
- **XGBoost** — GPA regression and risk classification models (Product B)
- **SHAP** — TreeExplainer for XAI feature attribution
- **Scikit-learn** — GMM clustering, StandardScaler, feature pipeline (Product D)

### Integrations
- **Google OAuth 2.0** — Google Classroom course history import
- **GitHub REST API** — repository analysis for Career Advisor (Product E)
- **Playwright** — headless Chromium for website scraping (Phases 1–2)

### DevOps
- **Git / GitHub** — version control and collaboration
- **python-dotenv** — environment variable management
- **pytest** — automated test suite

---

## Project Structure

```
AI-Powered-Smart-Campus-Assistant/
│
├── phase1_scrape_website.py          # Web crawler (Playwright)
├── phase2_extract_pdfs.py            # PDF downloader
├── phase3_clean_data.py              # Text cleaning + categorisation
├── phase4_chunk_and_embed.py         # Chunking + ChromaDB indexing
├── phase5_rag_pipeline.py            # RAG: CampusRAG class
├── phase6_conversational_memory.py   # Conversational session manager
├── phase7_streamlit_app.py           # Main Streamlit app (entry point)
├── phase8_advisor_engine.py          # Academic Advisor AI engine
├── phase8a_build_curriculum.py       # Curriculum graph builder
│
├── learning_analytics_xai/           # Products B, C, D, E
│   ├── dashboard/
│   │   ├── analytics_page.py         # XAI dashboard (Product B)
│   │   ├── forecasting_page.py       # GPA forecast UI (Product C)
│   │   ├── clustering_page.py        # Archetypes UI (Product D)
│   │   ├── career_page.py            # Career advisor UI (Product E)
│   │   └── study_tools_page.py       # Study tools
│   ├── data/
│   │   ├── course_catalog.json       # All Zewail courses + credits + prereqs
│   │   ├── prerequisites_graph.json  # Full prerequisite DAG (137 nodes)
│   │   ├── degree_requirements.json  # Per-programme credit requirements
│   │   ├── academic_regulations.json # Probation, GPA thresholds, policies
│   │   └── gpa_rules.json            # Grade-to-GPA conversion table
│   ├── data_ingestion/
│   │   ├── classroom_importer.py     # Google Classroom OAuth + import
│   │   └── student_profile_builder.py# Course matching + deduplication
│   ├── models/
│   │   ├── gpa_model_xgb.pkl         # XGBoost GPA regression (pre-trained)
│   │   ├── risk_model_xgb.pkl        # XGBoost risk classifier (pre-trained)
│   │   ├── student_clustering.pkl    # GMM clustering model (pre-trained)
│   │   └── model_metrics.json        # R², MAE, RMSE, accuracy scores
│   ├── forecasting/
│   │   └── gpa_forecaster.py         # LSTM + Attention model definition
│   ├── curriculum_intelligence/
│   │   └── curriculum_engine.py      # Prerequisite graph + plan engine
│   ├── xai/
│   │   └── explainability.py         # SHAP TreeExplainer wrapper
│   ├── feature_engineering/
│   │   └── feature_engineer.py       # 21 + 8 curriculum features
│   ├── recommendation_engine/
│   │   └── recommender.py            # Curriculum-first recommendations
│   ├── career/
│   │   └── github_analyzer.py        # GitHub 4-pillar portfolio analyzer
│   └── what_if_analysis/
│       └── what_if_engine.py         # What-if scenario simulator
│
├── data/
│   ├── clean/
│   │   └── cleaned_documents.jsonl   # Phase 3 output
│   ├── curriculum/
│   │   ├── courses.json              # Curriculum course list
│   │   └── study_plans.json          # Official semester sequences
│   └── sessions/                     # Per-session conversation state
│
├── db/
│   └── chroma_db/                    # ChromaDB vector store (pre-built)
│       └── chroma.sqlite3            # 4,841 chunks, 225 sources
│
├── model_comparisons/                # Evaluation notebooks + results
│   ├── notebook1_llm_comparison.ipynb
│   ├── notebook2_ml_comparison.ipynb
│   ├── notebook3_forecasting_comparison.ipynb
│   ├── notebook4_rag_vs_vanilla_evaluation.ipynb
│   ├── rag_vs_vanilla_results.csv    # 82.5% RAG vs 2.5% vanilla results
│   └── forecasting_r2_results.md
│
├── tests/                            # Automated test suite
│   ├── test_phase1_scrape.py
│   ├── test_phase5_rag.py
│   └── test_phase8_advisor.py
│
├── screenshots/                      # App screenshots and demo outputs
│   └── README.md
│
├── docs/                             # Extended documentation
│   ├── API.md                        # API and module reference
│   ├── DATABASE_SCHEMA.md            # Data models and schema
│   └── USER_GUIDE.md                 # End-user usage guide
│
├── .env.example                      # Environment variable template
├── requirements.txt                  # Python dependencies
├── runtime.txt                       # Python version pin (3.12)
└── README.md                         # This file
```

---

## Setup Instructions

### Prerequisites
- Python 3.12
- Git
- An OpenAI API key ([platform.openai.com](https://platform.openai.com/api-keys))
- A Google Cloud OAuth 2.0 Client ID and Secret (for Classroom import — optional)

### Step 1 — Clone the repository

```bash
git clone https://github.com/saamasameh889/-Al-Powered-Smart-Campus-Assistant---Graduation-Project.git
cd -Al-Powered-Smart-Campus-Assistant---Graduation-Project
```

### Step 2 — Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch (GPA Forecasting) requires a special index URL:
> ```bash
> pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
> ```

### Step 4 — Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-proj-...          # Required — for RAG and advisor
GOOGLE_CLIENT_ID=...                 # Optional — for Classroom import
GOOGLE_CLIENT_SECRET=...             # Optional — for Classroom import
GOOGLE_REDIRECT_URI=http://localhost:8501
```

### Step 5 — Verify the setup (optional smoke test)

```bash
python smoke_test.py
```

Expected output: `All imports OK — ready to run.`

---

## Deployment Instructions

### Option A — Local deployment (recommended for development)

```bash
streamlit run phase7_streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

The ChromaDB vector store (`db/chroma_db/`) and all pre-trained models are already committed to the repository — **no re-indexing or re-training required**.

### Option B — Rebuild the RAG index from scratch

Only needed if you want to re-scrape the Zewail website or update the knowledge base:

```bash
python phase1_scrape_website.py    # ~10 min — scrape website (JS-rendered)
python phase2_extract_pdfs.py      # Download PDF handbooks
python phase3_clean_data.py        # Clean and categorise documents
python phase4_chunk_and_embed.py   # Chunk + embed into ChromaDB
```

### Option C — Streamlit Community Cloud

1. Fork the repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set `phase7_streamlit_app.py` as the main file
4. Add secrets in the Streamlit Cloud dashboard (Settings → Secrets):
   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   GOOGLE_CLIENT_ID = "..."
   GOOGLE_CLIENT_SECRET = "..."
   GOOGLE_REDIRECT_URI = "https://your-app.streamlit.app"
   ```
5. Click **Deploy**

> Python version is pinned in `runtime.txt` as `python-3.12`.

---

## Usage Guide

See [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for full usage instructions.

### Quick Start

1. Open the app at [http://localhost:8501](http://localhost:8501)
2. **Chat tab** — type any question about Zewail courses, prerequisites, graduation, or policies
3. **Learning Analytics & XAI tab** — click "Connect Google Classroom" to import your course history, then view your performance dashboard
4. **GPA Forecast tab** — enter your academic profile and see your predicted GPA trajectory
5. **Student Archetypes tab** — discover which learning archetype matches your profile
6. **Career Advisor tab** — enter your GitHub username to get a portfolio gap analysis

### Example Chat Interactions

```
You: How many credits do I need to graduate from CSAI?
Advisor: CSAI requires 132 credit hours: 21 University Requirements +
         53 School Requirements + 58 Programme Requirements.

You: I'm in CSAI semester 4. I completed CSAI 101, MATH 101, PHYS 101,
     CSAI 201. I failed Machine Learning. What should I take next?
Advisor: [asks for your full completed course list to build an accurate plan]

You: What is the prerequisite for CSAI 302?
Advisor: CSAI 302 requires CSAI 201 and MATH 202 to be completed first.
```

---

## Screenshots / Demo

Screenshots are in the [`screenshots/`](screenshots/) folder.

| Screen | Description |
|--------|-------------|
| `screenshots/01_chat.png` | Academic Advisor chat interface |
| `screenshots/02_xai_dashboard.png` | Learning Analytics & XAI dashboard |
| `screenshots/03_donut_chart.png` | Course performance Plotly donut chart |
| `screenshots/04_gpa_forecast.png` | LSTM GPA trajectory forecast |
| `screenshots/05_archetypes.png` | Student archetype clustering |
| `screenshots/06_career_advisor.png` | GitHub Career Advisor |
| `screenshots/07_classroom_import.png` | Google Classroom OAuth sign-in |

---

## Testing and Evaluation

### Run the test suite

```bash
pytest tests/ -v --tb=short
```

### Key evaluation results

| Metric | Result |
|--------|--------|
| RAG accuracy (40-question benchmark) | **82.5%** |
| Vanilla GPT-4o-mini accuracy (same benchmark) | 2.5% |
| RAG improvement over baseline | **+80 percentage points** |
| GPA Forecasting R² (LSTM) | **0.9595** |
| GPA Forecasting R² (Transformer) | 0.9602 |
| GPA Forecasting R² (Prophet) | −0.7836 |
| Intent routing accuracy | **11 / 11 test cases** |
| Course matching (previously unmatched) | **11 / 11 resolved** |
| Program contamination (BUS→CSAI bleed) | **0 BUS chunks in CSAI queries** |

Evaluation notebooks and results are in [`model_comparisons/`](model_comparisons/).

---

## API Documentation

See [docs/API.md](docs/API.md) for full module and function reference.

### Core modules at a glance

| Module | Class / Function | Description |
|--------|-----------------|-------------|
| `phase5_rag_pipeline.py` | `CampusRAG` | RAG retrieval and generation |
| `phase6_conversational_memory.py` | `ConversationalAssistant` | Session + memory management |
| `phase8_advisor_engine.py` | `AdvisorEngine` | 7-intent academic advisor |
| `phase8_advisor_engine.py` | `IntentRouter` | Query intent classification |
| `learning_analytics_xai/data_ingestion/classroom_importer.py` | `ClassroomImporter` | Google Classroom OAuth + course import |
| `learning_analytics_xai/curriculum_intelligence/curriculum_engine.py` | `CurriculumEngine` | Prerequisite graph + plan engine |
| `learning_analytics_xai/forecasting/gpa_forecaster.py` | `GPAForecaster` | LSTM GPA trajectory prediction |
| `learning_analytics_xai/career/github_analyzer.py` | `GitHubAnalyzer` | 4-pillar portfolio analysis |

---

## Database Schema

See [docs/DATABASE_SCHEMA.md](docs/DATABASE_SCHEMA.md) for full schema documentation.

### ChromaDB Collections

| Collection | Chunks | Sources | Embedding Model | Dimension |
|-----------|--------|---------|----------------|-----------|
| `zewail_campus` | 4,841 | 225 | text-embedding-3-small | 1,536 |

### JSON Knowledge Base

| File | Records | Purpose |
|------|---------|---------|
| `course_catalog.json` | 150+ courses | Course codes, credits, prerequisites, aliases |
| `prerequisites_graph.json` | 137 nodes | Full prerequisite DAG |
| `study_plans.json` | 7 programs | Official semester-by-semester sequences |
| `degree_requirements.json` | 7 programs | Credit requirements per school/program |
| `academic_regulations.json` | — | Probation rules, GPA thresholds, policies |

---

## Known Limitations

- **Base tuition amount** — not indexed (the DB contains only refund schedules and scholarship percentages). The advisor redirects to [admission-and-scholarships](https://www.zewailcity.edu.eg/admission-and-scholarships).
- **GPA Forecasting** — requires PyTorch (~220 MB). Gracefully disabled if torch is unavailable.
- **Google Classroom import** — requires valid OAuth credentials (GOOGLE_CLIENT_ID / SECRET). The advisor functions fully without them.
- **Website content** — scraped at build time. Re-run Phases 1–4 to update with new Zewail content.

---

## License

This project is submitted as a graduation project for academic purposes at Zewail City of Science and Technology, June 2026. All rights reserved by the project team.
