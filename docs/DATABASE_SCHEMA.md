# Database Schema

Complete documentation of all data stores used by the AI-Powered Smart Campus Assistant.

---

## 1. ChromaDB Vector Store

**Location:** `db/chroma_db/chroma.sqlite3`  
**Type:** Local vector database (ChromaDB 0.6)  
**Collection name:** `zewail_campus`

### Collection Statistics

| Property | Value |
|----------|-------|
| Total chunks | 4,841 |
| Unique sources | 225 |
| Embedding model | `text-embedding-3-small` (OpenAI) |
| Embedding dimension | 1,536 |
| Distance metric | Cosine similarity |

### Document Schema (per chunk)

```json
{
  "id":       "uuid-string",
  "document": "chunk text content (200–500 tokens)",
  "metadata": {
    "source":   "CSAI - Policies and Regulations 2023.pdf",
    "page":     12,
    "category": "policy | curriculum | web | general",
    "title":    "Section 6.1 – Tuition and Scholarships"
  },
  "embedding": [1536-dimensional float vector]
}
```

### Source Categories

| Category | Description | Example Sources |
|----------|-------------|----------------|
| `curriculum` | Programme handbooks, curricula PDFs | `CSAI - Curricula 2022.pdf` |
| `policy` | Regulations, rules, probation policies | `BUS - Policies and Regulations 2023.pdf` |
| `web` | Scraped website pages | `https://www.zewailcity.edu.eg/program/40` |
| `general` | Admissions, scholarships, contacts | `https://admissions.zewailcity.edu.eg/` |

---

## 2. Course Catalog (`learning_analytics_xai/data/course_catalog.json`)

### Schema

```json
{
  "CSAI101": {
    "code":          "CSAI101",
    "name":          "Introduction to Programming",
    "credits":       3,
    "prerequisites": [],
    "school":        "CSAI",
    "aliases":       ["intro to programming", "programming 1", "CSAI 101"]
  },
  "CSAI201": {
    "code":          "CSAI201",
    "name":          "Data Structures",
    "credits":       3,
    "prerequisites": ["CSAI101"],
    "school":        "CSAI",
    "aliases":       ["data structures and algorithms", "DSA"]
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `code` | `string` | Canonical course code (no spaces, uppercase) |
| `name` | `string` | Official course name |
| `credits` | `int` | Credit hours |
| `prerequisites` | `list[str]` | Direct prerequisite course codes |
| `school` | `string` | Owning school (CSAI, BUS, SCI, ENGR, SCH) |
| `aliases` | `list[str]` | Alternative names for fuzzy matching |

---

## 3. Prerequisite Graph (`learning_analytics_xai/data/prerequisites_graph.json`)

### Schema

```json
{
  "nodes": [
    {"id": "CSAI101", "name": "Intro to Programming", "credits": 3},
    {"id": "CSAI201", "name": "Data Structures",      "credits": 3}
  ],
  "edges": [
    {"from": "CSAI101", "to": "CSAI201", "type": "prerequisite"}
  ]
}
```

**Graph properties:**
- Directed Acyclic Graph (DAG)
- 137 nodes (courses)
- Traversal: `get_all_dependents(codes)` returns full transitive closure of blocked courses

---

## 4. Study Plans (`data/curriculum/study_plans.json`)

### Schema

```json
{
  "CSAI": {
    "DSAI": {
      "semesters": {
        "1": ["CSAI101", "MATH101", "PHYS101", "ENGL001", "SCH101"],
        "2": ["CSAI201", "MATH202", "PHYS201", "ENGL002"],
        "3": ["CSAI301", "CSAI302", "MATH301"],
        "4": ["CSAI401", "CSAI402", "CSAI403"]
      },
      "total_credits": 132
    },
    "SWD": {
      "semesters": { ... },
      "tracks": {
        "APD": { "semesters": { ... } },
        "GCG": { "semesters": { ... } }
      }
    }
  }
}
```

---

## 5. Degree Requirements (`learning_analytics_xai/data/degree_requirements.json`)

### Schema

```json
{
  "CSAI": {
    "total":                132,
    "university_required":  21,
    "school_required":      53,
    "programme_required":   58,
    "minimum_gpa":          2.0,
    "internship_required":  true,
    "graduation_project":   true
  },
  "BUS": {
    "total":                114,
    "university_required":  21,
    "school_required":      25,
    "programme_required":   68
  }
}
```

---

## 6. Academic Regulations (`learning_analytics_xai/data/academic_regulations.json`)

### Schema

```json
{
  "probation": {
    "threshold_gpa":         2.0,
    "warning_gpa":           2.2,
    "max_probation_semesters": 2
  },
  "grade_scale": {
    "A":  4.0,
    "A-": 3.7,
    "B+": 3.3,
    "B":  3.0,
    "B-": 2.7,
    "C+": 2.3,
    "C":  2.0,
    "D":  1.0,
    "F":  0.0
  },
  "max_credits_per_semester": 21,
  "min_credits_per_semester": 12,
  "withdrawal_deadline_week": 10
}
```

---

## 7. XGBoost Models (Product B)

| File | Type | Task | Metric |
|------|------|------|--------|
| `models/gpa_model_xgb.pkl` | XGBoost Regressor | Predict cumulative GPA (0–4) | R² = 0.87, MAE = 0.18 |
| `models/risk_model_xgb.pkl` | XGBoost Classifier | Low / Medium / High risk | Accuracy = 91% |
| `models/scaler.pkl` | StandardScaler | Normalise 21 input features | — |
| `models/student_clustering.pkl` | GMM (3 components) | Student archetype clustering | Silhouette = 0.63 |

### Feature Vector (21 features)

```
avg_midterm, avg_final, avg_overall, avg_attendance,
failed_courses, retaken_courses, completed_credits,
gpa_trend (slope), semester, programme_encoded,
lab_courses_count, credit_load, pass_rate,
attendance_consistency, midterm_final_gap,
high_risk_courses_count, prereq_completion_proxy,
graduation_progress_ratio, core_completion_ratio,
curriculum_alignment_proxy, curriculum_readiness_score
```

---

## 8. Conversation Sessions (`data/sessions/`)

### File: `data/sessions/<session_id>.json`

```json
{
  "session_id": "uuid",
  "created_at": "2026-06-20T14:30:00Z",
  "profile": {
    "school":             "CSAI",
    "major":              "DSAI",
    "semester":           4,
    "gpa":                3.1,
    "completed_courses":  ["CSAI101", "MATH101", "CSAI201"],
    "failed_courses":     [],
    "current_courses":    ["CSAI301"],
    "completed_credits":  60
  },
  "history": [
    {"role": "user",      "content": "How many credits to graduate from CSAI?"},
    {"role": "assistant", "content": "CSAI requires 132 credit hours..."}
  ]
}
```

---

## 9. Environment Variables (`.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** | OpenAI API key for embeddings + generation |
| `GOOGLE_CLIENT_ID` | Optional | Google OAuth 2.0 Client ID (Classroom import) |
| `GOOGLE_CLIENT_SECRET` | Optional | Google OAuth 2.0 Client Secret |
| `GOOGLE_REDIRECT_URI` | Optional | OAuth callback URL (default: `http://localhost:8501`) |
