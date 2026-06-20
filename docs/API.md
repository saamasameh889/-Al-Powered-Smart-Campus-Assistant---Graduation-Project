# API & Module Reference

Complete reference for all core modules in the AI-Powered Smart Campus Assistant.

---

## Product A — RAG Pipeline (`phase5_rag_pipeline.py`)

### Class: `CampusRAG`

The central retrieval-augmented generation engine. Handles embedding, retrieval, reranking, and answer generation.

#### Constructor
```python
CampusRAG(
    db_path: str = "db/chroma_db",
    collection: str = "zewail_campus",
    model: str = "gpt-4o-mini"
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `retrieve` | `(query: str, top_k: int = 8) -> tuple[list[Chunk], str]` | Semantic retrieval with source-diversity cap (max 2 chunks per source). Returns chunks + query note. |
| `rerank` | `(query: str, chunks: list[Chunk], top_n: int = 6) -> list[Chunk]` | Cross-encoder reranking by relevance to query. |
| `generate` | `(question: str, chunks: list[Chunk], history: list[dict], query_note: str) -> str` | GPT-4o-mini generation with retrieved context and conversation history. |
| `classify_intent` | `(question: str) -> str` | Fast intent classification: `"graduation"`, `"prerequisites"`, `"general"`. |
| `answer_with_tools` | `(question: str, history: list[dict], top_k: int) -> tuple[str, list]` | Agentic tool-calling path for calculation-heavy queries. |

#### Data class: `Chunk`
```python
@dataclass
class Chunk:
    text:     str    # Retrieved chunk text
    source:   str    # Source file / URL
    page:     int    # Page number (PDFs) or 0 (web)
    score:    float  # Cosine similarity score (0–1)
    category: str    # "policy" | "curriculum" | "web" | "general"
```

---

## Product A — Conversational Memory (`phase6_conversational_memory.py`)

### Class: `ConversationalAssistant`

Wraps `CampusRAG` with session management, student profile extraction, and the advisor engine.

#### Constructor
```python
ConversationalAssistant(
    rag: CampusRAG,
    advisor: AdvisorEngine,
    session_dir: str = "data/sessions"
)
```

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `ask` | `(question: str, session: ConversationSession, top_k: int = 6) -> tuple[str, list]` | Full pipeline: update profile → try advisor → fall back to RAG. |
| `ask_stream` | `(question: str, session: ConversationSession, top_k: int = 6) -> tuple[list, str\|generator, bool]` | Streaming variant. Returns `(chunks, content, is_streamed)`. |

### Class: `ConversationSession`

| Method | Description |
|--------|-------------|
| `get_profile() -> StudentProfile` | Get current extracted student profile |
| `set_profile(p: StudentProfile)` | Update profile in session |
| `recent_history(n: int = 6) -> list[dict]` | Last N turns as `[{"role": ..., "content": ...}]` |
| `profile_summary() -> str` | Human-readable profile string for LLM context |
| `add_turn(role, content)` | Append a turn to conversation history |

---

## Product A — Academic Advisor Engine (`phase8_advisor_engine.py`)

### Class: `AdvisorEngine`

Seven-intent academic advisor with curriculum awareness, planning engine, and prerequisite graph.

#### Constructor
```python
AdvisorEngine(rag: CampusRAG)
```

#### Public Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `advise` | `(question: str, profile: StudentProfile, history: list[dict]) -> tuple[str\|None, list]` | Main entry point. Returns `(None, [])` for general intent (falls through to RAG). |

#### Key Internal Methods

| Method | Description |
|--------|-------------|
| `_detect_query_program(question) -> tuple[str, str]` | Extract (major, school) from current question. Prevents cross-session program contamination. |
| `_check_planning_readiness(question, profile) -> list[str]` | Returns list of missing info before a plan can be built. Enforces transcript-first policy. |
| `_full_advisory(question, profile, intent, history) -> tuple[str, list]` | Core advisory response: retrieval → planning → LLM generation. |

### Class: `IntentRouter`

| Method | Signature | Description |
|--------|-----------|-------------|
| `classify` | `(query: str, profile: StudentProfile) -> str` | Returns one of: `"planning"`, `"graduation"`, `"prerequisite"`, `"risk"`, `"profile"`, `"general"` |

**Intent routing rules:**
- `planning` — explicit planning keywords ("what should I take", "next semester")
- `graduation` — graduation keywords **AND** personal pronouns (I/my/am I). Factual queries → `general`
- `prerequisite` — prerequisite keywords ("prerequisite for", "can I take")
- `risk` — risk keywords **AND** self-reference (am I, will I)
- `profile` — personal info with no question mark
- `general` — everything else → standard RAG

### Class: `StudentProfile`

```python
@dataclass
class StudentProfile:
    school:            str       # e.g. "CSAI", "BUS", "SCI"
    major:             str       # e.g. "DSAI", "SWD"
    concentration:     str       # e.g. "APD", "GCG"
    semester:          int       # Current semester number
    gpa:               float     # Cumulative GPA
    completed_courses: list[str] # Explicitly confirmed completed courses
    failed_courses:    list[str] # Explicitly confirmed failed courses
    current_courses:   list[str] # Currently enrolled courses
    completed_credits: int       # Total completed credit hours
```

### Module-level function: `update_profile`

```python
update_profile(question: str, profile: StudentProfile) -> StudentProfile
```

Extracts academic information from natural language and incrementally updates the profile. Respects personal-context guard: won't overwrite a known school/major unless the student explicitly provides a correction with personal context ("I am in CSAI").

---

## Product B — Classroom Importer (`learning_analytics_xai/data_ingestion/classroom_importer.py`)

### Function: `build_oauth_flow`
```python
build_oauth_flow(
    redirect_uri: str,
    client_id: str = "",
    client_secret: str = ""
) -> Flow
```
Builds a Google OAuth 2.0 flow. Falls back to `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` env vars.

### Function: `get_auth_url`
```python
get_auth_url(redirect_uri: str, client_id: str = "", client_secret: str = "") -> tuple[str, str]
```
Returns `(authorization_url, state)`. Redirect the student to `authorization_url`.

### Function: `exchange_code_for_token`
```python
exchange_code_for_token(
    redirect_uri: str, code: str, state: str,
    client_id: str = "", client_secret: str = ""
) -> Credentials | None
```
Exchanges the OAuth callback code for Google credentials.

### Class: `ClassroomImporter`
```python
ClassroomImporter(credentials: Credentials)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `get_courses()` | `list[dict]` | All enrolled courses from Google Classroom |
| `get_course_work(course_id)` | `list[dict]` | Assignments + submissions for a course |
| `build_course_records()` | `list[CourseRecord]` | Full import: courses → grades → `CourseRecord` list |

### Class: `CourseRecord`
```python
@dataclass
class CourseRecord:
    course_name:      str
    course_section:   str   # e.g. "CSAI-490-LCTR-01"
    matched_code:     str   # e.g. "CSAI490" (after matching)
    score:            float # Best attempt score (0–100)
    pass_status:      str   # "completed_coursework" | "in_progress" | "not_started"
    credits_verified: bool
    coursework_complete: bool
    in_progress:      bool
```

---

## Product B — Curriculum Engine (`learning_analytics_xai/curriculum_intelligence/curriculum_engine.py`)

### Class: `CurriculumEngine` (singleton)

```python
engine = CurriculumEngine.get_instance()
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `match_course_code` | `(name_or_code: str) -> str \| None` | 4-priority matching: exact section → alias → substring → Jaccard fuzzy |
| `get_prerequisites` | `(code: str) -> list[str]` | Direct prerequisites for a course |
| `get_all_dependents` | `(codes: list[str]) -> list[str]` | Full transitive closure of blocked courses |
| `resolve_prog` | `(school, major) -> tuple[str, str] \| None` | Resolve to (school_key, program_key) |
| `get_current_sem_codes` | `(s, p, track, sem) -> list[str]` | Courses in semester N of official plan |
| `infer_presumed_completed` | `(school, major, sem, track) -> list[str]` | Official courses for semesters 1..N-1 |

---

## Product C — GPA Forecaster (`learning_analytics_xai/forecasting/gpa_forecaster.py`)

### Class: `GPAForecaster`

```python
forecaster = GPAForecaster(model_path="learning_analytics_xai/models/")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `predict` | `(static_features, temporal_sequence) -> dict` | Returns `{"optimistic": [...], "realistic": [...], "pessimistic": [...]}` — one value per future semester |
| `train` | `(dataset_path, epochs, batch_size) -> dict` | Train the LSTM model. Returns training metrics. |

**Model architecture:**
- Input: static features (program, load, etc.) + temporal GPA sequence
- Encoder: Static FC encoder + Bidirectional LSTM
- Attention: Temporal attention over LSTM hidden states
- Output: 3-quantile predictions (q10 / q50 / q90) for H future semesters
- Loss: Pinball (quantile) loss

---

## Product E — GitHub Analyzer (`learning_analytics_xai/career/github_analyzer.py`)

### Class: `GitHubAnalyzer`

```python
analyzer = GitHubAnalyzer(github_token: str = "")
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `analyze_portfolio` | `(username: str, programme: str, semester: int) -> dict` | Full 4-pillar analysis |
| `get_repositories` | `(username: str) -> list[dict]` | All public repositories |
| `score_documentation` | `(repos: list) -> float` | Documentation pillar score (0–100) |
| `score_code_quality` | `(repos: list) -> float` | Code quality pillar score |
| `score_maintenance` | `(repos: list) -> float` | Maintenance pillar score |
| `score_community` | `(repos: list) -> float` | Community impact pillar score |
| `generate_gap_analysis` | `(scores: dict, programme: str) -> str` | LLM-generated gap analysis vs programme requirements |

**4-pillar scoring:**
| Pillar | Signals |
|--------|---------|
| Documentation | README presence, docstrings, wiki, description |
| Code Quality | Language diversity, file structure, meaningful commits |
| Maintenance | Recent commits, open issues, release tags |
| Community | Stars, forks, watchers, contributors |
