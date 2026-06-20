# User Guide

Step-by-step guide for students using the AI-Powered Smart Campus Assistant.

---

## Getting Started

Open the app at **http://localhost:8501** after following the [setup instructions](../README.md#setup-instructions).

The app has **5 tabs** across the top navigation bar:

| Tab | Product | What it does |
|-----|---------|-------------|
| Chat | A | Ask any academic question |
| Learning Analytics & XAI | B | Import Classroom data, view analytics |
| Student Archetypes | D | Discover your learning archetype |
| GPA Forecast | C | Predict your GPA trajectory |
| Career Advisor | E | GitHub portfolio gap analysis |

---

## Tab 1 — Chat (Academic Advisor AI)

### What you can ask

**Factual questions (no profile needed):**
```
How many credits are required to graduate from CSAI?
What is the prerequisite for Machine Learning?
What does CSAI 302 cover?
What are the graduation requirements for BUS?
What is the academic probation policy?
```

**Personal planning (tell the advisor about yourself first):**
```
I'm in CSAI, semester 4. My GPA is 3.1.
I completed CSAI 101, MATH 101, PHYS 101, CSAI 201, MATH 202.
I'm currently taking CSAI 301 and MATH 301.
What should I register next semester?
```

**Prerequisite checks:**
```
Can I take CSAI 402 if I completed CSAI 301?
What courses does failing Machine Learning block?
```

**Graduation timeline:**
```
Am I on track to graduate in 4 years?
When will I graduate at my current pace?
How many semesters do I have left?
```

### How the Advisor builds your profile

As you chat, the advisor automatically extracts:
- Your school and program (e.g. "I'm in CSAI")
- Your semester (e.g. "I'm in semester 4")
- Your GPA (e.g. "my GPA is 3.1")
- Completed courses (e.g. "I completed CSAI 101, MATH 101")
- Failed courses (e.g. "I failed Machine Learning")

Your profile appears in the **sidebar** on the left. You can start fresh with **"+ New Conversation"**.

### Planning — what the advisor needs

For a semester plan, the advisor requires:
1. Your program (e.g. CSAI — DSAI track)
2. Your current semester
3. Your **actual completed course list** (not just semester number)

If any of these are missing, the advisor will ask for them before generating a plan. This ensures the plan is based on your real academic history, not a guess.

---

## Tab 2 — Learning Analytics & XAI

### Option A — Connect Google Classroom (Recommended)

1. Click **"Connect Google Classroom"**
2. Sign in with your `@zewailcity.edu.eg` account
3. The app imports all your courses automatically
4. View your **Credit Audit Report** — matched courses, verified credits, unresolved names

### Option B — Enter Manually

1. Click **"Import or Enter Manually"**
2. Type your academic profile in the text area
3. Click **"Analyze"**

### What the Dashboard Shows

After import:
- **Performance Donut Chart** — distribution of your courses across Excellent / Strong / Average / Weak / At-Risk / In Progress / No Data
- **GPA Risk Prediction** — Low / Medium / High based on XGBoost model
- **SHAP Waterfall** — which factors most affect your predicted GPA
- **Credit Audit Report** — raw entries, after dedup, verified credits, matched vs unresolved
- **Recommendations** — prioritised by curriculum impact (critical first)
- **What-If Simulator** — adjust attendance, scores, and see projected GPA impact

### Google Classroom Matching

The app automatically matches Classroom course names to the official Zewail catalog using a 4-priority system:
1. Exact section code (e.g. `CSAI-490-LCTR-01` → `CSAI490`)
2. Alias lookup (e.g. "Graduation Project" → `CSAI499`)
3. Title substring match
4. Fuzzy Jaccard similarity

**Business rules applied automatically:**
- Internship courses (x399) → always marked Completed
- Graduation Project Part 1 (x498) → auto-passed if Part 2 (x499) exists in your Classroom
- Duplicate entries → highest-score attempt kept; lab variants (CSAI201L) merged to parent (CSAI201)

---

## Tab 3 — Student Archetypes

1. Enter your academic profile (GPA, attendance, semester, failed courses, credit load)
2. Click **"Analyze My Archetype"**
3. View your cluster position on the scatter plot
4. Read your named archetype and what it means for your academic strategy

**Archetypes include:** High Achiever, Steady Performer, At-Risk Student, Comeback Student, Overloader, and others.

---

## Tab 4 — GPA Forecast

1. Enter your current GPA, semester, program, and recent semester scores
2. Click **"Forecast"**
3. View the trajectory chart showing:
   - **Optimistic trajectory** (q90)
   - **Realistic trajectory** (q50 — median prediction)
   - **Pessimistic trajectory** (q10)

The LSTM model (R² = 0.9595) predicts your GPA for the next 2–4 semesters based on your academic pattern.

> **Note:** Requires PyTorch. If "Forecasting import failed" appears, PyTorch is not installed — run:
> ```bash
> pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Tab 5 — Career Advisor

1. Enter your **GitHub username**
2. Select your **programme** (e.g. CSAI)
3. Enter your **semester**
4. Optionally add a **GitHub token** (increases API rate limit from 60 to 5,000 req/hour)
5. Click **"Analyze Portfolio"**

**The report shows:**
- **Overall score** (0–100) across 4 pillars
- **Documentation score** — README quality, docstrings, project descriptions
- **Code Quality score** — language diversity, structure, commit messages
- **Maintenance score** — recent activity, issue management, releases
- **Community score** — stars, forks, watchers, contributors
- **Gap analysis** — what's missing compared to CSAI programme expectations
- **Actionable recommendations** — specific steps to improve each pillar

---

## Tips for Best Results

1. **Be specific about your courses** — use official codes when possible (CSAI 101, not "intro programming")
2. **Share your full completed list** — the more context you give, the more accurate the plan
3. **Use the Classroom import** — it's faster and more accurate than manual entry
4. **Ask follow-up questions** — the advisor remembers your profile within the session
5. **Check the sources** — every chat answer shows which documents were used (expand "Sources")

---

## Frequently Asked Questions

**Q: Does the advisor store my data?**  
A: Conversation sessions are stored locally in `data/sessions/` on the machine running the app. Nothing is sent externally except to OpenAI for answer generation and Google for Classroom import.

**Q: Why does the advisor ask for my course list before making a plan?**  
A: The advisor uses a Transcript-First Policy — it only generates a semester plan from your actual course history. Inferring from your semester number alone can produce wrong recommendations if you took summers, failed courses, or deviated from the standard sequence.

**Q: The Classroom import shows some "Unresolved" courses — what does that mean?**  
A: These are courses that don't match any entry in the Zewail course catalog. They may have unusual names in Classroom. You can ignore them if they're not credit-bearing, or note them manually.

**Q: Can I use a personal Gmail account for Classroom import?**  
A: The system requests access to your Google Classroom data. You must approve access through the OAuth consent screen. Your Zewail `@zewailcity.edu.eg` account is recommended.
