# Model Evaluation Report
## Explainable AI Learning Analytics Platform — Zewail City

---

## Target 1: GPA Prediction (Regression)

| Model | R² | MAE | RMSE |
|-------|-----|-----|------|
| linear_regression | 0.9864 | 0.0767 | 0.1326 |
| xgboost | 0.9989 | 0.0288 | 0.0381 |
| lightgbm | 0.9990 | 0.0273 | 0.0360 |

**Production Model: XGBoost**
- R²: 0.9989 (explains 99.9% of GPA variance)
- MAE: 0.0288 GPA points average error
- RMSE: 0.0381

### Top Feature Importances (GPA Model)
| Feature | Importance |
|---------|-----------|
| avg_overall | 0.5264 |
| failed_course_ratio | 0.1637 |
| avg_final | 0.1180 |
| avg_midterm | 0.0966 |
| credit_completion_ratio | 0.0690 |
| failed_courses | 0.0123 |
| avg_labs | 0.0090 |
| avg_assignments | 0.0032 |
| avg_quizzes | 0.0008 |
| credits_passed | 0.0002 |

---

## Target 2: Risk Classification (Low / Medium / High)

| Model | Accuracy | F1 (weighted) | AUC |
|-------|----------|---------------|-----|
| logistic_regression | 0.9733 | 0.9734 | N/A |
| xgboost | 0.9887 | 0.9888 | 0.9995 |
| lightgbm | 0.9887 | 0.9888 | 0.9996 |

**Production Model: XGBoost**

### Classification Report (XGBoost)
```
              precision    recall  f1-score   support

    Low Risk       1.00      0.99      1.00       640
 Medium Risk       0.94      0.99      0.96       226
   High Risk       1.00      0.99      0.99       634

    accuracy                           0.99      1500
   macro avg       0.98      0.99      0.98      1500
weighted avg       0.99      0.99      0.99      1500

```

### Confusion Matrix (XGBoost)
```
Predicted: Low   Med   High
Low Risk :   635     5     0
Med Risk :     0   223     3
High Risk:     0     9   625
```

### Top Feature Importances (Risk Model)
| Feature | Importance |
|---------|-----------|
| avg_overall | 0.3954 |
| failed_courses | 0.1062 |
| avg_final | 0.0955 |
| avg_midterm | 0.0852 |
| avg_attendance | 0.0654 |
| credit_completion_ratio | 0.0602 |
| failed_course_ratio | 0.0461 |
| attendance_risk_score | 0.0450 |
| avg_labs | 0.0288 |
| avg_quizzes | 0.0188 |

---

## Conclusion

Both XGBoost models achieve strong performance and are selected as production models.
SHAP explainability is applied in Phase 6.

---

## Product E — GitHub Career Advisor: LLM Model Selection

Experiment script: `compare_career_models.py`  
Judge model: `gpt-4o-mini` (blind scoring, 1–5 per criterion)  
Profiles tested: 3 (CSAI Year-2, SWE Year-3, DSAI Year-4)

### Evaluation Criteria
| Criterion | Description |
|-----------|-------------|
| Specificity | References exact repo names, percentages, and data from the student's profile |
| Actionability | Suggestions are concrete and immediately doable this week |
| Programme Fit | Advice is tailored to the student's specific programme (CSAI / SWE / DSAI) |
| Structure | Follows required 5-section format (Verdict / Strengths / Gaps / Projects / Battle Plan) |

### Results (avg across 3 profiles)

| Model | Specificity | Actionability | Prog. Fit | Structure | Overall | Avg Time |
|-------|-------------|---------------|-----------|-----------|---------|----------|
| **gpt-4o** | **5.00** | 4.67 | 4.67 | **5.00** | **4.75/5** | **4.52s** |
| gpt-4o-mini | 4.00 | **5.00** | **5.00** | **5.00** | 4.75/5 | 5.47s |

### Decision: GPT-4o selected for production

Both models achieve identical overall average (4.75/5). GPT-4o is selected because:
- **Higher specificity (5.0 vs 4.0)** — the most critical dimension; advice must reference exact repo names, metrics, and percentages to be actionable
- **Faster response time (4.52s vs 5.47s)** — better UX during portfolio analysis
- GPT-4o-mini produces more generic advice bodies despite correct structure

Cost trade-off is acceptable: the career advisor is a low-frequency, high-value interaction (once per portfolio analysis, not per chat message).

> **Note on Claude claude-sonnet-4-6:** Included in the experiment design (`compare_career_models.py`) but excluded from the final run due to a missing `anthropic` package in the evaluation environment. Re-evaluation recommended — Claude is expected to match GPT-4o on specificity based on its structured long-form writing strengths.

Full per-profile breakdown: `learning_analytics_xai/models/career_advisor_model_comparison.json`
