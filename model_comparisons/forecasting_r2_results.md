# Forecasting R² Results — GPA Trajectory Forecasting
**Dataset:** 7,500 real Zewail City students × 5 augmentations = 37,500 sequences  
**Task:** Predict GPA for next 3 semesters from 4-semester history  
**Metric:** R² on median quantile (q50), GPA scale 0–4

## Results

| Model               | R² (overall) | Sem+1  | Sem+2  | Sem+3  |
|---------------------|-------------|--------|--------|--------|
| LSTM + Attention    | 0.9595      | 0.9781 | 0.9593 | 0.9418 |
| Transformer Encoder | 0.9602      | 0.9782 | 0.9599 | 0.9431 |
| Prophet (cohort avg)| -0.7836     | N/A    | N/A    | N/A    |

## Key Observations

- **LSTM and Transformer are essentially tied** (~96% R²) — difference is <0.001 overall
- Both degrade gracefully as horizon increases (Sem+1 → Sem+3): expected, further future = more uncertainty
- **Prophet completely fails** (R²=-0.7836) — negative R² means it performs worse than simply predicting the mean; it cannot model per-student features
- **LSTM chosen for production** despite near-identical R²: Bahdanau attention provides interpretability (which past semesters drove the prediction), and the sequential inductive bias suits short sequences (T=4)
