import sys
sys.path.insert(0, 'learning_analytics_xai')
sys.path.insert(0, 'learning_analytics_xai/forecasting')
from pathlib import Path
from gpa_forecaster import GPAForecaster
import numpy as np

model = GPAForecaster.load(Path("learning_analytics_xai/models/gpa_forecaster.pt"))
print(f"Model loaded — val losses: {len(model.val_losses)} epochs, best={min(model.val_losses):.5f}")

def predict(label, gpas, sem=6, att=82, fin=70, loads=None):
    if loads is None:
        loads = [18.0] * 4
    static_x = np.array([0.0, 0.0, att/100, fin/100, 0.05, 0.55,
                          (sum(loads) + 27) / 160.0],   # ~27 credits before window
                         dtype=np.float32)
    temporal_x = np.zeros((4, 5), dtype=np.float32)
    running_credits = 27.0
    for i, (g, lo) in enumerate(zip(gpas, loads)):
        running_credits += lo
        temporal_x[i, 0] = g / 4.0
        temporal_x[i, 1] = lo / 24.0
        temporal_x[i, 2] = 1.0 if g < 2.0 else 0.5 if g < 2.5 else 0.25 if g < 3.0 else 0.0
        temporal_x[i, 3] = (gpas[i] - gpas[i-1]) / 4.0 if i > 0 else 0.0
        temporal_x[i, 4] = running_credits / 160.0

    pred = model.predict(static_x, temporal_x)
    sems = pred['gpa_median']
    d01 = sems[1] - sems[0]
    d12 = sems[2] - sems[1]
    ci0 = pred['gpa_q90'][0] - pred['gpa_q10'][0]
    print(f"\n{label}")
    print(f"  History:  {' -> '.join(f'{g:.2f}' for g in gpas)}")
    print(f"  Forecast: {sems[0]:.3f} -> {sems[1]:.3f} -> {sems[2]:.3f}")
    print(f"  Deltas:   d1={d01:+.3f}  d2={d12:+.3f}  |  CI-width={ci0:.2f}")
    print(f"  CI:       [{pred['gpa_q10'][0]:.2f}-{pred['gpa_q90'][0]:.2f}]  "
          f"[{pred['gpa_q10'][1]:.2f}-{pred['gpa_q90'][1]:.2f}]  "
          f"[{pred['gpa_q10'][2]:.2f}-{pred['gpa_q90'][2]:.2f}]")

predict("User screenshot case",      [2.60, 2.65, 2.50, 2.55])
predict("Improving student",         [2.20, 2.45, 2.70, 2.90])
predict("Declining student",         [2.60, 2.35, 2.10, 1.90])
predict("Volatile student",          [2.35, 2.80, 2.40, 2.75])
predict("At-risk student",           [1.80, 1.70, 1.65, 1.55])
predict("High achiever",             [3.40, 3.45, 3.38, 3.42])
# Credit inertia demo: same GPA history, different semesters
predict("Sem 4 student (low inertia)",  [2.55, 2.50, 2.55, 2.60], sem=4,
        loads=[12.0, 15.0, 18.0, 18.0])
predict("Sem 8 student (high inertia)", [2.55, 2.50, 2.55, 2.60], sem=8,
        loads=[21.0, 21.0, 21.0, 18.0])
