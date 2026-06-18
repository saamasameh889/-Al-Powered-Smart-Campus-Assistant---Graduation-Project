import sys
sys.path.insert(0, 'learning_analytics_xai')
sys.path.insert(0, 'learning_analytics_xai/forecasting')
from pathlib import Path
from gpa_forecaster import GPAForecaster
import numpy as np

model = GPAForecaster.load(Path("learning_analytics_xai/models/gpa_forecaster.pt"))
print("Model loaded OK")
print(f"Best val loss: {min(model.val_losses):.5f}  epochs trained: {len(model.val_losses)}")

def test(label, gpas, trend_desc):
    static_x = np.array([0.0, 0.0, 0.82, 0.70, 0.05, 0.55], dtype=np.float32)
    temporal_x = np.zeros((4, 4), dtype=np.float32)
    for i, g in enumerate(gpas):
        temporal_x[i, 0] = g / 4.0
        temporal_x[i, 1] = 18.0 / 24.0
        temporal_x[i, 2] = 1.0 if g < 2.0 else 0.5 if g < 2.5 else 0.25 if g < 3.0 else 0.0
        temporal_x[i, 3] = (gpas[i] - gpas[i-1]) / 4.0 if i > 0 else 0.0
    pred = model.predict(static_x, temporal_x)
    sems = pred['gpa_median']
    delta01 = sems[1] - sems[0]
    delta12 = sems[2] - sems[1]
    print(f"\n{label} ({trend_desc})")
    print(f"  History:  {' -> '.join(f'{g:.2f}' for g in gpas)}")
    print(f"  Forecast: {sems[0]:.3f} -> {sems[1]:.3f} -> {sems[2]:.3f}")
    print(f"  Deltas:   sem1-sem2={delta01:+.3f}  sem2-sem3={delta12:+.3f}")
    print(f"  CI:       [{pred['gpa_q10'][0]:.2f}-{pred['gpa_q90'][0]:.2f}]  "
          f"[{pred['gpa_q10'][1]:.2f}-{pred['gpa_q90'][1]:.2f}]  "
          f"[{pred['gpa_q10'][2]:.2f}-{pred['gpa_q90'][2]:.2f}]")

test("DECLINING student", [2.60, 2.35, 2.10, 1.90], "clear downtrend")
test("IMPROVING student", [2.20, 2.45, 2.70, 2.90], "clear uptrend")
test("VOLATILE student",  [2.35, 2.80, 2.40, 2.75], "zigzag pattern")
test("STABLE student",    [3.40, 3.45, 3.38, 3.42], "flat high achiever")
test("AT-RISK student",   [1.80, 1.70, 1.65, 1.55], "at-risk decline")
