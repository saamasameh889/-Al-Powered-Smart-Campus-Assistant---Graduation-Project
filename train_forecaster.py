"""Retrain the GPA forecaster with the updated 4-feature temporal input."""
import sys
sys.path.insert(0, 'learning_analytics_xai')
sys.path.insert(0, 'learning_analytics_xai/forecasting')

from pathlib import Path
from sequence_generator import generate_sequences, STATIC_DIM, TEMPORAL_DIM
from gpa_forecaster import GPAForecaster
import numpy as np

MODEL_PATH = Path("learning_analytics_xai/models/gpa_forecaster.pt")
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"STATIC_DIM={STATIC_DIM}  TEMPORAL_DIM={TEMPORAL_DIM}")

print("Generating sequences...")
data = generate_sequences(n_augmentations=5, history_len=4, horizon=3, seed=42)
print(f"  sequences: {len(data['static']):,}")
print(f"  temporal shape: {data['temporal'].shape}")

targets = data['target']
deltas = np.abs(targets[:, 1] - targets[:, 0])
print(f"  avg GPA delta between horizon steps (normalised): {deltas.mean():.4f}")

print("\nBuilding model...")
forecaster = GPAForecaster(
    static_dim=STATIC_DIM,
    temporal_dim=TEMPORAL_DIM,
    lstm_hidden=128,
    lstm_layers=2,
    horizon=3,
    dropout=0.25,
)

epoch_log = []

def cb(epoch, total, train_loss, val_loss):
    epoch_log.append((epoch, train_loss, val_loss))
    if epoch % 5 == 0 or epoch == total:
        print(f"  epoch {epoch:3d}/{total}  train={train_loss:.4f}  val={val_loss:.4f}")

print("\nTraining...")
forecaster.fit(
    data,
    epochs=80,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    val_frac=0.20,
    patience=12,
    progress_cb=cb,
)

forecaster.save(MODEL_PATH)
print(f"\nSaved -> {MODEL_PATH}")
best_val = min(forecaster.val_losses)
best_ep  = int(np.argmin(forecaster.val_losses)) + 1
print(f"Best val loss: {best_val:.5f} at epoch {best_ep}")

# Quick inference smoke test
import numpy as np_
static_x  = np_.array([0.0, 0.0, 0.82, 0.70, 0.05, 0.55, 80.0/160.0], dtype=np_.float32)
temporal_x = np_.array([
    [2.60/4, 18/24, 0.25, 0.00,      45/160],
    [2.65/4, 18/24, 0.25, 0.05/4,    63/160],
    [2.50/4, 21/24, 0.50, -0.15/4,   84/160],
    [2.55/4, 21/24, 0.25, 0.05/4,   105/160],
], dtype=np_.float32)

pred = forecaster.predict(static_x, temporal_x)
print(f"\nSmoke test (GPA history: 2.60, 2.65, 2.50, 2.55):")
for i, (q10, q50, q90) in enumerate(zip(pred['gpa_q10'], pred['gpa_median'], pred['gpa_q90'])):
    print(f"  Sem +{i+1}: {q50:.3f}  [{q10:.3f} – {q90:.3f}]")

print("\nDone!")
