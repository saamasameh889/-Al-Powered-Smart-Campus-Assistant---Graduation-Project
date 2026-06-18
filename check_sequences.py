import sys
sys.path.insert(0, 'learning_analytics_xai')
sys.path.insert(0, 'learning_analytics_xai/forecasting')

from sequence_generator import generate_sequences, STATIC_DIM, TEMPORAL_DIM, _trajectory_params
import pandas as pd
import numpy as np

print(f'TEMPORAL_DIM = {TEMPORAL_DIM}  (expected 4)')

df = pd.read_csv('learning_analytics_xai/data/students_summary.csv')
row = df.iloc[0]
params = _trajectory_params(row)
print(f'First student trajectory params: trend={params[0]:.3f}, noise={params[1]:.3f}, pattern={params[2]}')

data = generate_sequences(n_augmentations=1, history_len=4, horizon=3, seed=42)
print(f'temporal shape: {data["temporal"].shape}  (expected (N, 4, 4))')

targets = data['target']
deltas = np.abs(targets[:, 1] - targets[:, 0])
print(f'Avg target GPA delta sem1->sem2 (normalised): {deltas.mean():.4f}  (was ~0.002, want ~0.04-0.10)')
print(f'Max target GPA delta: {deltas.max():.4f}')
print(f'Std of target GPA delta: {deltas.std():.4f}')

temporal = data['temporal']
print(f'\nSample temporal[0] (4 steps x 4 features):')
print(temporal[0])
print(f'Feature 3 (gpa_delta) values: {temporal[0, :, 3]}')
