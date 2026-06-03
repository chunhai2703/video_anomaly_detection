import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.threshold import apply_threshold
from src.evaluation.evaluator import evaluate_model

# Ground Truth
y_true = np.array([
    0,0,0,1,1,1,0,1,0,1
])

# Anomaly Scores từ model
scores = np.array([
    0.10,
    0.20,
    0.15,
    0.90,
    0.85,
    0.70,
    0.25,
    0.95,
    0.30,
    0.88
])

# Threshold
threshold = 0.5

# Chuyển score -> prediction
y_pred = apply_threshold(
    scores,
    threshold
)

# Evaluation
results = evaluate_model(
    y_true,
    y_pred,
    scores
)

print(results)