"""Evaluation for BirdCLEF+ 2026.

Metric: macro-averaged ROC-AUC that skips classes with no true positive labels.
This is the official competition metric.
"""

import csv
import json
from datetime import UTC, datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import Config


def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition metric: macro-averaged ROC-AUC, skipping classes with no positives.

    Args:
        y_true: Binary targets of shape (n_samples, n_classes).
        y_pred: Predicted probabilities of shape (n_samples, n_classes).

    Returns:
        Macro-averaged ROC-AUC score.
    """
    aucs = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        # Skip classes with no positive labels
        if col_true.sum() == 0:
            continue
        # Skip classes with only positive labels (no negatives)
        if col_true.sum() == len(col_true):
            continue
        try:
            auc = roc_auc_score(col_true, y_pred[:, i])
            aucs.append(auc)
        except ValueError:
            continue

    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def get_cv_splitter(cfg: Config):
    """Return stratified CV splitter.

    For multilabel, we stratify on the primary label to ensure balanced folds.
    """
    return StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)


def log_experiment(cfg: Config, result: dict) -> None:
    """Save experiment result as JSON and append to CSV in logs/."""
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    result["timestamp"] = timestamp

    # JSON (detailed, per-experiment)
    json_path = cfg.logs_dir / f"{timestamp}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str))

    # CSV (summary, append-only)
    csv_path = cfg.logs_dir / "experiments.csv"
    flat = {k: str(v) if isinstance(v, list) else v for k, v in result.items()}
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat)
