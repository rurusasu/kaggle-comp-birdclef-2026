import csv
import json
from datetime import UTC, datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.config import Config


def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition metric. Replace with competition-specific metric."""
    return accuracy_score(y_true, y_pred)


def get_cv_splitter(cfg: Config):
    """Return CV splitter. Replace with StratifiedKFold/GroupKFold as needed."""
    return KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)


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
