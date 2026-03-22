import csv
import json

import numpy as np

from src.config import Config
from src.evaluate import log_experiment, metric_fn


def test_metric_fn_perfect_score():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])
    score = metric_fn(y_true, y_pred)
    assert score == 1.0


def test_metric_fn_worst_score():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([1, 0, 0, 1])
    score = metric_fn(y_true, y_pred)
    assert score == 0.0


def test_log_experiment_creates_json(tmp_path):
    cfg = Config(logs_dir=tmp_path / "logs")
    cfg.logs_dir.mkdir(parents=True)
    result = {
        "experiment": "test_run",
        "fold_scores": [0.8, 0.85, 0.82],
        "mean_score": 0.823,
    }
    log_experiment(cfg, result)
    log_files = list(cfg.logs_dir.glob("*.json"))
    assert len(log_files) == 1
    data = json.loads(log_files[0].read_text())
    assert data["experiment"] == "test_run"
    assert data["mean_score"] == 0.823


def test_log_experiment_appends_csv(tmp_path):
    cfg = Config(logs_dir=tmp_path / "logs")
    cfg.logs_dir.mkdir(parents=True)
    log_experiment(cfg, {"experiment": "run1", "mean_score": 0.8})
    log_experiment(cfg, {"experiment": "run2", "mean_score": 0.9})
    csv_path = cfg.logs_dir / "experiments.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["experiment"] == "run1"
    assert rows[1]["experiment"] == "run2"
