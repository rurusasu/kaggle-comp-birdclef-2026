"""Inference entrypoint. Loads saved models and generates predictions.

Usage:
    uv run python scripts/predict.py
    uv run python scripts/predict.py --model-dir outputs/models
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_test
from src.features import build_features
from src.model import load_model, predict
from src.submit import create_submission
from src.utils import Timer, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    set_seed(cfg.seed)
    model_dir = Path(args.model_dir) if args.model_dir else cfg.models_dir

    with Timer("load data"):
        df = load_test(cfg)

    with Timer("build features"):
        df = build_features(df, cfg, is_train=False)

    feature_cols = [c for c in df.columns if c != "id"]

    # Ensemble predictions from all fold models
    model_paths = sorted(model_dir.glob("model_fold*.pkl"))
    if not model_paths:
        print(f"No models found in {model_dir}")
        sys.exit(1)

    all_preds = []
    for path in model_paths:
        model = load_model(path)
        preds = predict(model, df[feature_cols])
        all_preds.append(preds)

    ensemble_preds = np.mean(all_preds, axis=0)
    submission_path = create_submission(cfg, df["id"].tolist(), ensemble_preds.tolist())
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
