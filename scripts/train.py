"""Training entrypoint.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --seed 0 --n-folds 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_train
from src.evaluate import get_cv_splitter, log_experiment, metric_fn
from src.features import build_features
from src.model import predict, save_model, train
from src.utils import Timer, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    cfg = Config(seed=args.seed, n_folds=args.n_folds)
    set_seed(cfg.seed)

    with Timer("load data"):
        df = load_train(cfg)

    with Timer("build features"):
        df = build_features(df, cfg, is_train=True)

    target_col = "target"
    feature_cols = [c for c in df.columns if c not in ["id", target_col]]
    X = df[feature_cols]
    y = df[target_col].values

    splitter = get_cv_splitter(cfg)
    fold_scores = []
    oof_preds = np.zeros(len(df))

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y)):
        with Timer(f"fold {fold}"):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            model = train(X_train, y_train, X_valid, y_valid)
            preds = predict(model, X_valid)
            oof_preds[valid_idx] = preds

            score = metric_fn(y_valid, preds)
            fold_scores.append(score)
            print(f"Fold {fold}: {score:.4f}")

            save_model(model, cfg.models_dir / f"model_fold{fold}.pkl")

    mean_score = np.mean(fold_scores)
    print(f"\nCV Mean: {mean_score:.4f} (+/- {np.std(fold_scores):.4f})")

    # Save OOF predictions
    cfg.oof_dir.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame({"id": df["id"], "oof_pred": oof_preds})
    oof_df.to_csv(cfg.oof_dir / "oof_predictions.csv", index=False)

    log_experiment(
        cfg,
        {
            "experiment": f"seed{cfg.seed}_folds{cfg.n_folds}",
            "seed": cfg.seed,
            "n_folds": cfg.n_folds,
            "fold_scores": fold_scores,
            "mean_score": float(mean_score),
        },
    )


if __name__ == "__main__":
    main()
