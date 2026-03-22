import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> object:
    """Train a model. Replace with competition-specific model.

    Returns a trained model object.
    """
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])
    return model


def predict(model: object, X: pd.DataFrame) -> np.ndarray:
    """Generate predictions from a trained model."""
    return model.predict(X)


def save_model(model: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)
