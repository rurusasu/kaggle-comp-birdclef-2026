"""Submission generation for BirdCLEF+ 2026.

Creates submission.csv with row_id + one probability column per species.
Each row is a 5-second audio window prediction.
"""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config


def create_submission(
    cfg: Config,
    row_ids: list[str],
    predictions: np.ndarray,
    species_list: list[str],
) -> Path:
    """Create submission CSV in the competition format.

    Args:
        cfg: Config object.
        row_ids: List of row_id strings for each 5-second window.
        predictions: Array of shape (n_windows, n_species) with probabilities.
        species_list: Ordered list of species column names.

    Returns:
        Path to saved submission file.
    """
    cfg.submissions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = cfg.submissions_dir / f"submission_{timestamp}.csv"

    df = pd.DataFrame(predictions, columns=species_list)
    df.insert(0, "row_id", row_ids)
    df.to_csv(path, index=False)
    print(f"Submission saved: {path} ({len(df)} rows, {len(species_list)} species)")
    return path


def create_kaggle_submission(
    cfg: Config,
    row_ids: list[str],
    predictions: np.ndarray,
    species_list: list[str],
) -> Path:
    """Create the final submission.csv for Kaggle notebook submission."""
    path = Path("submission.csv")
    df = pd.DataFrame(predictions, columns=species_list)
    df.insert(0, "row_id", row_ids)
    df.to_csv(path, index=False)
    print(f"Kaggle submission saved: {path} ({len(df)} rows)")
    return path
