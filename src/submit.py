from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import Config


def create_submission(
    cfg: Config,
    ids: Sequence,
    predictions: Sequence,
    id_col: str = "id",
    target_col: str = "target",
) -> Path:
    """Create submission CSV. Adjust id_col/target_col per competition."""
    cfg.submissions_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    path = cfg.submissions_dir / f"submission_{timestamp}.csv"
    df = pd.DataFrame({id_col: ids, target_col: predictions})
    df.to_csv(path, index=False)
    return path
