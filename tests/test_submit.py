import pandas as pd

from src.config import Config
from src.submit import create_submission


def test_create_submission(tmp_path):
    cfg = Config(output_dir=tmp_path / "outputs")
    ids = [1, 2, 3]
    predictions = [0.1, 0.5, 0.9]
    path = create_submission(cfg, ids, predictions)
    assert path.exists()
    assert path.parent == cfg.submissions_dir
    df = pd.read_csv(path)
    assert list(df.columns) == ["id", "target"]
    assert len(df) == 3


def test_create_submission_filename_has_timestamp(tmp_path):
    cfg = Config(output_dir=tmp_path / "outputs")
    path = create_submission(cfg, [1], [0.5])
    assert "submission_" in path.name
    assert path.suffix == ".csv"
