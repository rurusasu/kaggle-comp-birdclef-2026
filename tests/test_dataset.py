import pandas as pd
import pytest

from src.config import Config
from src.dataset import load_test, load_train


@pytest.fixture
def sample_data(tmp_path):
    cfg = Config(data_dir=tmp_path / "data")
    raw = cfg.raw_dir
    raw.mkdir(parents=True)
    train_df = pd.DataFrame({"id": [1, 2], "feature": [0.1, 0.2], "target": [0, 1]})
    test_df = pd.DataFrame({"id": [3, 4], "feature": [0.3, 0.4]})
    train_df.to_csv(raw / "train.csv", index=False)
    test_df.to_csv(raw / "test.csv", index=False)
    return cfg


def test_load_train(sample_data):
    df = load_train(sample_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "target" in df.columns


def test_load_test(sample_data):
    df = load_test(sample_data)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "target" not in df.columns


def test_load_train_missing_file():
    cfg = Config(data_dir="nonexistent")
    with pytest.raises(FileNotFoundError):
        load_train(cfg)
