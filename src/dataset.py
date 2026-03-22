import pandas as pd

from src.config import Config


def load_train(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / "train.csv"
    return pd.read_csv(path)


def load_test(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_dir / "test.csv"
    return pd.read_csv(path)
