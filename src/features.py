import pandas as pd

from src.config import Config


def build_features(df: pd.DataFrame, cfg: Config, is_train: bool = True) -> pd.DataFrame:
    """Build features from raw DataFrame.

    Replace this with competition-specific feature engineering.
    Stateful transforms (encoders, scalers) should be fit only when is_train=True.
    """
    return df.copy()
