import pandas as pd

from src.config import Config
from src.features import build_features


def test_build_features_returns_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cfg = Config()
    result = build_features(df, cfg, is_train=True)
    assert isinstance(result, pd.DataFrame)


def test_build_features_preserves_shape():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    cfg = Config()
    result = build_features(df, cfg, is_train=True)
    assert result.shape == df.shape


def test_build_features_does_not_mutate_input():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    original = df.copy()
    cfg = Config()
    build_features(df, cfg, is_train=True)
    pd.testing.assert_frame_equal(df, original)
