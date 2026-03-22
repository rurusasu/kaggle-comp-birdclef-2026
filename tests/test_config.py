from pathlib import Path

from src.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.competition_name == "default"
    assert cfg.seed == 42
    assert cfg.n_folds == 5


def test_config_paths_are_pathlib():
    cfg = Config()
    assert isinstance(cfg.data_dir, Path)
    assert isinstance(cfg.output_dir, Path)


def test_config_subdirs():
    cfg = Config()
    assert cfg.raw_dir == cfg.data_dir / "raw"
    assert cfg.models_dir == cfg.output_dir / "models"
    assert cfg.submissions_dir == cfg.output_dir / "submissions"
    assert cfg.oof_dir == cfg.output_dir / "oof"
    assert cfg.logs_dir == Path("logs")


def test_config_override():
    cfg = Config(competition_name="titanic", seed=0, n_folds=10)
    assert cfg.competition_name == "titanic"
    assert cfg.seed == 0
    assert cfg.n_folds == 10
