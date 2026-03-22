from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    competition_name: str = "default"
    seed: int = 42
    n_folds: int = 5

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.logs_dir = Path(self.logs_dir)

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def submissions_dir(self) -> Path:
        return self.output_dir / "submissions"

    @property
    def oof_dir(self) -> Path:
        return self.output_dir / "oof"
