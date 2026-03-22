from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    competition_name: str = "birdclef-2026"
    seed: int = 42
    n_folds: int = 5

    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    logs_dir: Path = Path("logs")

    # Audio settings
    sample_rate: int = 32000
    audio_duration: float = 5.0  # seconds per chunk
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    fmin: int = 50
    fmax: int = 14000

    # Model settings
    model_name: str = "tf_efficientnet_b0_ns"
    pretrained: bool = True
    num_classes: int = 234  # species in taxonomy
    in_channels: int = 1  # mono mel spectrogram

    # Training settings
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    mixup_alpha: float = 0.5
    label_smoothing: float = 0.01

    # Species labels (populated at runtime)
    species_list: list[str] = field(default_factory=list)

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

    @property
    def zip_path(self) -> Path:
        return self.raw_dir / "birdclef-2026.zip"

    @property
    def audio_length_samples(self) -> int:
        return int(self.sample_rate * self.audio_duration)
