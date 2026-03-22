"""Data loading for BirdCLEF+ 2026.

Handles reading metadata from the competition zip and loading audio data.
Audio files are .ogg format at 32kHz. Train data includes:
  - train_audio/: per-species folders of individual recordings (35,549 files, 206 species)
  - train_soundscapes/: continuous field recordings (10,657 files, 66 unique)
  - train_soundscapes_labels.csv: multilabel annotations per 5s window
  - taxonomy.csv: full species list (234 species across Aves, Amphibia, Insecta, Mammalia, Reptilia)
"""

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config


def load_taxonomy(cfg: Config) -> pd.DataFrame:
    """Load taxonomy.csv from the competition zip."""
    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        with z.open("taxonomy.csv") as f:
            return pd.read_csv(f)


def load_species_list(cfg: Config) -> list[str]:
    """Load ordered species list from sample_submission.csv columns."""
    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        with z.open("sample_submission.csv") as f:
            sub = pd.read_csv(f, nrows=0)
    return [c for c in sub.columns if c != "row_id"]


def load_train(cfg: Config) -> pd.DataFrame:
    """Load train.csv metadata from the competition zip."""
    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        with z.open("train.csv") as f:
            return pd.read_csv(f)


def load_soundscape_labels(cfg: Config) -> pd.DataFrame:
    """Load train_soundscapes_labels.csv from the competition zip.

    Each row has: filename, start, end, primary_label (semicolon-separated species).
    """
    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        with z.open("train_soundscapes_labels.csv") as f:
            return pd.read_csv(f)


def load_sample_submission(cfg: Config) -> pd.DataFrame:
    """Load sample_submission.csv from the competition zip."""
    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        with z.open("sample_submission.csv") as f:
            return pd.read_csv(f)


def load_test(cfg: Config) -> pd.DataFrame:
    """Load test data (sample_submission structure for inference)."""
    return load_sample_submission(cfg)


def read_audio_from_zip(cfg: Config, filepath: str) -> np.ndarray:
    """Read an audio file from the competition zip and return as numpy array.

    Args:
        cfg: Config object.
        filepath: Path within the zip, e.g. 'train_audio/22930/XC123456.ogg'.

    Returns:
        Audio waveform as float32 numpy array, resampled to cfg.sample_rate.
    """
    import soundfile as sf

    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        with z.open(filepath) as f:
            audio_bytes = BytesIO(f.read())
    audio, sr = sf.read(audio_bytes)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != cfg.sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.sample_rate)

    return audio.astype(np.float32)


def pad_or_trim(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Pad (with zeros) or trim audio to exact target_length samples."""
    if len(audio) > target_length:
        # Random offset for training augmentation
        max_start = len(audio) - target_length
        start = np.random.randint(0, max_start + 1)
        return audio[start : start + target_length]
    elif len(audio) < target_length:
        pad_width = target_length - len(audio)
        return np.pad(audio, (0, pad_width), mode="constant")
    return audio


def extract_audio_to_dir(cfg: Config, target_dir: Path | None = None) -> Path:
    """Extract train_audio from zip to a local directory for faster access.

    Only extracts if not already done. Returns the extraction directory.
    """
    if target_dir is None:
        target_dir = cfg.processed_dir / "train_audio"

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Audio already extracted to {target_dir}")
        return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting train_audio to {target_dir}...")

    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        audio_files = [n for n in z.namelist() if n.startswith("train_audio/")]
        for name in audio_files:
            z.extract(name, target_dir.parent.parent)

    print(f"Extracted {len(audio_files)} files")
    return cfg.processed_dir.parent / "raw" / "train_audio"
