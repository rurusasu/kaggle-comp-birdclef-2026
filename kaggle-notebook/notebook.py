"""BirdCLEF+ 2026 Baseline Inference Notebook.

Self-contained Kaggle inference script for CPU-only submission.
Loads pretrained EfficientNet-B0 fold models, processes test soundscapes
in 5-second windows, and outputs submission.csv.

Expected Kaggle input layout:
  /kaggle/input/birdclef-2026/           <- competition data
  /kaggle/input/birdclef-2026-models/    <- uploaded model weights
"""

import os
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
COMPETITION_DIR = Path("/kaggle/input/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/birdclef-2026-models")
OUTPUT_PATH = Path("submission.csv")

# ---------------------------------------------------------------------------
# Config (must match training)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 32000
AUDIO_DURATION = 5.0  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 160000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 50
FMAX = 14000
NUM_CLASSES = 234
MODEL_NAME = "tf_efficientnet_b0_ns"
BATCH_SIZE = 32
SEED = 42


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Model (self-contained copy to avoid src/ dependency)
# ---------------------------------------------------------------------------
class BirdCLEFModel(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        model_name: str = MODEL_NAME,
        pretrained: bool = False,
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )

        with torch.no_grad():
            dummy = torch.randn(1, 1, N_MELS, 313)
            features = self.backbone(dummy)
            feat_dim = features.shape[1]

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        avg_pool = self.global_avg_pool(features).flatten(1)
        max_pool = self.global_max_pool(features).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.head(pooled)


def load_model(path: Path, num_classes: int = NUM_CLASSES) -> nn.Module:
    model = BirdCLEFModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature pipeline (self-contained copy)
# ---------------------------------------------------------------------------
def audio_to_melspec(audio: np.ndarray) -> np.ndarray:
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_melspec(melspec: np.ndarray) -> np.ndarray:
    mean = melspec.mean()
    std = melspec.std()
    if std < 1e-6:
        return melspec - mean
    return (melspec - mean) / std


def build_features(audio: np.ndarray) -> np.ndarray:
    melspec = audio_to_melspec(audio)
    melspec = normalize_melspec(melspec)
    return melspec[np.newaxis, ...]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestSoundscapeDataset(Dataset):
    def __init__(self, row_ids: list[str], audio_chunks: list[np.ndarray]):
        self.row_ids = row_ids
        self.audio_chunks = audio_chunks

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, idx):
        audio = self.audio_chunks[idx]
        melspec = build_features(audio)
        return {
            "melspec": torch.tensor(melspec, dtype=torch.float32),
            "row_id": self.row_ids[idx],
        }


# ---------------------------------------------------------------------------
# Soundscape loading
# ---------------------------------------------------------------------------
def load_test_soundscapes() -> tuple[list[str], list[np.ndarray]]:
    """Load test soundscapes and split into 5-second chunks."""
    import soundfile as sf

    test_dir = COMPETITION_DIR / "test_soundscapes"
    row_ids: list[str] = []
    audio_chunks: list[np.ndarray] = []

    if not test_dir.exists():
        print("test_soundscapes directory not found")
        return row_ids, audio_chunks

    test_files = sorted(test_dir.glob("*.ogg"))
    print(f"Found {len(test_files)} test soundscape files")

    for filepath in test_files:
        basename = filepath.stem
        audio, sr = sf.read(filepath)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Split into 5-second windows
        for start in range(0, len(audio), CHUNK_SAMPLES):
            end_sec = (start + CHUNK_SAMPLES) // SAMPLE_RATE * 5
            if end_sec == 0:
                end_sec = 5
            row_id = f"{basename}_{end_sec}"
            chunk = audio[start : start + CHUNK_SAMPLES]

            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode="constant")

            row_ids.append(row_id)
            audio_chunks.append(chunk.astype(np.float32))

    return row_ids, audio_chunks


def load_species_list() -> list[str]:
    """Load species list from sample_submission.csv."""
    sub = pd.read_csv(COMPETITION_DIR / "sample_submission.csv", nrows=0)
    return [c for c in sub.columns if c != "row_id"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    model.eval()
    all_preds = []
    for batch in dataloader:
        x = batch["melspec"]
        logits = model(x)
        all_preds.append(torch.sigmoid(logits).numpy())
    return np.concatenate(all_preds, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    set_seed(SEED)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Competition dir: {COMPETITION_DIR}")
    print(f"Model dir: {MODEL_DIR}")

    # Load species list
    species_list = load_species_list()
    num_classes = len(species_list)
    print(f"Species: {num_classes}")

    # Load test soundscapes
    print("Loading test soundscapes...")
    row_ids, audio_chunks = load_test_soundscapes()
    print(f"Test windows: {len(row_ids)}")

    if not row_ids:
        # Fallback: use sample_submission with uniform predictions
        print("No test soundscapes found, using sample_submission format")
        sub = pd.read_csv(COMPETITION_DIR / "sample_submission.csv")
        row_ids = sub["row_id"].tolist()
        predictions = np.full((len(row_ids), num_classes), 1.0 / num_classes)
    else:
        dataset = TestSoundscapeDataset(row_ids, audio_chunks)
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
        )

        # Load and ensemble fold models
        model_paths = sorted(MODEL_DIR.glob("model_fold*.pt"))
        if not model_paths:
            print(f"No models found in {MODEL_DIR}, using uniform predictions")
            predictions = np.full((len(row_ids), num_classes), 1.0 / num_classes)
        else:
            all_preds = []
            for path in model_paths:
                print(f"Loading model: {path.name}")
                model = load_model(path, num_classes=num_classes)
                preds = run_inference(model, dataloader)
                all_preds.append(preds)
                del model
                print(f"  -> predictions shape: {preds.shape}")

            predictions = np.mean(all_preds, axis=0)
            print(f"Ensemble of {len(all_preds)} models")

    # Create submission
    df = pd.DataFrame(predictions, columns=species_list)
    df.insert(0, "row_id", row_ids)
    df.to_csv(OUTPUT_PATH, index=False)

    elapsed = time.perf_counter() - t_start
    print(f"\nSubmission saved: {OUTPUT_PATH}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
