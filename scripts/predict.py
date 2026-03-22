"""Inference entrypoint for BirdCLEF+ 2026.

Loads saved models and generates submission CSV from test soundscapes.
Processes each soundscape file in 5-second windows.

Usage:
    uv run python scripts/predict.py
    uv run python scripts/predict.py --model-dir outputs/models
"""

import argparse
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_sample_submission, load_species_list
from src.features import build_features
from src.model import BirdCLEFModel, load_model, predict
from src.submit import create_submission
from src.utils import Timer, set_seed


class TestSoundscapeDataset(Dataset):
    """Dataset for test soundscape inference.

    Splits each soundscape into 5-second windows and generates mel spectrograms.
    """

    def __init__(self, row_ids: list[str], audio_chunks: list[np.ndarray], cfg: Config):
        self.row_ids = row_ids
        self.audio_chunks = audio_chunks
        self.cfg = cfg

    def __len__(self):
        return len(self.row_ids)

    def __getitem__(self, idx):
        audio = self.audio_chunks[idx]
        melspec = build_features(audio, self.cfg, is_train=False)
        return {
            "melspec": torch.tensor(melspec, dtype=torch.float32),
            "row_id": self.row_ids[idx],
        }


def load_test_soundscapes(cfg: Config) -> tuple[list[str], list[np.ndarray]]:
    """Load test soundscapes and split into 5-second chunks.

    Returns:
        Tuple of (row_ids, audio_chunks).
    """
    import soundfile as sf

    row_ids = []
    audio_chunks = []
    chunk_samples = cfg.audio_length_samples

    with zipfile.ZipFile(cfg.zip_path, "r") as z:
        test_files = sorted([n for n in z.namelist() if n.startswith("test_soundscapes/") and n.endswith(".ogg")])

        for filepath in test_files:
            basename = Path(filepath).stem
            with z.open(filepath) as f:
                audio_bytes = BytesIO(f.read())
            audio, sr = sf.read(audio_bytes)

            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != cfg.sample_rate:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=cfg.sample_rate)

            # Split into 5-second windows
            for start in range(0, len(audio), chunk_samples):
                end_sec = (start + chunk_samples) // cfg.sample_rate * 5
                if end_sec == 0:
                    end_sec = 5
                row_id = f"{basename}_{end_sec}"
                chunk = audio[start : start + chunk_samples]

                # Pad last chunk if needed
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode="constant")

                row_ids.append(row_id)
                audio_chunks.append(chunk.astype(np.float32))

    return row_ids, audio_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(seed=args.seed)
    set_seed(cfg.seed)
    model_dir = Path(args.model_dir) if args.model_dir else cfg.models_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    species_list = load_species_list(cfg)
    print(f"Species: {len(species_list)}")

    # Load test soundscapes
    with Timer("load test soundscapes"):
        row_ids, audio_chunks = load_test_soundscapes(cfg)
        print(f"Test windows: {len(row_ids)}")

    if not row_ids:
        # Fallback: use sample_submission row_ids with uniform predictions
        print("No test soundscapes found, using sample_submission format")
        sub = load_sample_submission(cfg)
        row_ids = sub["row_id"].tolist()
        predictions = np.full((len(row_ids), len(species_list)), 1.0 / len(species_list))
    else:
        test_dataset = TestSoundscapeDataset(row_ids, audio_chunks, cfg)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)

        # Load and ensemble all fold models
        model_paths = sorted(model_dir.glob("model_fold*.pt"))
        if not model_paths:
            print(f"No models found in {model_dir}, using uniform predictions")
            predictions = np.full((len(row_ids), len(species_list)), 1.0 / len(species_list))
        else:
            all_preds = []
            for path in model_paths:
                print(f"Loading {path}")
                model = load_model(path, num_classes=len(species_list), model_name=cfg.model_name)
                model = model.to(device)
                preds = predict(model, test_loader, device)
                all_preds.append(preds)

            predictions = np.mean(all_preds, axis=0)

    submission_path = create_submission(cfg, row_ids, predictions, species_list)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
