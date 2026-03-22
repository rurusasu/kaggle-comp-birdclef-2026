"""Training entrypoint for BirdCLEF+ 2026.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --seed 0 --n-folds 5 --epochs 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.dataset import load_species_list, load_train, pad_or_trim, read_audio_from_zip
from src.evaluate import get_cv_splitter, log_experiment, metric_fn
from src.features import build_features
from src.model import BirdCLEFModel, save_model, train_one_epoch, validate
from src.utils import Timer, set_seed


class BirdCLEFTrainDataset(Dataset):
    """PyTorch dataset for BirdCLEF training data."""

    def __init__(self, df, species_list: list[str], cfg: Config):
        self.df = df.reset_index(drop=True)
        self.species_list = species_list
        self.species_to_idx = {s: i for i, s in enumerate(species_list)}
        self.cfg = cfg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = f"train_audio/{row['filename']}"

        # Load and preprocess audio
        audio = read_audio_from_zip(self.cfg, filepath)
        audio = pad_or_trim(audio, self.cfg.audio_length_samples)
        melspec = build_features(audio, self.cfg, is_train=True)

        # Create multilabel target
        target = np.zeros(len(self.species_list), dtype=np.float32)
        primary = str(row["primary_label"])
        if primary in self.species_to_idx:
            target[self.species_to_idx[primary]] = 1.0

        # Handle secondary labels
        secondary = row.get("secondary_labels", "[]")
        if isinstance(secondary, str) and secondary != "[]":
            for label in secondary.strip("[]").replace("'", "").replace('"', "").split(","):
                label = label.strip()
                if label in self.species_to_idx:
                    target[self.species_to_idx[label]] = 1.0

        return {
            "melspec": torch.tensor(melspec, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        n_folds=args.n_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load metadata and species list
    with Timer("load metadata"):
        train_df = load_train(cfg)
        species_list = load_species_list(cfg)
        cfg.species_list = species_list
        print(f"Train samples: {len(train_df)}, Species: {len(species_list)}")

    # CV split on primary_label
    splitter = get_cv_splitter(cfg)
    fold_scores = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_df, train_df["primary_label"])):
        print(f"\n{'='*60}")
        print(f"Fold {fold}")
        print(f"{'='*60}")

        train_data = train_df.iloc[train_idx]
        valid_data = train_df.iloc[valid_idx]

        train_dataset = BirdCLEFTrainDataset(train_data, species_list, cfg)
        valid_dataset = BirdCLEFTrainDataset(valid_data, species_list, cfg)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = BirdCLEFModel(
            num_classes=cfg.num_classes,
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs * len(train_loader))

        best_score = 0.0
        for epoch in range(cfg.epochs):
            with Timer(f"epoch {epoch}"):
                train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, cfg.mixup_alpha)
                val_loss, val_preds, val_targets = validate(model, valid_loader, device)
                score = metric_fn(val_targets, val_preds)

                print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ROC-AUC={score:.4f}")

                if score > best_score:
                    best_score = score
                    save_model(model, cfg.models_dir / f"model_fold{fold}.pt")
                    print(f"  -> New best: {best_score:.4f}")

        fold_scores.append(best_score)
        print(f"Fold {fold} best ROC-AUC: {best_score:.4f}")

    mean_score = np.mean(fold_scores)
    print(f"\nCV Mean ROC-AUC: {mean_score:.4f} (+/- {np.std(fold_scores):.4f})")

    log_experiment(
        cfg,
        {
            "experiment": f"baseline_{cfg.model_name}",
            "seed": cfg.seed,
            "n_folds": cfg.n_folds,
            "epochs": cfg.epochs,
            "model_name": cfg.model_name,
            "fold_scores": fold_scores,
            "mean_score": float(mean_score),
        },
    )


if __name__ == "__main__":
    main()
