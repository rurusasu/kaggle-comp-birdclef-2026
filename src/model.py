"""Model definition for BirdCLEF+ 2026.

Uses a pretrained EfficientNet backbone with a custom classification head
for multilabel species identification from mel spectrograms.
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class BirdCLEFModel(nn.Module):
    """CNN model for bird species classification from mel spectrograms.

    Architecture:
        - Pretrained EfficientNet-B0 backbone (timm)
        - Global average + max pooling
        - Dropout + linear classification head
    """

    def __init__(self, num_classes: int = 234, model_name: str = "tf_efficientnet_b0_ns", pretrained: bool = True):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,  # mono mel spectrogram
            num_classes=0,  # remove default head
            global_pool="",
        )

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy = torch.randn(1, 1, 128, 313)  # (B, C, n_mels, time)
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
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, n_mels, time_frames).

        Returns:
            Logits of shape (batch, num_classes).
        """
        features = self.backbone(x)
        avg_pool = self.global_avg_pool(features).flatten(1)
        max_pool = self.global_max_pool(features).flatten(1)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        return self.head(pooled)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    mixup_alpha: float = 0.0,
) -> float:
    """Train model for one epoch. Returns average loss."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        x = batch["melspec"].to(device)
        y = batch["target"].to(device)

        # Mixup augmentation
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            idx = torch.randperm(x.size(0), device=device)
            x = lam * x + (1 - lam) * x[idx]
            y = lam * y + (1 - lam) * y[idx]

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Validate model. Returns (loss, predictions, targets)."""
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        x = batch["melspec"].to(device)
        y = batch["target"].to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(y.cpu().numpy())

    avg_loss = total_loss / max(len(dataloader), 1)
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return avg_loss, preds, targets


@torch.no_grad()
def predict(model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> np.ndarray:
    """Generate predictions from a trained model."""
    model.eval()
    all_preds = []

    for batch in dataloader:
        x = batch["melspec"].to(device)
        logits = model(x)
        all_preds.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def save_model(model: nn.Module, path: Path) -> None:
    """Save model weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, num_classes: int = 234, model_name: str = "tf_efficientnet_b0_ns") -> nn.Module:
    """Load model weights."""
    model = BirdCLEFModel(num_classes=num_classes, model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model
