"""Feature engineering for BirdCLEF+ 2026.

Converts raw audio waveforms into mel spectrograms suitable for CNN input.
Pipeline: audio waveform -> mel spectrogram -> normalized log-mel -> tensor
"""

import numpy as np

from src.config import Config


def audio_to_melspec(audio: np.ndarray, cfg: Config) -> np.ndarray:
    """Convert audio waveform to log-mel spectrogram.

    Args:
        audio: 1D float32 array of audio samples.
        cfg: Config with spectrogram parameters.

    Returns:
        2D numpy array of shape (n_mels, time_frames), log-scaled.
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    # Convert to log scale (dB), clamp to avoid log(0)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def normalize_melspec(melspec: np.ndarray) -> np.ndarray:
    """Normalize a mel spectrogram to zero mean and unit variance."""
    mean = melspec.mean()
    std = melspec.std()
    if std < 1e-6:
        return melspec - mean
    return (melspec - mean) / std


def build_features(audio: np.ndarray, cfg: Config, is_train: bool = True) -> np.ndarray:
    """Full feature pipeline: audio -> normalized mel spectrogram.

    Args:
        audio: 1D float32 array (already padded/trimmed to cfg.audio_length_samples).
        cfg: Config.
        is_train: Whether this is training data (for potential augmentations).

    Returns:
        Normalized log-mel spectrogram as float32 array of shape (1, n_mels, time_frames).
    """
    melspec = audio_to_melspec(audio, cfg)
    melspec = normalize_melspec(melspec)
    # Add channel dimension: (1, n_mels, time_frames)
    return melspec[np.newaxis, ...]
