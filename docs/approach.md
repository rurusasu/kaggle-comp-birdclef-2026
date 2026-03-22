# BirdCLEF+ 2026 - Baseline Approach

## Data Summary

- **Location:** Pantanal wetlands, Brazil (-16.5 to -21.6 lat, -55.9 to -57.6 lon)
- **Audio format:** .ogg, 32kHz sample rate
- **Train audio:** 35,549 individual recordings across 206 species (in `train_audio/`)
- **Train soundscapes:** 10,657 continuous field recordings (66 unique sites, in `train_soundscapes/`)
- **Soundscape labels:** 1,478 annotated 5-second windows with multilabel species (`train_soundscapes_labels.csv`)
- **Taxonomy:** 234 species total (162 Aves, 35 Amphibia, 28 Insecta, 8 Mammalia, 1 Reptilia)
- **Test:** Hidden test soundscapes, each split into 5-second windows
- **Sources:** Xeno-canto (23,043) and iNaturalist (12,506)
- **Class imbalance:** Heavy - top species have ~500 recordings, bottom species have 1-3

## Task

Multilabel classification: for each 5-second audio window, predict probability of each of the 234 species being present.

## Evaluation

- Macro-averaged ROC-AUC, skipping classes with no true positives
- Probabilities or logits accepted (sigmoid auto-applied if values outside [0,1])

## Data Pipeline

```
Raw audio (.ogg, 32kHz)
    -> Load with soundfile/librosa
    -> Pad or trim to 5 seconds (160,000 samples)
    -> Mel spectrogram (n_fft=2048, hop=512, 128 mels, 50-14000 Hz)
    -> Log scale (power_to_db)
    -> Normalize (zero mean, unit variance)
    -> Shape: (1, 128, 313)  # (channels, mels, time_frames)
```

## Model Architecture

### Baseline: EfficientNet-B0

- Pretrained `tf_efficientnet_b0_ns` backbone via timm
- Single-channel input (mono mel spectrogram)
- Global average + max pooling concatenated
- Head: Linear(feat*2, 512) -> BN -> ReLU -> Dropout(0.3) -> Linear(512, 234)
- Output: 234 logits -> BCE with logits loss

### Future improvements

- **Backbone upgrades:** EfficientNet-B2/B3, ConvNeXt, EfficientNetV2
- **BirdNET embeddings:** Use pretrained BirdNET as feature extractor
- **SED models:** Sound Event Detection with attention pooling (e.g., PANNs)
- **Multi-resolution:** Combine spectrograms at different time/frequency resolutions

## Training Strategy

### Baseline
- 5-fold StratifiedKFold on primary_label
- AdamW optimizer (lr=1e-3, weight_decay=1e-4)
- Cosine annealing LR schedule
- BCEWithLogitsLoss
- Mixup augmentation (alpha=0.5)
- 30 epochs per fold, save best by validation ROC-AUC

### Augmentations (planned)
- **Audio-level:** Time shift, gain noise, background noise mixing
- **Spectrogram-level:** SpecAugment (time/frequency masking), mixup, cutmix
- **Label smoothing:** 0.01 (configured)

### Data strategy
- Primary training: individual recordings from `train_audio/`
- Validation: use soundscape labels for realistic evaluation
- Semi-supervised: use soundscape predictions to generate pseudo-labels

## Inference Pipeline

```
Test soundscape (.ogg)
    -> Split into 5-second windows
    -> Mel spectrogram per window (same as training)
    -> Model inference (ensemble of K fold models)
    -> Average predictions across folds
    -> Output: row_id + 234 species probabilities
```

## Submission Format

```csv
row_id,1161364,116570,1176823,...,yehcar1,yeofly1
BC2026_Test_0001_S05_20250227_010002_5,0.004,0.001,...
BC2026_Test_0001_S05_20250227_010002_10,0.002,0.003,...
```

- row_id: `{soundscape_basename}_{end_second}`
- 234 species probability columns
- Each row = one 5-second window

## Constraints

- CPU only (90 min runtime limit)
- No internet access during inference
- GPU submissions disabled (1 min limit)
- Pre-trained models allowed (publicly available)

## Key Challenges

1. **Severe class imbalance:** 1-500 samples per species
2. **Domain gap:** Training on individual recordings, testing on continuous soundscapes
3. **Multilabel:** Multiple species can vocalize simultaneously
4. **CPU-only inference:** Must be efficient (no GPU)
5. **Non-bird species:** Includes amphibians, insects, mammals, reptiles
