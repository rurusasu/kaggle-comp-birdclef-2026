# BirdCLEF+ 2026 - Acoustic Species Identification

## Competition Info

- **URL:** https://www.kaggle.com/competitions/birdclef-2026
- **Deadline:** 2026-06-03 23:59 UTC
- **Prize:** $50,000 + Working Note Bonus $5,000
- **Category:** Research
- **Organizer:** Cornell Lab of Ornithology, TU Chemnitz, Google DeepMind

## Task

南米パンタナール地域の音響録音データから鳥の種を識別する。150,000+ km² の湿地帯に設置された1,000以上の音響レコーダーからのデータを使用。650以上の鳥種が対象。

## Evaluation

- **Metric:** Modified ROC-AUC（マルチラベル分類）
- True-positive ラベルがないクラスはスキップ
- 確率値とロジット値の両方を受け付け（[0,1]外の値にはシグモイドを自動適用）

## Submission Format

- CSV: `row_id` + 各鳥種ごとのカラム（確率値）
- 各行は5秒間の音声ウィンドウに対応
- 約10,932種の予測

## Data

- 音声フォーマット: 32,000 Hz
- 5秒チャンクに分割して予測
- フィールドレコーダーからのリアルワールド音声データ

## Special Rules

- マルチラベル分類（1つの音声セグメントに複数種が存在可能）
- Code Competition（Kaggle Notebook ベース）

## Documentation

**IMPORTANT: Before starting any implementation work, you MUST read the relevant docs first.**

- [docs/overview.md](docs/overview.md) — Competition description, goal, background
- [docs/evaluation.md](docs/evaluation.md) — Evaluation metric, scoring methodology
- [docs/submission.md](docs/submission.md) — Submission format, file structure, requirements
- [docs/timeline.md](docs/timeline.md) — Important dates and deadlines
- [docs/rules.md](docs/rules.md) — Full competition rules
- [docs/prizes.md](docs/prizes.md) — Prize structure

### Required Reading Order

1. Before EDA or feature engineering → read `overview.md` and `evaluation.md`
2. Before building submission pipeline → read `submission.md`
3. Before using external data or models → read `rules.md`
4. Before final submission → read `timeline.md` to confirm deadlines

---

# Kaggle Competition Workspace

## Structure

- `src/config.py` — All configuration (paths, params, seed). Change settings HERE, not in other modules.
- `src/dataset.py` — Stateless data I/O. `load_train()` / `load_test()` return raw DataFrames.
- `src/features.py` — Feature engineering. Stateful transforms (fit on train only).
- `src/model.py` — Model train/predict/save/load.
- `src/evaluate.py` — CV splitter, metrics, experiment logging. Owns all writes to `logs/`.
- `src/submit.py` — Generates timestamped submission CSVs.
- `src/utils.py` — `set_seed()`, `Timer`.
- `scripts/train.py` — Training entrypoint. Runs full CV pipeline.
- `scripts/predict.py` — Inference entrypoint. Loads saved models, generates submission.

## Conventions

- Format with ruff (line-length=120, Python 3.14)
- Type hints encouraged
- Config changes go in `src/config.py` only
- Experiment logs go in `logs/` via `src/evaluate.py` only

## Commands

- `task setup` — Install deps + download data
- `task train` — Train models
- `task predict` — Generate predictions
- `task submit` — Submit to Kaggle
- `task lint` — Check code style
- `task test` — Run tests

## Current Approach

### Workflow

1. **Local training:** Run `uv run python scripts/train.py` to train 5-fold CV EfficientNet-B0 models. Saves best weights per fold to `outputs/models/model_fold{0..4}.pt`.
2. **Upload models:** Upload `outputs/models/` as a Kaggle Dataset named `birdclef-2026-models` (e.g., `kaggle datasets create -p outputs/models`).
3. **Kaggle inference:** Submit `kaggle-notebook/notebook.py` as a Kaggle Notebook. It loads models from `/kaggle/input/birdclef-2026-models/`, processes test soundscapes, and writes `submission.csv`.

### File Layout

```
src/
  config.py          — Audio params, model config (EfficientNet-B0, 234 classes)
  dataset.py         — Zip-based data loading, audio reading
  features.py        — Mel spectrogram pipeline
  model.py           — BirdCLEFModel, train/predict functions
  evaluate.py        — ROC-AUC metric, CV splitter, experiment logging
  submit.py          — Submission CSV generation
  utils.py           — set_seed(), Timer
scripts/
  train.py           — Local CV training loop
  predict.py         — Local inference pipeline
kaggle-notebook/
  notebook.py        — Self-contained Kaggle inference script (CPU only)
  kernel-metadata.json — Kaggle notebook metadata
docs/
  approach.md        — Detailed approach documentation
```

### How to Train Locally

```bash
# Install dependencies
task setup

# Train with defaults (5 folds, 30 epochs, EfficientNet-B0)
uv run python scripts/train.py

# Custom training
uv run python scripts/train.py --seed 0 --n-folds 3 --epochs 20 --batch-size 16 --lr 5e-4
```

Models are saved to `outputs/models/model_fold{N}.pt`. Training logs go to `logs/`.

### How to Submit via Kaggle

```bash
# 1. Upload trained models as a Kaggle dataset
cd outputs/models
kaggle datasets init -p .
# Edit dataset-metadata.json: set slug to "YOUR_USERNAME/birdclef-2026-models"
kaggle datasets create -p .

# 2. Push the notebook
cd kaggle-notebook
kaggle kernels push
```

The notebook reads models from `/kaggle/input/birdclef-2026-models/` and competition data from `/kaggle/input/birdclef-2026/`. It must complete within 90 minutes on CPU with no internet.

### Improvement Ideas

- **Backbone upgrades:** EfficientNet-B2/B3, ConvNeXt, EfficientNetV2 for better accuracy
- **BirdNET embeddings:** Use pretrained BirdNET as feature extractor
- **SED models:** Sound Event Detection with attention pooling (PANNs)
- **Data augmentations:** SpecAugment, time/freq masking, background noise mixing, cutmix
- **Pseudo-labeling:** Use soundscape predictions to generate pseudo-labels for unlabeled data
- **TTA (Test-Time Augmentation):** Average predictions over time-shifted windows
- **Threshold tuning:** Per-species thresholds optimized on validation set
- **Knowledge distillation:** Train smaller model from ensemble for faster CPU inference
