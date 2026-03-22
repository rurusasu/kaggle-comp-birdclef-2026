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
