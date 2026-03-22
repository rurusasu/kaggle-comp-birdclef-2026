# Next Task: BirdCLEF+ 2026

## Status

- 音声データは `data/processed/train_audio/` に展開済み（15GB）
- EfficientNet-B0 ベースラインコードは `src/` に実装済み
- ローカル GPU (RTX 4060 8GB) で学習可能
- Kaggle Notebook（推論用）は `kaggle-notebook/notebook.py` に作成済み

## Step 1: ローカル GPU 学習の実行

zip 読み込みから展開済みディレクトリ読み込みに切り替えて学習する。

1. `src/dataset.py` を確認。`read_audio_from_zip()` ではなく展開済みディレクトリからの読み込み関数を使うように `scripts/train.py` を修正。
2. 以下で短いテスト実行:
```bash
uv run python scripts/train.py --n-folds 1 --epochs 2 --batch-size 32
```
3. 動作確認後、本格学習:
```bash
uv run python scripts/train.py --n-folds 3 --epochs 10 --batch-size 32
```

### 注意点
- `pin_memory=True` は CUDA がない場合は `False` にする
- 展開されたファイルは `data/processed/train_audio/` または `data/raw/train_audio/` にある。パスを確認。
- `train.csv` の `filename` カラムは `{species_id}/{file}.ogg` 形式

## Step 2: 学習済みモデルを Kaggle にアップロード

```bash
# Dataset 作成
mkdir -p kaggle-dataset/birdclef-2026-models
cp outputs/models/*.pt kaggle-dataset/birdclef-2026-models/
cd kaggle-dataset
# dataset-metadata.json を作成して push
uv run kaggle datasets create -p .
```

## Step 3: 推論 Notebook を更新して Submit

1. `kaggle-notebook/notebook.py` のモデルパスを更新
2. `kaggle-notebook/kernel-metadata.json` の `dataset_sources` にアップロードした Dataset を追加
3. Push して Submit:
```bash
cd kaggle-notebook && uv run kaggle kernels push -p .
```

## Step 4: 改善案

1. **Focal Loss**: クラス不均衡対策。BCEWithLogitsLoss → FocalLoss に変更。
2. **Mixup / SpecAugment**: データ拡張を強化。
3. **SED (Sound Event Detection)**: 5秒ウィンドウ内の時間方向のアテンションを追加。
4. **Pretrained BirdNET**: 鳥の音声に特化した事前学習モデルを backbone に使用。
5. **EfficientNet-B2/B3**: より大きなモデル（CPU 90分制限に注意）。
6. **Soundscape labels の活用**: `train_soundscapes_labels.csv` からマルチラベルの学習データを作成。

## Important Notes

- **CPU 90分制限**: Kaggle 推論は CPU のみ。GPU Notebook は 1 分制限で実質使えない。
- **234 target species** のうち 28 種は train_audio に存在しない。taxonomy.csv と soundscape labels で補完。
- **マルチラベル**: BCEWithLogitsLoss が適切。
- **docs/approach.md** にアプローチの詳細あり。
