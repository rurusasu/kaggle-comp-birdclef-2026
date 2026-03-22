# kaggle-comp-{name}

Kaggle competition workspace.

## Setup

```bash
task setup
```

## Workflow

```bash
task train              # Train with CV
task predict            # Generate submission from trained models
task submit             # Submit to Kaggle
```

## Configuration

Edit `src/config.py` to set competition name, parameters, and paths.

## For DL competitions

Uncomment torch dependencies in `pyproject.toml`:

```bash
uv sync --dev --extra dl
```
