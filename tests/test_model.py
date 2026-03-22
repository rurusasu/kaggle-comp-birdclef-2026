from src.model import load_model, save_model


def test_save_and_load_model_roundtrip(tmp_path):
    dummy_model = {"type": "dummy", "params": [1, 2, 3]}
    path = tmp_path / "model.pkl"
    save_model(dummy_model, path)
    loaded = load_model(path)
    assert loaded == dummy_model


def test_save_model_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "model.pkl"
    save_model({"x": 1}, path)
    assert path.exists()
