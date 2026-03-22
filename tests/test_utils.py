import numpy as np

from src.utils import Timer, set_seed


def test_set_seed_numpy_reproducibility():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_timer_measures_elapsed(capsys):
    import time

    with Timer("test"):
        time.sleep(0.1)
    captured = capsys.readouterr()
    assert "test" in captured.out
    assert "0." in captured.out
