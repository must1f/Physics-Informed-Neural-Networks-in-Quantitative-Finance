"""Tests for prediction metrics on known inputs."""
import numpy as np
import pytest

from src.evaluation.metrics import (
    rmse, mae, r_squared, directional_accuracy, mape,
)


def test_rmse_perfect(perfect_prediction):
    pred, actual = perfect_prediction
    assert rmse(pred, actual) == pytest.approx(0.0, abs=1e-12)


def test_rmse_known_value():
    pred = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.0, 2.0, 5.0])
    assert rmse(pred, actual) == pytest.approx(np.sqrt(4 / 3))


def test_mae_known_value():
    pred = np.array([1.0, 2.0, 3.0])
    actual = np.array([1.0, 2.0, 5.0])
    assert mae(pred, actual) == pytest.approx(2 / 3)


def test_r_squared_perfect(perfect_prediction):
    pred, actual = perfect_prediction
    assert r_squared(pred, actual) == pytest.approx(1.0)


def test_r_squared_constant_pred_is_zero():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.full_like(actual, actual.mean())
    assert r_squared(pred, actual) == pytest.approx(0.0)


def test_directional_accuracy_all_correct():
    pred = np.array([0.01, -0.02, 0.03, -0.04])
    actual = np.array([0.02, -0.01, 0.01, -0.05])
    assert directional_accuracy(pred, actual) == pytest.approx(1.0)


def test_directional_accuracy_half():
    pred = np.array([0.01, 0.01, -0.01, -0.01])
    actual = np.array([0.01, -0.01, 0.01, -0.01])
    assert directional_accuracy(pred, actual) == pytest.approx(0.5)


def test_mape_known():
    pred = np.array([110.0, 95.0])
    actual = np.array([100.0, 100.0])
    assert mape(pred, actual) == pytest.approx(0.075)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        rmse(np.zeros(3), np.zeros(4))
