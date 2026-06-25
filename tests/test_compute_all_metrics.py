"""Tests the compute_all_metrics aggregator returns the expected keys."""
import numpy as np

from src.evaluation.metrics import compute_all_metrics


def test_compute_all_metrics_keys(noisy_prediction):
    pred, actual = noisy_prediction
    returns = actual
    result = compute_all_metrics(pred, actual, returns)
    expected_keys = {
        "rmse", "mae", "r_squared", "directional_accuracy", "mape",
        "sharpe", "sortino", "max_drawdown", "calmar", "information_ratio",
    }
    assert set(result) == expected_keys
    for v in result.values():
        assert isinstance(v, float)
        assert np.isfinite(v)
