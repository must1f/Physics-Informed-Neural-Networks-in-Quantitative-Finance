"""Tests for compare_models."""
import pandas as pd
import pytest

from src.evaluation.comparison import compare_models
from src.training.result import EpochMetrics, TrainingResult


def _make_result(name, val_loss, metrics):
    r = TrainingResult(model_name=name, ticker="AAPL")
    r.history = [EpochMetrics(epoch=0, train_loss=0.1, val_loss=val_loss)]
    r.test_metrics = metrics
    return r


def test_compare_models_columns_and_sort():
    results = {
        "lstm": _make_result("lstm", 0.05, {
            "rmse": 0.02, "mae": 0.015, "r_squared": 0.4,
            "directional_accuracy": 0.55, "sharpe": 0.8,
            "sortino": 1.0, "max_drawdown": 0.2, "calmar": 1.1,
            "mape": 0.03,
        }),
        "gbm_pinn": _make_result("gbm_pinn", 0.03, {
            "rmse": 0.018, "mae": 0.014, "r_squared": 0.5,
            "directional_accuracy": 0.58, "sharpe": 1.5,
            "sortino": 1.8, "max_drawdown": 0.15, "calmar": 1.6,
            "mape": 0.025,
        }),
    }
    df = compare_models(results)
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "model", "rmse", "mae", "r_squared", "directional_accuracy",
        "sharpe", "sortino", "max_drawdown", "calmar",
        "best_val_loss", "best_epoch",
    }
    assert expected_cols.issubset(df.columns)
    assert df.iloc[0]["model"] == "gbm_pinn"


def test_compare_models_empty_raises():
    with pytest.raises(ValueError):
        compare_models({})
