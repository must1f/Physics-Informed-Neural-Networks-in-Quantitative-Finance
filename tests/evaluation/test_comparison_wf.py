import pandas as pd
import pytest
from src.evaluation.comparison import compare_walk_forward
from src.training.result import WalkForwardResult


def _make_wf_result(model_name, sharpe_mean, sharpe_std, rmse_mean):
    return WalkForwardResult(
        model_name=model_name,
        fold_metrics=[{"sharpe": sharpe_mean, "rmse": rmse_mean}],
        mean_metrics={"sharpe": sharpe_mean, "rmse": rmse_mean,
                      "directional_accuracy": 0.54, "mae": 0.008,
                      "r_squared": 0.01, "sortino": 1.1,
                      "max_drawdown": 0.15, "calmar": 0.5},
        std_metrics={"sharpe": sharpe_std, "rmse": 0.001,
                     "directional_accuracy": 0.02, "mae": 0.001,
                     "r_squared": 0.005, "sortino": 0.1,
                     "max_drawdown": 0.02, "calmar": 0.05},
        test_years=[2018],
        seeds=[42],
    )


def test_compare_walk_forward_returns_dataframe():
    results = {
        "gbm_pinn": _make_wf_result("gbm_pinn", 1.2, 0.1, 0.01),
        "ou_pinn":  _make_wf_result("ou_pinn",  0.9, 0.2, 0.012),
    }
    df = compare_walk_forward(results)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_compare_walk_forward_has_mean_std_columns():
    results = {"gbm_pinn": _make_wf_result("gbm_pinn", 1.2, 0.1, 0.01)}
    df = compare_walk_forward(results)
    assert "sharpe_mean" in df.columns
    assert "sharpe_std" in df.columns
    assert "rmse_mean" in df.columns


def test_compare_walk_forward_sorted_by_sharpe_descending():
    results = {
        "gbm_pinn": _make_wf_result("gbm_pinn", 1.2, 0.1, 0.01),
        "ou_pinn":  _make_wf_result("ou_pinn",  0.9, 0.2, 0.012),
    }
    df = compare_walk_forward(results)
    assert df.iloc[0]["model"] == "gbm_pinn"


def test_compare_walk_forward_raises_on_empty():
    with pytest.raises(ValueError):
        compare_walk_forward({})
