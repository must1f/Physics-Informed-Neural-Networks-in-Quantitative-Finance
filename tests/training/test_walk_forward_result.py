from src.training.result import WalkForwardResult


def test_walk_forward_result_to_dict():
    result = WalkForwardResult(
        model_name="gbm_pinn",
        fold_metrics=[{"rmse": 0.01, "sharpe": 1.2}, {"rmse": 0.02, "sharpe": 0.9}],
        mean_metrics={"rmse": 0.015, "sharpe": 1.05},
        std_metrics={"rmse": 0.005, "sharpe": 0.15},
        test_years=[2018, 2019],
        seeds=[42, 123],
    )
    d = result.to_dict()
    assert d["model_name"] == "gbm_pinn"
    assert d["mean_metrics"]["sharpe"] == 1.05
    assert len(d["fold_metrics"]) == 2


def test_walk_forward_result_fold_count():
    result = WalkForwardResult(
        model_name="ou_pinn",
        fold_metrics=[{"rmse": 0.01}] * 6,
        mean_metrics={"rmse": 0.01},
        std_metrics={"rmse": 0.0},
        test_years=[2018, 2019, 2020, 2021, 2022, 2023],
        seeds=[42, 123, 456],
    )
    assert result.n_folds == 6
