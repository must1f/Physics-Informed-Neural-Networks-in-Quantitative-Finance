"""Verify the public surface of src.evaluation."""
import src.evaluation as E


def test_public_exports():
    for name in (
        "rmse", "mae", "r_squared", "directional_accuracy", "mape",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
        "compute_all_metrics",
        "SignBasedBacktester", "BacktestResult",
        "classify_metric",
        "compare_models", "plot_comparison",
        "evaluate_on_test",
    ):
        assert hasattr(E, name), f"missing public export: {name}"
