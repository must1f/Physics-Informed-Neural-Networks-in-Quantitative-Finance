"""Smoke test for plot_comparison — renders to a file."""
import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from src.evaluation.comparison import plot_comparison
from src.training.result import EpochMetrics, TrainingResult


def _mk(name, val_losses, metrics):
    r = TrainingResult(model_name=name, ticker="AAPL")
    r.history = [
        EpochMetrics(epoch=i, train_loss=v + 0.01, val_loss=v)
        for i, v in enumerate(val_losses)
    ]
    r.test_metrics = metrics
    return r


def test_plot_comparison_writes_file(tmp_path: Path):
    results = {
        "lstm": _mk("lstm", [0.1, 0.08, 0.07], {
            "rmse": 0.02, "sharpe": 0.8, "max_drawdown": 0.2,
            "directional_accuracy": 0.55,
        }),
        "gbm_pinn": _mk("gbm_pinn", [0.09, 0.06, 0.05], {
            "rmse": 0.018, "sharpe": 1.5, "max_drawdown": 0.15,
            "directional_accuracy": 0.58,
        }),
    }
    out = tmp_path / "comparison.png"
    plot_comparison(results, save_path=out)
    assert out.exists()
    assert out.stat().st_size > 1000
