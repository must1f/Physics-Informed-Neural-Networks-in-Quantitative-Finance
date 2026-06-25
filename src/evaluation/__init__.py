"""Evaluation layer — metrics, backtester, forecast tests, and model comparison.

Sub-modules:

* :mod:`~src.evaluation.metrics` — pure-function prediction and financial metrics
  (RMSE, MAE, R², DA, MAPE, Sharpe, Sortino, MDD, Calmar, IR).
* :mod:`~src.evaluation.evaluator` — :func:`~src.evaluation.evaluator.evaluate_on_test`
  runs a trained model over a held-out dataset and returns metrics + raw arrays.
* :mod:`~src.evaluation.backtester` — :class:`~src.evaluation.backtester.SignBasedBacktester`
  simulates a long/short strategy with transaction costs.
* :mod:`~src.evaluation.benchmarks` — :func:`~src.evaluation.benchmarks.classify_metric`
  maps raw metric values to qualitative bands against literature thresholds.
* :mod:`~src.evaluation.comparison` — :func:`~src.evaluation.comparison.compare_models`
  and :func:`~src.evaluation.comparison.compare_walk_forward` build ranked summary tables.
* :mod:`~src.evaluation.forecast_tests` — Diebold–Mariano and Pesaran–Timmermann tests.
* :mod:`~src.evaluation.volatility` — QLIKE, Mincer–Zarnowitz, and rolling-RV helpers.
"""
from src.evaluation.metrics import (
    rmse,
    mae,
    r_squared,
    directional_accuracy,
    mape,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    compute_all_metrics,
)
from src.evaluation.backtester import SignBasedBacktester, BacktestResult
from src.evaluation.benchmarks import classify_metric
from src.evaluation.comparison import compare_models, plot_comparison
from src.evaluation.evaluator import evaluate_on_test

__all__ = [
    "rmse", "mae", "r_squared", "directional_accuracy", "mape",
    "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio",
    "compute_all_metrics",
    "SignBasedBacktester", "BacktestResult",
    "classify_metric",
    "compare_models", "plot_comparison",
    "evaluate_on_test",
]
