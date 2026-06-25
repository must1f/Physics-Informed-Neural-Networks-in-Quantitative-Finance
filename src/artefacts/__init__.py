"""Artefact layer — figure generation, per-model outputs, and walk-forward orchestration.

Re-exports the two public sub-modules:

* :mod:`~src.artefacts.plotting` — low-level plot builders (loss curves, equity
  curves, residual diagnostics, S&P 500 overlays, PINN physics plots).
* :mod:`~src.artefacts.orchestration` — high-level orchestrators that call the
  plot builders and flush all files to disk after training completes.

Call :func:`setup_theme` once per process before using any plot function.
"""
from .plotting import (
    setup_theme, CB,
    _save, _annotate,
    _plot_loss_curve,
    _plot_pred_vs_actual,
    _plot_residuals,
    _plot_rolling_error,
    _plot_directional_confusion,
    _plot_equity_curve,
    _plot_sp500_predictions,
    _plot_physics_breakdown,
    _plot_physics_ratio,
    _plot_lambda_schedule,
)
from .orchestration import (
    emit_model_artefacts,
    emit_wf_artefacts,
    train_one,
)

__all__ = [
    "setup_theme", "CB",
    "_save", "_annotate",
    "_plot_loss_curve", "_plot_pred_vs_actual", "_plot_residuals",
    "_plot_rolling_error", "_plot_directional_confusion", "_plot_equity_curve",
    "_plot_sp500_predictions", "_plot_physics_breakdown", "_plot_physics_ratio",
    "_plot_lambda_schedule",
    "emit_model_artefacts", "emit_wf_artefacts", "train_one",
]
