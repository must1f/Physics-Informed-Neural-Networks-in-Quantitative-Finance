"""Smoke tests for src.artefacts — import checks and plot round-trips."""
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace


def test_imports():
    from src.artefacts.plotting import (
        setup_theme, CB, _save, _annotate,
        _plot_loss_curve, _plot_pred_vs_actual, _plot_residuals,
        _plot_rolling_error, _plot_directional_confusion, _plot_equity_curve,
        _plot_physics_breakdown, _plot_physics_ratio, _plot_lambda_schedule,
    )
    from src.artefacts.orchestration import (
        emit_model_artefacts, emit_wf_artefacts, train_one,
    )


def test_setup_theme_populates_cb():
    from src.artefacts.plotting import setup_theme, CB
    setup_theme()
    assert len(CB) >= 6


def test_plot_residuals_creates_files(tmp_path):
    from src.artefacts.plotting import setup_theme, _plot_residuals
    setup_theme()
    resid = np.random.default_rng(0).normal(0, 0.01, 120)
    _plot_residuals(resid, "test_model", tmp_path)
    assert (tmp_path / "residuals.png").exists()
    assert (tmp_path / "residuals.pdf").exists()


def test_plot_equity_curve_creates_files(tmp_path):
    from src.artefacts.plotting import setup_theme, _plot_equity_curve
    setup_theme()
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0005, 0.01, 120)
    ns = SimpleNamespace(
        equity_curve=np.cumprod(1 + np.sign(rets) * rets),
        buy_hold_curve=np.cumprod(1 + rets),
        test_metrics={"sharpe": 0.5, "sortino": 0.6, "max_drawdown": -0.1, "calmar": 1.0},
    )
    _plot_equity_curve(ns, "test_model", tmp_path)
    assert (tmp_path / "equity_curve.png").exists()
    assert (tmp_path / "equity_curve.pdf").exists()


def test_plot_rolling_error_creates_files(tmp_path):
    from src.artefacts.plotting import setup_theme, _plot_rolling_error
    setup_theme()
    rng = np.random.default_rng(2)
    pred   = rng.normal(0, 0.01, 150)
    actual = rng.normal(0, 0.01, 150)
    _plot_rolling_error(pred, actual, "test_model", tmp_path)
    assert (tmp_path / "rolling_error.png").exists()


def test_plot_directional_confusion_creates_files(tmp_path):
    from src.artefacts.plotting import setup_theme, _plot_directional_confusion
    setup_theme()
    rng = np.random.default_rng(3)
    pred   = rng.normal(0, 0.01, 200)
    actual = rng.normal(0, 0.01, 200)
    _plot_directional_confusion(pred, actual, "test_model", tmp_path)
    assert (tmp_path / "directional_confusion.png").exists()
