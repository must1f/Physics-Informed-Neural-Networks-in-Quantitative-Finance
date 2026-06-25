"""Regression: financial metrics must depend on predictions, not just actuals.

Pre-2026-04-18 the training/evaluation stack passed ``test_actual`` as the
``returns`` argument to :func:`compute_all_metrics`, so every model reported
the buy-and-hold Sharpe of the test window regardless of its forecasts. This
test locks that bug out by asserting that two different prediction vectors
evaluated against the same ``actual`` produce different Sharpe numbers when
the strategy-return convention (``sign(pred) * actual``) is used.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import compute_all_metrics


def _strategy_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return compute_all_metrics(pred, actual, np.sign(pred) * actual)


def test_different_predictions_give_different_sharpes():
    rng = np.random.default_rng(0)
    actual = rng.normal(0.0, 0.01, size=500)

    zero_forecast = np.zeros_like(actual)          # random-walk style
    oracle_forecast = actual.copy()                # perfect direction

    zero_metrics = _strategy_metrics(zero_forecast, actual)
    oracle_metrics = _strategy_metrics(oracle_forecast, actual)

    # Flat-forecast strategy never trades → Sharpe must be 0 under the
    # (std < 1e-12) guard in sharpe_ratio.
    assert zero_metrics["sharpe"] == 0.0
    # Oracle strategy must beat it — if these are equal we've regressed to the
    # buy-and-hold bug.
    assert oracle_metrics["sharpe"] > zero_metrics["sharpe"]
    assert oracle_metrics["sharpe"] != zero_metrics["sharpe"]


def test_sortino_zero_series_returns_zero():
    """A flat (all-zero) strategy must report Sortino=0, not -sqrt(252).

    Pre-fix, sortino_ratio on zero returns produced excess = -rf/252 (all
    negative), downside_dev = rf/252, and Sortino = -sqrt(252) ≈ -15.87 —
    a degenerate artefact. The zero-series guard now matches sharpe_ratio's
    std-guard and returns 0.0 for the no-trade case.
    """
    from src.evaluation.metrics import sortino_ratio
    flat = np.zeros(500)
    assert sortino_ratio(flat) == 0.0


def test_buy_and_hold_convention_still_works():
    """Passing ``actual`` as ``returns`` is still a valid (buy-and-hold) call."""
    rng = np.random.default_rng(1)
    actual = rng.normal(0.0, 0.01, size=500)
    pred_a = rng.normal(0.0, 0.01, size=500)
    pred_b = rng.normal(0.0, 0.01, size=500)

    bh_a = compute_all_metrics(pred_a, actual, actual)
    bh_b = compute_all_metrics(pred_b, actual, actual)

    # Under buy-and-hold, Sharpe is a property of ``actual`` only, so it must
    # match across predictions. This documents the historical behaviour —
    # callers who want strategy Sharpe MUST pass strategy returns.
    assert bh_a["sharpe"] == bh_b["sharpe"]
