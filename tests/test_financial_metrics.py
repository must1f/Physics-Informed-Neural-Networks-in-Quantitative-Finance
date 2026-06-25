"""Tests for financial metrics — Sharpe, Sortino, MDD, Calmar."""
import numpy as np
import pytest

from src.evaluation.metrics import (
    sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio,
)


def test_sharpe_constant_positive_returns(constant_returns):
    assert sharpe_ratio(constant_returns) == pytest.approx(0.0)


def test_sharpe_known_value():
    returns = np.array([0.01, -0.005, 0.002, 0.007, -0.003])
    excess = returns - 0.02 / 252
    expected = (excess.mean() / excess.std(ddof=1)) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-6)


def test_sortino_only_penalises_downside():
    returns = np.array([0.01, 0.02, 0.03, -0.01])
    val = sortino_ratio(returns, rf=0.0)
    assert val > 0
    assert np.isfinite(val)


def test_sortino_no_negative_returns_returns_zero():
    returns = np.array([0.01, 0.02, 0.03])
    assert sortino_ratio(returns, rf=0.0) == pytest.approx(0.0)


def test_max_drawdown_monotone_up_is_zero():
    equity = np.array([1.0, 1.1, 1.2, 1.3])
    assert max_drawdown(equity) == pytest.approx(0.0)


def test_max_drawdown_known():
    equity = np.array([1.0, 1.5, 0.75, 1.2])
    assert max_drawdown(equity) == pytest.approx(0.5)


def test_calmar_positive_return_bounded_drawdown():
    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01] * 50)
    val = calmar_ratio(returns)
    assert np.isfinite(val)
