"""Tests for the sign-based backtester."""
import numpy as np
import pytest

from src.evaluation.backtester import SignBasedBacktester, BacktestResult


def test_no_cost_perfect_predictions_beats_buy_and_hold():
    actual = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
    bt = SignBasedBacktester(transaction_cost=0.0)
    res = bt.run(predictions=actual, actual_returns=actual)
    assert isinstance(res, BacktestResult)
    assert res.strategy_returns == pytest.approx(np.full(5, 0.01))
    assert res.cumulative_return == pytest.approx((1.01 ** 5) - 1)


def test_transaction_cost_reduces_pnl():
    actual = np.array([0.01, -0.01, 0.01, -0.01, 0.01])
    no_cost = SignBasedBacktester(0.0).run(actual, actual)
    with_cost = SignBasedBacktester(0.001).run(actual, actual)
    assert with_cost.cumulative_return < no_cost.cumulative_return


def test_positions_follow_prediction_sign():
    pred = np.array([0.02, -0.01, 0.0, 0.03])
    actual = np.array([0.01, -0.02, 0.01, 0.02])
    res = SignBasedBacktester(0.0).run(pred, actual)
    assert res.positions.tolist() == [1, -1, -1, 1]


def test_equity_curve_length_and_start():
    pred = np.array([0.01, -0.01, 0.02])
    res = SignBasedBacktester(0.0).run(pred, pred)
    assert len(res.equity_curve) == 3
    assert res.equity_curve[-1] == pytest.approx((1.01) * (1.01) * (1.02))


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        SignBasedBacktester().run(np.zeros(3), np.zeros(4))


def test_metrics_populated():
    rng = np.random.default_rng(0)
    actual = rng.standard_normal(500) * 0.01
    pred = actual + rng.standard_normal(500) * 0.005
    res = SignBasedBacktester(0.001).run(pred, actual)
    for name in ("sharpe", "sortino", "max_drawdown", "cumulative_return"):
        assert isinstance(getattr(res, name), float)
        assert np.isfinite(getattr(res, name))
