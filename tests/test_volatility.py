"""Tests for volatility-forecast evaluation helpers."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.volatility import realised_vol_parkinson
from src.evaluation.volatility import realised_vol_squared_returns


def test_parkinson_scalar_case():
    """Parkinson estimator: σ² = (ln H − ln L)² / (4 ln 2).

    For H=110, L=90: ln(110/90) ≈ 0.20067, σ² ≈ 0.01452, σ ≈ 0.12052.
    """
    high = np.array([110.0])
    low = np.array([90.0])
    sigma = realised_vol_parkinson(high, low)
    assert sigma.shape == (1,)
    assert np.isclose(sigma[0], 0.12051503, rtol=1e-6)


def test_parkinson_flat_bar_is_zero():
    """High == Low (no intraday range) → zero estimated vol, not NaN."""
    high = np.array([100.0, 100.0, 100.0])
    low = high.copy()
    sigma = realised_vol_parkinson(high, low)
    assert np.allclose(sigma, 0.0)


def test_parkinson_rejects_bad_inputs():
    """Bad shapes, non-1-D arrays, and low > high all raise ValueError."""
    with pytest.raises(ValueError):
        realised_vol_parkinson(np.zeros(5), np.zeros(6))
    with pytest.raises(ValueError):
        realised_vol_parkinson(np.zeros((2, 3)), np.zeros((2, 3)))
    # low > high is an arbitrage-impossible bar
    with pytest.raises(ValueError):
        realised_vol_parkinson(np.array([100.0]), np.array([110.0]))


def test_parkinson_rejects_non_positive_prices():
    """Zero or negative prices raise ValueError — log would otherwise silently produce -inf/nan."""
    with pytest.raises(ValueError, match="non-positive"):
        realised_vol_parkinson(np.array([100.0, 0.0]), np.array([99.0, 0.0]))
    with pytest.raises(ValueError, match="non-positive"):
        realised_vol_parkinson(np.array([-1.0]), np.array([-2.0]))


def test_squared_returns_is_absolute_return():
    """Squared-returns estimator: σ_t = |r_t|. Unbiased but noisy.

    Since the proxy is E[r_t²]^(1/2) under iid zero-mean returns, the
    single-sample estimator is just |r_t| on the same scale as the
    forecaster outputs.
    """
    r = np.array([-0.02, 0.015, 0.0, 0.03])
    sigma = realised_vol_squared_returns(r)
    assert np.allclose(sigma, np.abs(r))


def test_squared_returns_rejects_bad_inputs():
    """Non-1-D arrays raise ValueError."""
    with pytest.raises(ValueError):
        realised_vol_squared_returns(np.zeros((2, 3)))


from src.evaluation.volatility import rolling_realised_vol


def test_rolling_rv_22day_constant_returns():
    """Constant-magnitude returns → σ̂_t = |r| for every t past warm-up."""
    r = np.full(100, 0.01)
    sigma = rolling_realised_vol(r, window=22)
    assert sigma.shape == (100,)
    # First 22 entries are NaN (warm-up); thereafter all equal 0.01.
    assert np.all(np.isnan(sigma[:22]))
    assert np.allclose(sigma[22:], 0.01)


def test_rolling_rv_window_too_large():
    """Window >= len(returns) raises ValueError."""
    with pytest.raises(ValueError):
        rolling_realised_vol(np.zeros(10), window=20)


def test_rolling_rv_window_must_be_positive():
    """Zero or negative window raises ValueError."""
    with pytest.raises(ValueError):
        rolling_realised_vol(np.zeros(10), window=0)


from src.evaluation.volatility import qlike_loss


def test_qlike_perfect_forecast_matches_log_term():
    """QLIKE = log(σ²_f) + σ̂²/σ²_f. When σ̂==σ_f, QLIKE = log(σ²_f) + 1."""
    forecast = np.full(100, 0.01)
    proxy = forecast.copy()
    loss = qlike_loss(forecast, proxy)
    expected = np.log(0.01 ** 2) + 1.0
    assert np.isclose(loss, expected, rtol=1e-10)


def test_qlike_penalises_underprediction_more_than_overprediction():
    """Asymmetric loss: halving σ_f hurts more than doubling (Patton 2011)."""
    proxy = np.full(500, 0.01)
    under = qlike_loss(np.full(500, 0.005), proxy)
    over  = qlike_loss(np.full(500, 0.020), proxy)
    perfect = qlike_loss(proxy, proxy)
    assert under > perfect and over > perfect
    assert under > over


def test_qlike_rejects_zero_forecast():
    """Zero forecast is undefined under QLIKE (log(0) = -inf)."""
    with pytest.raises(ValueError):
        qlike_loss(np.zeros(10), np.ones(10))


from src.evaluation.volatility import mincer_zarnowitz


def test_mz_perfect_forecast_gives_alpha0_beta1():
    """Perfect forecast: regress proxy=α+β·forecast → α=0, β=1, R²=1."""
    rng = np.random.default_rng(0)
    forecast = rng.uniform(0.005, 0.03, size=500)
    proxy = forecast.copy()
    res = mincer_zarnowitz(forecast, proxy)
    assert np.isclose(res["alpha"], 0.0, atol=1e-9)
    assert np.isclose(res["beta"], 1.0, atol=1e-9)
    assert np.isclose(res["r_squared"], 1.0, atol=1e-9)
    # Joint Wald test H0: α=0, β=1 → should NOT reject.
    # For a perfectly deterministic relationship the F-test may be
    # degenerate (zero-variance residual); accept NaN or p > 0.99.
    assert np.isnan(res["joint_p"]) or res["joint_p"] > 0.99


def test_mz_biased_forecast_rejects_joint_null():
    """Forecast = 0.5·proxy (in σ-space) → forecast² = 0.25·proxy² → regression recovers β=4, joint test rejects."""
    rng = np.random.default_rng(1)
    proxy = rng.uniform(0.005, 0.03, size=500)
    forecast = 0.5 * proxy
    res = mincer_zarnowitz(forecast, proxy)
    # Note: MZ regresses proxy² on forecast², so a 0.5× scale in σ-space
    # is a 0.25× scale in σ²-space → slope = 1/0.25 = 4.
    assert np.isclose(res["beta"], 4.0, atol=1e-9)
    assert res["joint_p"] < 1e-3


def test_mz_rejects_mismatched_shapes():
    """Shape mismatch raises ValueError."""
    with pytest.raises(ValueError):
        mincer_zarnowitz(np.zeros(5), np.zeros(6))


def test_mz_rejects_too_few_rows():
    """< 10 valid rows after NaN drop raises ValueError."""
    with pytest.raises(ValueError):
        mincer_zarnowitz(np.arange(1, 6, dtype=float), np.arange(1, 6, dtype=float))


from src.evaluation.volatility import evaluate_volatility_forecast


def test_evaluate_volatility_forecast_end_to_end():
    """End-to-end: feed forecast + proxy, get a flat dict with all metrics."""
    rng = np.random.default_rng(2)
    proxy = np.abs(rng.normal(0.0, 0.01, size=500))
    # A forecast that's 1.1× the proxy — biased but correlated.
    forecast = 1.1 * proxy + rng.normal(0.0, 1e-4, size=500).clip(min=1e-6)
    out = evaluate_volatility_forecast(forecast, proxy)
    assert set(out) >= {"qlike", "mz_alpha", "mz_beta", "mz_r_squared",
                         "mz_joint_p", "n"}
    assert np.isfinite(out["qlike"])
    assert 0.0 <= out["mz_r_squared"] <= 1.0
    assert out["n"] == 500
