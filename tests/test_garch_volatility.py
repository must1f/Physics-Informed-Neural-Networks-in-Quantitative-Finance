"""Tests for GARCHModel.predict_volatility (1-step-ahead conditional σ)."""
from __future__ import annotations

import numpy as np
import pytest

from src.models.classical import GARCHModel


@pytest.fixture
def fitted_model():
    rng = np.random.default_rng(0)
    # Simulate a conditionally heteroskedastic series so the fit is non-trivial.
    n = 2000
    r = np.zeros(n)
    sigma2 = np.full(n, 1e-4)
    omega, alpha, beta = 1e-6, 0.08, 0.90
    for t in range(1, n):
        sigma2[t] = omega + alpha * r[t-1]**2 + beta * sigma2[t-1]
        r[t] = rng.normal(0.0, np.sqrt(sigma2[t]))
    m = GARCHModel()
    m.fit(r[:1500])
    return m, r[1500:]  # train on first 1500, predict on last 500


def test_predict_volatility_shape(fitted_model):
    """predict_volatility returns an array the same length as test_returns."""
    m, test_r = fitted_model
    sigma = m.predict_volatility(test_r, last_train_return=0.0)
    assert sigma.shape == test_r.shape


def test_predict_volatility_positive(fitted_model):
    """Every σ_t forecast must be strictly positive (no NaN, no zero)."""
    m, test_r = fitted_model
    sigma = m.predict_volatility(test_r, last_train_return=0.0)
    assert np.all(sigma > 0.0)
    assert not np.any(np.isnan(sigma))


def test_predict_volatility_tracks_known_clustering(fitted_model):
    """σ_t should rise after a large |r_{t-1}| — the definitional GARCH behaviour."""
    m, test_r = fitted_model
    if not m._fit_ok:
        pytest.skip("arch unavailable — fallback path cannot track clustering")
    sigma = m.predict_volatility(test_r, last_train_return=0.0)
    # Look at indices 0..498 so that index+1 is valid.
    abs_r_lag = np.abs(test_r[:-1])
    sigma_next = sigma[1:]  # σ at step t+1 uses r_t
    big = np.argsort(abs_r_lag)[-20:]
    small = np.argsort(abs_r_lag)[:20]
    assert sigma_next[big].mean() > sigma_next[small].mean()


def test_predict_volatility_fallback_without_arch():
    """When _fit_ok=False (arch unavailable) falls back to unconditional σ."""
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, size=1000)
    m = GARCHModel()
    # Force fallback path without invoking arch at all.
    m._fit_ok = False
    m._mean = float(r[:500].mean())
    m._train_var = float(np.var(r[:500], ddof=1))
    sigma = m.predict_volatility(r[500:], last_train_return=float(r[499]))
    expected = float(np.sqrt(m._train_var))
    assert np.allclose(sigma, expected, rtol=1e-12)
    assert sigma.shape == (500,)
