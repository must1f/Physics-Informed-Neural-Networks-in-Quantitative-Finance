"""Tests for Diebold–Mariano and Pesaran–Timmermann forecast-accuracy tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.forecast_tests import diebold_mariano, pesaran_timmermann


# ── Diebold–Mariano ──────────────────────────────────────────────────────────

def test_dm_identical_forecasts_returns_neutral():
    """Identical errors → zero DM statistic, p = 1."""
    e = np.random.default_rng(0).normal(size=500)
    dm, p = diebold_mariano(e, e.copy())
    assert dm == 0.0 and p == 1.0


def test_dm_sign_convention_model_a_better_gives_negative_stat():
    """Model A with uniformly smaller squared error → DM < 0, significant."""
    rng = np.random.default_rng(1)
    e_a = rng.normal(0.0, 0.5, size=1000)   # lower variance
    e_b = rng.normal(0.0, 1.5, size=1000)   # higher variance
    dm, p = diebold_mariano(e_a, e_b, loss="se")
    assert dm < 0.0
    assert p < 0.01


def test_dm_abs_error_vs_squared_error_both_run():
    rng = np.random.default_rng(2)
    e_a = rng.normal(size=300)
    e_b = rng.normal(size=300)
    for lo in ("se", "ae"):
        dm, p = diebold_mariano(e_a, e_b, loss=lo)
        assert np.isfinite(dm) and 0.0 <= p <= 1.0


def test_dm_rejects_bad_inputs():
    a = np.zeros(10)
    b = np.zeros(12)
    with pytest.raises(ValueError):
        diebold_mariano(a, b)
    with pytest.raises(ValueError):
        diebold_mariano(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        diebold_mariano(a, a, h=0)
    with pytest.raises(ValueError):
        diebold_mariano(a, a, loss="xx")


# ── Pesaran–Timmermann ───────────────────────────────────────────────────────

def test_pt_strong_direction_is_highly_significant():
    """A noisy-oracle (correct sign ~95% of the time) must be highly significant.

    Perfectly correlated signs degenerate the PT variance formula
    (``Var(P*)=0`` at p*=1), so use a small amount of noise to stay in the
    regime where the test is well-defined — which is also the realistic
    regime for any real forecaster.
    """
    rng = np.random.default_rng(3)
    actual = rng.normal(0.0, 0.01, size=500)
    flip_mask = rng.random(500) < 0.05   # ~5% sign flips
    pred = np.where(flip_mask, -actual, actual)
    stat, p = pesaran_timmermann(pred, actual)
    assert stat > 5.0
    assert p < 1e-6


def test_pt_random_prediction_is_not_significant():
    rng = np.random.default_rng(4)
    actual = rng.normal(0.0, 0.01, size=2000)
    pred = rng.normal(0.0, 0.01, size=2000)
    _, p = pesaran_timmermann(pred, actual)
    assert p > 0.05


def test_pt_degenerate_marginal_returns_neutral():
    """All-zero pred (random_walk case) → degenerate marginal → p = 0.5."""
    actual = np.random.default_rng(5).normal(size=200)
    pred = np.zeros_like(actual)
    stat, p = pesaran_timmermann(pred, actual)
    # x = (pred >= 0).astype(float) → all 1s → px = 1 → degenerate marginal
    assert stat == 0.0 and p == 0.5


def test_pt_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pesaran_timmermann(np.zeros(10), np.zeros(12))
    with pytest.raises(ValueError):
        pesaran_timmermann(np.zeros((2, 3)), np.zeros((2, 3)))
