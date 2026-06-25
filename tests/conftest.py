"""Shared pytest fixtures for the FINN test suite."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """Seeded numpy Generator — deterministic across tests."""
    return np.random.default_rng(42)


@pytest.fixture
def perfect_prediction(rng):
    """Prediction identical to actual — metrics should be trivial."""
    actual = rng.standard_normal(500) * 0.01
    return actual.copy(), actual


@pytest.fixture
def noisy_prediction(rng):
    """Actual log-returns + gaussian noise for realistic metric values."""
    actual = rng.standard_normal(500) * 0.012
    pred = actual + rng.standard_normal(500) * 0.004
    return pred, actual


@pytest.fixture
def constant_returns():
    """Deterministic 252-day return series for financial metrics."""
    return np.full(252, 0.0004)  # ~10% annual
