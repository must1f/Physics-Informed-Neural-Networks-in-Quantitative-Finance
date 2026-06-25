"""Tests for classify_metric — turns numbers into quality labels."""
import pytest

from src.evaluation.benchmarks import classify_metric


@pytest.mark.parametrize("value,expected", [
    (1.6, "EXCELLENT"),
    (1.2, "GOOD"),
    (0.7, "AVERAGE"),
    (0.2, "POOR"),
    (-0.1, "BROKEN"),
    (3.0, "SUSPICIOUS"),
])
def test_sharpe_bands(value, expected):
    assert classify_metric("sharpe", value) == expected


@pytest.mark.parametrize("value,expected", [
    (0.57, "EXCELLENT"),
    (0.54, "GOOD"),
    (0.52, "AVERAGE"),
    (0.50, "POOR"),
    (0.48, "BROKEN"),
    (0.62, "SUSPICIOUS"),
])
def test_directional_accuracy_bands(value, expected):
    assert classify_metric("directional_accuracy", value) == expected


def test_rmse_requires_sigma():
    # ratio 0.533 → EXCELLENT (< 0.90)
    assert classify_metric("rmse", 0.008, sigma=0.015) == "EXCELLENT"
    # ratio 0.933 → GOOD (0.90–0.97)
    assert classify_metric("rmse", 0.014, sigma=0.015) == "GOOD"
    # ratio 1.333 → BROKEN (> 1.0)
    assert classify_metric("rmse", 0.020, sigma=0.015) == "BROKEN"
    # missing sigma → UNKNOWN
    assert classify_metric("rmse", 0.020) == "UNKNOWN"


def test_unknown_metric_returns_unknown():
    assert classify_metric("totally_made_up", 1.0) == "UNKNOWN"
