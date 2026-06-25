"""Benchmark thresholds for ML financial forecasting metrics.

Thresholds are drawn from the dissertation's literature review
([[ML_Return_Forecasting_Benchmarks]]). Every
band is a rough convention — use these to *flag* results for human
review, not to make publish/no-publish decisions automatically.

The ``classify_metric(name, value, sigma=None)`` function returns one
of: EXCELLENT, GOOD, AVERAGE, POOR, BROKEN, SUSPICIOUS, UNKNOWN.

SUSPICIOUS is reserved for values that are *too good* and likely
reflect leakage or overfitting (e.g. daily Sharpe > 2.5, DA > 0.60,
R² > 0.05, MDD < 0.05 on a 3yr daily backtest).
"""
from __future__ import annotations

_ABSOLUTE_BANDS: dict[str, dict[str, float]] = {
    "sharpe":                {"excellent": 1.5, "good": 1.0, "average": 0.4,  "poor": 0.0,  "suspicious": 2.5},
    "sortino":               {"excellent": 1.5, "good": 1.0, "average": 0.6,  "poor": 0.0,  "suspicious": 3.0},
    "calmar":                {"excellent": 1.0, "good": 0.5, "average": 0.3,  "poor": 0.0,  "suspicious": 3.0},
    "directional_accuracy":  {"excellent": 0.56,"good": 0.53,"average": 0.51, "poor": 0.49, "suspicious": 0.60},
    "r_squared":             {"excellent": 0.01,"good": 0.005,"average": 0.001,"poor": 0.0, "suspicious": 0.05},
}

_MDD_BANDS = {"excellent": 0.10, "good": 0.20, "average": 0.35, "poor": 0.50, "suspicious_lo": 0.05}

_RELATIVE_BANDS: dict[str, dict[str, float]] = {
    "rmse": {"excellent": 0.90, "good": 0.97, "average": 1.00},
    "mae":  {"excellent": 0.75, "good": 0.85, "average": 0.95},
}


def classify_metric(name: str, value: float, sigma: float | None = None) -> str:
    """Label a metric value against literature benchmark bands.

    Turns a raw number into one of seven qualitative buckets so
    notebooks and dashboards can flag suspicious or broken results
    without hard-coded thresholds scattered through notebook cells.
    Bands follow the table in the Phase 5 plan's "Benchmark Reference"
    section (Fischer & Krauss 2018; Gu, Kelly & Xiu 2020; Lyu & Wang
    2024; López de Prado 2018).

    Three band families are supported:

      * **Absolute** (``sharpe``, ``sortino``, ``calmar``,
        ``directional_accuracy``, ``r_squared``) — higher is better;
        a ``suspicious`` ceiling flags values that likely reflect
        overfitting or data leakage.
      * **Max drawdown** — lower is better; a ``suspicious_lo`` floor
        flags unrealistically small drawdowns (no real exposure).
      * **Sigma-relative** (``rmse``, ``mae``) — thresholds are a
        fraction of the target standard deviation ``sigma`` supplied
        by the caller. Returns ``UNKNOWN`` if ``sigma`` is missing or
        non-positive.

    Args:
        name: Metric key matching a ``compute_all_metrics`` output
            name — one of ``{sharpe, sortino, calmar,
            directional_accuracy, r_squared, max_drawdown, rmse,
            mae}``. Unknown names return ``"UNKNOWN"``.
        value: Observed metric value on the same scale the metric
            was computed at.
        sigma: Target standard deviation for sigma-relative metrics
            (e.g. daily log-return σ for ``rmse``). Ignored for
            absolute metrics.

    Returns:
        One of ``"EXCELLENT"``, ``"GOOD"``, ``"AVERAGE"``, ``"POOR"``,
        ``"BROKEN"``, ``"SUSPICIOUS"``, or ``"UNKNOWN"``.
    """
    if name in _ABSOLUTE_BANDS:
        b = _ABSOLUTE_BANDS[name]
        if value >= b["suspicious"]:
            return "SUSPICIOUS"
        if value >= b["excellent"]:
            return "EXCELLENT"
        if value >= b["good"]:
            return "GOOD"
        if value >= b["average"]:
            return "AVERAGE"
        if value >= b["poor"]:
            return "POOR"
        return "BROKEN"

    if name == "max_drawdown":
        if value < _MDD_BANDS["suspicious_lo"]:
            return "SUSPICIOUS"
        if value < _MDD_BANDS["excellent"]:
            return "EXCELLENT"
        if value < _MDD_BANDS["good"]:
            return "GOOD"
        if value < _MDD_BANDS["average"]:
            return "AVERAGE"
        if value < _MDD_BANDS["poor"]:
            return "POOR"
        return "BROKEN"

    if name in _RELATIVE_BANDS:
        if sigma is None or sigma <= 0:
            return "UNKNOWN"
        b = _RELATIVE_BANDS[name]
        ratio = value / sigma
        if ratio < b["excellent"]:
            return "EXCELLENT"
        if ratio < b["good"]:
            return "GOOD"
        if ratio < b["average"]:
            return "AVERAGE"
        if ratio < 1.0:
            return "POOR"
        return "BROKEN"

    return "UNKNOWN"
