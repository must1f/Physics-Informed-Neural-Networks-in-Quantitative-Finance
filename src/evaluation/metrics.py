"""Pure-function metrics for model evaluation.

All public metrics accept 1-D NumPy arrays of equal length on the
data scale chosen by the caller (log-returns, simple-returns, or
prices — be consistent within a call). No mutable state, no classes,
no PyTorch imports.

Prediction-quality metrics: :func:`rmse`, :func:`mae`,
:func:`r_squared`, :func:`directional_accuracy`, :func:`mape`.

Financial metrics: :func:`sharpe_ratio`, :func:`sortino_ratio`,
:func:`max_drawdown`, :func:`calmar_ratio`.

Aggregator: :func:`compute_all_metrics` returns a flat dict of all
nine metrics in one call.
"""
from __future__ import annotations

import numpy as np

from src.constants import RISK_FREE_RATE, TRADING_DAYS_PER_YEAR


def _check(pred: np.ndarray, actual: np.ndarray) -> None:
    """Validate two metric inputs share shape and are 1-D.

    Args:
        pred: Any NumPy array — compared to ``actual``.
        actual: Any NumPy array — compared to ``pred``.

    Raises:
        ValueError: If shapes differ or ``ndim != 1``.
    """
    if pred.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs actual {actual.shape}"
        )
    if pred.ndim != 1:
        raise ValueError(f"Expected 1-D arrays, got ndim={pred.ndim}")


def rmse(pred: np.ndarray, actual: np.ndarray) -> float:
    """Root Mean Squared Error — penalises large errors quadratically.

    Computes ``sqrt(mean((pred - actual)**2))``. Lower is better.
    Scale-dependent: reported in the same units as ``actual`` (e.g.
    log-return units if predicting log-returns).

    Args:
        pred: 1-D NumPy array of predictions, shape ``(N,)``.
        actual: 1-D NumPy array of ground-truth values, shape ``(N,)``.

    Returns:
        Non-negative scalar in the original data units.

    Raises:
        ValueError: If ``pred`` and ``actual`` differ in shape or are
            not 1-D.
    """
    _check(pred, actual)
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean Absolute Error — robust-to-outliers L1 error.

    Computes ``mean(|pred - actual|)``. Lower is better. Reported in
    the same units as ``actual``.

    Args:
        pred: 1-D predictions, shape ``(N,)``.
        actual: 1-D ground truth, shape ``(N,)``.

    Returns:
        Non-negative scalar in the original data units.

    Raises:
        ValueError: Shape mismatch or non-1-D input.
    """
    _check(pred, actual)
    return float(np.mean(np.abs(pred - actual)))


def r_squared(pred: np.ndarray, actual: np.ndarray) -> float:
    """Coefficient of determination — variance explained by the model.

    Computes ``1 - SS_res / SS_tot`` where
    ``SS_res = sum((actual - pred)**2)`` and
    ``SS_tot = sum((actual - mean(actual))**2)``.

    Interpretation:
      * ``1.0`` — perfect fit.
      * ``0.0`` — model equivalent to predicting ``mean(actual)``.
      * negative — model is *worse* than the mean predictor.

    Args:
        pred: 1-D predictions, shape ``(N,)``.
        actual: 1-D ground truth, shape ``(N,)``.

    Returns:
        Dimensionless scalar in ``(-∞, 1]``. If ``actual`` is constant
        (zero total variance), returns ``0.0`` by convention.

    Raises:
        ValueError: Shape mismatch or non-1-D input.
    """
    _check(pred, actual)
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def directional_accuracy(pred: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of samples where ``sign(pred) == sign(actual)``.

    Used on return-space predictions to measure whether the model
    calls the *direction* correctly — the primitive skill behind a
    long/short strategy.

    Tie-break: ``np.sign(0) == 0``, so zeros are only correct when
    both arrays are zero at that index. Callers predicting log-returns
    should rarely see exact zeros.

    Args:
        pred: 1-D predicted returns, shape ``(N,)``.
        actual: 1-D realised returns, shape ``(N,)``.

    Returns:
        Scalar in ``[0, 1]``. ``0.5`` = coin flip; > 0.5 beats random.

    Raises:
        ValueError: Shape mismatch or non-1-D input.
    """
    _check(pred, actual)
    return float(np.mean(np.sign(pred) == np.sign(actual)))


def mape(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean Absolute Percentage Error — scale-invariant L1 error.

    Computes ``mean(|(pred - actual) / actual|)`` over indices where
    ``actual != 0``. **Do not use on returns** (division by values
    near zero blows up) — use on prices or other strictly-positive
    targets.

    Args:
        pred: 1-D predictions, shape ``(N,)``.
        actual: 1-D ground truth, shape ``(N,)``. Indices where
            ``actual == 0`` are dropped.

    Returns:
        Non-negative scalar expressing average relative error as a
        fraction (multiply by 100 for a percentage). Returns ``0.0``
        if every sample has ``actual == 0``.

    Raises:
        ValueError: Shape mismatch or non-1-D input.
    """
    _check(pred, actual)
    mask = actual != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((pred[mask] - actual[mask]) / actual[mask])))


def sharpe_ratio(
    returns: np.ndarray,
    rf: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio — risk-adjusted excess return.

    Computes ``sqrt(periods) * mean(excess) / std(excess, ddof=1)``,
    where ``excess = returns - rf / periods``. Sample std (``ddof=1``)
    is used so the Sharpe of a short series is unbiased.

    Args:
        returns: 1-D simple-return series at the sampling frequency
            (daily if ``periods=252``), shape ``(N,)``.
        rf: Annual risk-free rate, decimal (default
            :data:`~src.constants.RISK_FREE_RATE` = 0.02 = 2%).
        periods: Samples per year used for annualisation — 252 for
            daily, 52 weekly, 12 monthly (default
            :data:`~src.constants.TRADING_DAYS_PER_YEAR` = 252).

    Returns:
        Dimensionless annualised Sharpe. Returns ``0.0`` if the excess
        series is constant (``std < 1e-12``) to avoid divide-by-zero
        producing spurious infinities.

    Raises:
        ValueError: If ``returns`` is not 1-D.
    """
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D, got ndim={returns.ndim}")
    excess = returns - rf / periods
    std = float(excess.std(ddof=1)) if excess.size > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / std)


def sortino_ratio(
    returns: np.ndarray,
    rf: float = RISK_FREE_RATE,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sortino ratio — Sharpe using downside deviation only.

    Computes ``sqrt(periods) * mean(excess) / downside_dev``, where
    ``downside_dev = sqrt(mean(excess_neg**2))`` over *negative*
    excess returns only. Upside volatility is not penalised, making
    this a more appropriate risk measure for skewed strategies.

    Args:
        returns: 1-D simple-return series at the sampling frequency,
            shape ``(N,)``.
        rf: Annual risk-free rate, decimal (default
            :data:`~src.constants.RISK_FREE_RATE`).
        periods: Samples per year for annualisation (default 252).

    Returns:
        Dimensionless annualised Sortino. Returns ``0.0`` if no
        excess return is negative (zero downside variance) **or** if
        ``returns`` is identically zero (flat-cash strategy). Without
        the zero-series guard, a no-trade strategy produces the
        degenerate value ``-sqrt(periods)`` (≈ -15.87 daily) because
        every daily excess return equals ``-rf/periods`` — a
        mathematical artefact of annualising a constant-negative
        series, not a financial statement about the strategy.

    Raises:
        ValueError: If ``returns`` is not 1-D.
    """
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D, got ndim={returns.ndim}")
    # Zero-series guard: a flat (all-zero) return stream is the "no
    # trade" / "held cash" case. Report 0.0 to match the corresponding
    # sharpe_ratio std-guard above; otherwise the formula collapses to
    # -sqrt(periods) purely from the rf/periods offset.
    if returns.size == 0 or np.allclose(returns, 0.0):
        return 0.0
    excess = returns - rf / periods
    downside = excess[excess < 0]
    if downside.size == 0:
        return 0.0
    dd = float(np.sqrt(np.mean(downside ** 2)))
    if dd == 0.0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / dd)


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline of an equity curve.

    Computes ``max((peak - value) / peak)`` where ``peak`` is the
    running maximum of ``equity``. Expressed as a *positive* fraction
    so that ``0.33`` means "33% drawdown from the prior high".

    Args:
        equity: 1-D strictly-positive cumulative equity curve, e.g.
            ``np.cumprod(1 + strategy_returns)``, shape ``(N,)``.

    Returns:
        Scalar in ``[0, 1]``. ``0.0`` = monotone non-decreasing curve.

    Raises:
        ValueError: If ``equity`` is empty or not 1-D.
    """
    if equity.ndim != 1 or equity.size == 0:
        raise ValueError("equity must be non-empty 1-D array")
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / peak
    return float(drawdowns.max())


def calmar_ratio(
    returns: np.ndarray,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised return divided by the max drawdown of the equity curve.

    Builds the equity curve internally from ``returns`` via
    ``cumprod(1 + returns)``. The annualised return is computed via
    CAGR: ``equity[-1] ** (periods / N) - 1``.

    Args:
        returns: 1-D simple-return series at the sampling frequency,
            shape ``(N,)``. Values must satisfy ``returns > -1`` so
            the equity curve stays positive.
        periods: Samples per year for annualisation (default 252).

    Returns:
        Dimensionless Calmar ratio. Returns ``0.0`` if the strategy
        had no drawdown (no losing period) to avoid divide-by-zero.
        Can be negative if total return is negative.
    """
    equity = np.cumprod(1.0 + returns)
    mdd = max_drawdown(equity)
    if mdd == 0.0:
        return 0.0
    annual_return = equity[-1] ** (periods / len(returns)) - 1.0
    return float(annual_return / mdd)


def information_ratio(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Information Ratio — Sharpe of excess return over benchmark.

    Computes ``Sharpe(strategy_returns - benchmark_returns)``. A model
    that is permanently long (``sign(pred) = +1`` always) produces
    strategy_returns == benchmark_returns, so IR = 0 by construction.
    This corrects the deficiency of raw Sharpe, which conflates model
    skill with the market's own return during the test period.

    Args:
        strategy_returns: 1-D sign-based strategy returns
            ``np.sign(pred) * actual``, shape ``(N,)``.
        benchmark_returns: 1-D buy-and-hold returns (``actual``
            log-returns), same shape as ``strategy_returns``.
        periods: Samples per year for annualisation (default 252).

    Returns:
        Dimensionless annualised IR. Positive = beats buy-and-hold,
        zero = identical to buy-and-hold, negative = worse.
    """
    excess = strategy_returns - benchmark_returns
    std = float(excess.std(ddof=1)) if excess.size > 1 else 0.0
    if std < 1e-12:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / std)


def compute_all_metrics(
    pred: np.ndarray,
    actual: np.ndarray,
    returns: np.ndarray,
) -> dict[str, float]:
    """Aggregate every prediction + financial metric into one dict.

    Convenience wrapper that evaluates all ten metrics against a
    single ``(pred, actual, returns)`` triplet. The equity curve for
    the financial metrics is built internally from ``returns``.

    Args:
        pred: 1-D model predictions, shape ``(N,)``. Scale matches
            ``actual`` (log-returns if ``actual`` is log-returns).
        actual: 1-D ground-truth values aligned with ``pred``, shape
            ``(N,)``.
        returns: 1-D per-period P&L series driving the financial
            metrics (Sharpe, Sortino, MDD, Calmar), shape ``(N,)``.
            For strategy evaluation pass the **strategy returns**, e.g.
            ``np.sign(pred) * actual`` for a sign-based long/short
            rule — passing ``actual`` here reports buy-and-hold and
            makes every model look identical on the financial metrics.
            Must satisfy ``returns > -1`` so the internal equity curve
            stays positive.

    Returns:
        Dict with float values under ten fixed keys:
        ``rmse``, ``mae``, ``r_squared``, ``directional_accuracy``,
        ``mape``, ``sharpe``, ``sortino``, ``max_drawdown``,
        ``calmar``, ``information_ratio``.

        ``information_ratio`` is the Sharpe of ``returns - actual``
        (strategy excess over buy-and-hold). A permanently-long model
        has IR = 0 regardless of market conditions, making it a
        market-direction-neutral measure of model skill.
    """
    equity = np.cumprod(1.0 + returns)
    return {
        "rmse": rmse(pred, actual),
        "mae": mae(pred, actual),
        "r_squared": r_squared(pred, actual),
        "directional_accuracy": directional_accuracy(pred, actual),
        "mape": mape(pred, actual),
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(returns),
        "information_ratio": information_ratio(returns, actual),
    }
