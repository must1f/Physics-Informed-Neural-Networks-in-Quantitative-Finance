"""Pure-function helpers for volatility-forecast evaluation.

Used by the GARCH volatility-forecast appendix (Documentation/Plans/
2026-04-18-garch-volatility-appendix.md). All public functions accept
1-D NumPy arrays, return NumPy arrays or Python scalars, and have no
mutable state.

Entry points:
    realised_vol_parkinson: daily vol proxy from High/Low bars.
    realised_vol_squared_returns: fall-back proxy for close-only series.
    rolling_realised_vol: sliding-window RV benchmark forecaster.
    mincer_zarnowitz: OLS forecast-vs-proxy diagnostic + F-test.
    qlike_loss: asymmetric volatility loss (Patton 2011).
"""
from __future__ import annotations

import numpy as np


_LN2_TIMES_4 = 4.0 * np.log(2.0)


def realised_vol_parkinson(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Parkinson (1980) daily volatility estimator from OHLC High/Low bars.

    Computes ``σ_t = sqrt((ln H_t − ln L_t)² / (4 ln 2))`` — the efficient
    daily vol proxy when only High/Low are available. About 5× lower
    variance than squared returns for the same sample size under
    geometric Brownian motion (Parkinson 1980; Alizadeh et al. 2002).

    Args:
        high: 1-D array of daily high prices, shape (N,), strictly positive.
        low:  1-D array of daily low prices, shape (N,), strictly positive
              and element-wise ``<= high``.

    Returns:
        1-D array of σ_t estimates on the native price-return scale (same
        units as a daily log-return standard deviation), shape (N,).
        Flat bars (H==L) yield exactly 0.0 — not NaN — so the series stays
        usable in downstream regressions without dropna.

    Raises:
        ValueError: If shapes differ, inputs are not 1-D, any
            ``low > high`` bar is present, or any price is
            non-positive.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if high.ndim != 1 or low.ndim != 1:
        raise ValueError("high and low must be 1-D arrays")
    if high.shape != low.shape:
        raise ValueError(f"Shape mismatch: high {high.shape} vs low {low.shape}")
    if np.any(low > high):
        raise ValueError("Found a bar with low > high (arbitrage-impossible)")
    if np.any(high <= 0.0) or np.any(low <= 0.0):
        raise ValueError("Prices must be strictly positive (found non-positive values)")
    log_hl = np.log(high) - np.log(low)
    return np.sqrt(log_hl ** 2 / _LN2_TIMES_4)


def realised_vol_squared_returns(returns: np.ndarray) -> np.ndarray:
    """Single-sample squared-returns proxy: σ_t = |r_t|.

    Fall-back when OHLC High/Low are unavailable (close-only series).
    Unbiased for E[σ²_t] but an order of magnitude noisier than
    Parkinson; prefer :func:`realised_vol_parkinson` whenever the
    High/Low columns are present.

    Args:
        returns: 1-D array of log-returns, shape (N,), any scale.

    Returns:
        1-D array of |r_t| on the same scale as ``returns``, shape (N,).

    Raises:
        ValueError: If ``returns`` is not 1-D.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D, got ndim={returns.ndim}")
    return np.abs(returns)


def rolling_realised_vol(
    returns: np.ndarray,
    window: int = 22,
) -> np.ndarray:
    """Rolling-window realised-volatility benchmark forecaster.

    Computes ``σ̂_t = sqrt(mean(r²_{t-window..t-1}))`` using a left-aligned
    trailing window. The forecast at time t uses only information
    strictly before t (no look-ahead) — the canonical "naive forecaster"
    benchmark in the volatility literature. Any serious σ_t forecaster
    should beat it on QLIKE.

    Args:
        returns: 1-D log-return series, shape (N,), any scale.
        window: Trailing window length in periods (default 22, ≈ one
            trading month). Must satisfy ``1 <= window < N``.

    Returns:
        1-D array of one-step-ahead volatility forecasts, shape (N,).
        The first ``window`` entries are NaN (warm-up); from index
        ``window`` onward each entry is the RMS of the preceding
        ``window`` returns.

    Raises:
        ValueError: If ``returns`` is not 1-D, ``window < 1``, or
            ``window >= len(returns)``.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError(f"Expected 1-D, got ndim={returns.ndim}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if window >= returns.size:
        raise ValueError(
            f"window={window} must be < len(returns)={returns.size}"
        )
    r2 = returns ** 2
    out = np.full_like(returns, np.nan)
    cumsum = np.concatenate([[0.0], np.cumsum(r2)])
    # Forecast at t uses r²_{t-window..t-1} — no look-ahead.
    for t in range(window, returns.size):
        out[t] = np.sqrt((cumsum[t] - cumsum[t - window]) / window)
    return out


def qlike_loss(
    forecast: np.ndarray,
    proxy: np.ndarray,
) -> float:
    """Mean QLIKE loss for variance forecasts (Patton 2011).

    Computes ``mean(log(σ²_f) + σ̂²/σ²_f)`` where ``σ_f`` is the forecast
    and ``σ̂`` is the realised-vol proxy. QLIKE is the standard scoring
    rule for variance forecasts because (a) it is robust to the noisy
    proxy under conditional unbiasedness and (b) it penalises
    under-prediction strictly more than over-prediction — economically
    correct for risk-management use cases.

    Lower is better. NaN entries (e.g. warm-up from rolling_realised_vol)
    are dropped before averaging.

    Args:
        forecast: 1-D array of σ_t forecasts, shape (N,), strictly
            positive (zero forecasts are undefined under QLIKE). NaN
            entries are dropped before averaging alongside ``proxy``.
        proxy: 1-D array of σ_t realised-vol proxy, shape (N,), on the
            same scale as ``forecast``. May contain NaN warm-up.

    Returns:
        Scalar mean QLIKE loss (finite float). Returns ``float("nan")``
        if every aligned sample is NaN.

    Raises:
        ValueError: If shapes differ, arrays are not 1-D, or any
            non-NaN ``forecast`` entry is ``<= 0``.
    """
    forecast = np.asarray(forecast, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    if forecast.ndim != 1 or proxy.ndim != 1:
        raise ValueError("forecast and proxy must be 1-D arrays")
    if forecast.shape != proxy.shape:
        raise ValueError(
            f"Shape mismatch: forecast {forecast.shape} vs proxy {proxy.shape}"
        )
    valid = ~np.isnan(forecast)
    if np.any(forecast[valid] <= 0.0):
        raise ValueError("forecast must be strictly positive for QLIKE")
    mask = valid & ~np.isnan(proxy)
    if not mask.any():
        return float("nan")
    f2 = forecast[mask] ** 2
    p2 = proxy[mask] ** 2
    return float(np.mean(np.log(f2) + p2 / f2))


def mincer_zarnowitz(
    forecast: np.ndarray,
    proxy: np.ndarray,
) -> dict[str, float]:
    """Mincer–Zarnowitz (1969) regression on variance forecasts.

    Runs OLS on ``σ̂²_t = α + β · σ²_f_t + ε_t`` and tests the joint
    null H0: ``(α, β) = (0, 1)`` via an F-test. The canonical
    diagnostic for forecast unbiasedness plus calibration:

    * α ≠ 0       → systematic bias (forecast over- or under-states).
    * β ≠ 1       → calibration error (forecast is too flat / too sharp).
    * low R²      → forecast captures little of the proxy's variation.
    * small p     → joint null rejected; forecast is misspecified.

    NaN rows in either input are dropped before fitting.

    Args:
        forecast: 1-D σ_t forecasts, shape (N,). Strictly positive
            (the regression is on σ², so zero is allowed numerically
            but has no economic interpretation).
        proxy: 1-D σ_t realised-vol proxy, shape (N,), on the same
            scale as ``forecast``.

    Returns:
        Dict with:
            ``alpha`` (float): OLS intercept on σ² scale.
            ``beta`` (float): OLS slope on forecast variance.
            ``r_squared`` (float): regression R² on σ² scale.
            ``joint_p`` (float): F-test p-value for H0 (α=0, β=1).
                May be NaN when the regression is perfectly deterministic
                (zero-variance residual).
            ``n`` (int): sample size after NaN drop.

    Raises:
        ValueError: Shape mismatch, non-1-D input, or < 10 valid rows
            after NaN drop.
    """
    import statsmodels.api as sm

    forecast = np.asarray(forecast, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    if forecast.ndim != 1 or proxy.ndim != 1:
        raise ValueError("forecast and proxy must be 1-D arrays")
    if forecast.shape != proxy.shape:
        raise ValueError(
            f"Shape mismatch: forecast {forecast.shape} vs proxy {proxy.shape}"
        )
    mask = ~(np.isnan(forecast) | np.isnan(proxy))
    if mask.sum() < 10:
        raise ValueError(
            f"Need at least 10 valid rows for MZ regression, got {int(mask.sum())}"
        )
    f2 = forecast[mask] ** 2
    p2 = proxy[mask] ** 2
    X = sm.add_constant(f2)
    model = sm.OLS(p2, X).fit()
    # Joint Wald test for (α=0, β=1) using statsmodels' string syntax.
    # When the fitted parameters are already at the null (α≈0, β≈1) AND
    # the fit is perfectly deterministic (RSS≈0), the F-statistic is
    # numerically degenerate; return NaN to signal this edge case.
    alpha_hat = float(model.params[0])
    beta_hat = float(model.params[1])
    _perfectly_at_null = (
        abs(alpha_hat) < 1e-12
        and abs(beta_hat - 1.0) < 1e-9
        and model.ssr < 1e-20 * max(model.centered_tss, 1e-100)
    )
    try:
        if _perfectly_at_null:
            joint_p = float("nan")
        else:
            f_test = model.f_test("const = 0, x1 = 1")
            joint_p = float(f_test.pvalue)
    except Exception:
        joint_p = float("nan")
    return {
        "alpha": float(model.params[0]),
        "beta": float(model.params[1]),
        "r_squared": float(model.rsquared),
        "joint_p": joint_p,
        "n": int(mask.sum()),
    }


def evaluate_volatility_forecast(
    forecast: np.ndarray,
    proxy: np.ndarray,
) -> dict[str, float]:
    """One-shot aggregator: QLIKE + MZ regression in a single dict.

    Convenience wrapper for notebook cells and sidecar JSON. Computes
    both :func:`qlike_loss` and :func:`mincer_zarnowitz` on the same
    ``(forecast, proxy)`` pair and returns a flat dict suitable for
    ``json.dump``.

    Args:
        forecast: 1-D σ_t forecasts, shape (N,), strictly positive.
        proxy: 1-D σ_t realised-vol proxy, shape (N,), on the same
            scale as ``forecast``.

    Returns:
        Flat dict with keys:
            ``qlike`` (float): mean QLIKE loss, lower is better.
            ``mz_alpha``, ``mz_beta``, ``mz_r_squared``, ``mz_joint_p``
            (float): MZ regression outputs (``mz_joint_p`` may be NaN
            when the fit is perfectly deterministic at the null).
            ``n`` (int): valid sample count after NaN drop.
    """
    mz = mincer_zarnowitz(forecast, proxy)
    return {
        "qlike": qlike_loss(forecast, proxy),
        "mz_alpha": mz["alpha"],
        "mz_beta": mz["beta"],
        "mz_r_squared": mz["r_squared"],
        "mz_joint_p": mz["joint_p"],
        "n": mz["n"],
    }
