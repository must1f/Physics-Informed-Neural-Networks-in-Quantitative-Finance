"""Pure-function feature engineering for OHLCV data.

Computes all 19 research features defined in configs/dissertation.yaml.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # type: ignore
    _HAS_TA = True
except ImportError:
    ta = None
    _HAS_TA = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Public API ──────────────────────────────────────────────────────


def compute_features(
    df: pd.DataFrame,
    vix: pd.Series | None = None,
    tnx: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute research features from OHLCV data, optionally including VIX.

    Inputs:
        df: OHLCV DataFrame indexed by trading date. Must contain columns
            Open, Close, High, Low, Volume (raw prices / raw volume, not
            normalised). The index is expected to be a DatetimeIndex;
            its type is preserved through the warmup-row drop.
        vix: optional raw VIX Close series (pd.Series, DatetimeIndex,
            positive floats, percentage-point scale e.g. 20.0 = 20% annual
            implied vol). When supplied, adds ``vix_level``, ``vix_change``,
            and ``vol_premium`` columns. Missing VIX dates are forward-filled
            then back-filled to handle exchange calendar mismatches.
        tnx: optional raw TNX Close series (pd.Series, DatetimeIndex,
            positive floats, percentage-point scale e.g. 4.5 = 4.5%
            annualised yield). When supplied, adds ``tnx_level`` and
            ``tnx_change`` columns. Missing TNX dates are forward-filled
            then back-filled to handle calendar mismatches.

    Returns:
        DataFrame with 16 base engineered features (log_return,
        simple_return, rolling_volatility_{5,20}, momentum_{5,20}, rsi_14,
        macd, macd_signal, bollinger_upper, bollinger_lower, atr_14,
        volume_normalized, close_normalized, overnight_gap,
        rolling_skewness_20) plus, when vix is supplied, ``vix_level``,
        ``vix_change``, and ``vol_premium`` (19 features total, or 21 when
        tnx is also supplied). NaN warmup rows removed. DatetimeIndex
        preserved.
    """
    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]
    open_ = out["Open"]

    # Returns
    out["log_return"] = _log_return(close)
    out["simple_return"] = _simple_return(close)

    # Rolling volatility (annualised)
    out["rolling_volatility_5"] = _rolling_volatility(close, window=5)
    out["rolling_volatility_20"] = _rolling_volatility(close, window=20)

    # Momentum
    out["momentum_5"] = _momentum(close, window=5)
    out["momentum_20"] = _momentum(close, window=20)

    # Technical indicators
    out["rsi_14"] = _rsi(close, period=14)
    macd_line, macd_sig = _macd(close)
    out["macd"] = macd_line
    out["macd_signal"] = macd_sig
    bb_upper, bb_lower = _bollinger_bands(close, window=20)
    out["bollinger_upper"] = bb_upper
    out["bollinger_lower"] = bb_lower
    out["atr_14"] = _atr(high, low, close, period=14)

    # Normalised series
    out["volume_normalized"] = _normalize_volume(volume, window=20)
    out["close_normalized"] = _normalize_close(close, window=20)

    # After-hours and distributional features
    out["overnight_gap"] = _overnight_gap(open_, close)
    out["rolling_skewness_20"] = _rolling_skewness(close, window=20)

    # VIX macro features (optional)
    if vix is not None:
        vix_aligned = vix.reindex(out.index).ffill().bfill()
        out["vix_level"] = _vix_level(vix_aligned)
        out["vix_change"] = _vix_change(vix_aligned)
        out["vol_premium"] = _vol_premium(out["rolling_volatility_20"], vix_aligned)

    # TNX macro features (optional)
    if tnx is not None:
        tnx_aligned = tnx.reindex(out.index).ffill().bfill()
        out["tnx_level"] = _tnx_level(tnx_aligned)
        out["tnx_change"] = _tnx_change(tnx_aligned)

    n_before = len(out)
    # Drop warmup NaN rows but preserve the DatetimeIndex so downstream
    # consumers (TimeSeriesDataset, rolling error plots, equity curve
    # timestamping) keep their calendar-aware indexing.
    out = out.dropna()
    logger.info(
        "Features computed: {} cols, dropped {} warmup rows, {} rows remain",
        out.shape[1],
        n_before - len(out),
        len(out),
    )
    return out


# ── Private helpers ─────────────────────────────────────────────────


def _log_return(close: pd.Series) -> pd.Series:
    """Compute one-step log return: log(Close_t / Close_{t-1}).

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.

    Returns:
        pd.Series of log-returns (dimensionless). First value is NaN.
    """
    return np.log(close / close.shift(1))


def _simple_return(close: pd.Series) -> pd.Series:
    """Compute one-step simple return: (Close_t - Close_{t-1}) / Close_{t-1}.

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.

    Returns:
        pd.Series of simple returns (dimensionless). First value is NaN.
    """
    return close.pct_change()


def _rolling_volatility(close: pd.Series, window: int) -> pd.Series:
    """Annualised rolling volatility of log returns over ``window`` trading days.

    Computed as the rolling standard deviation of log returns scaled by
    sqrt(252) to annualise.

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.
        window: Lookback window in trading days.

    Returns:
        pd.Series of annualised volatility (decimal scale, e.g. 0.15 = 15%
        annual vol). First ``window`` values are NaN.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def _momentum(close: pd.Series, window: int) -> pd.Series:
    """Compute price momentum over ``window`` trading days.

    Defined as (Close_t / Close_{t-window}) - 1, i.e. the simple return
    over the lookback horizon.

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.
        window: Lookback horizon in trading days.

    Returns:
        pd.Series of momentum values (dimensionless simple return). First
        ``window`` values are NaN.
    """
    return close / close.shift(window) - 1


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) using Wilder smoothing.

    Uses ``pandas_ta`` when available; falls back to a manual EWM
    implementation with ``alpha = 1/period`` (equivalent to Wilder's
    smoothing). Output is bounded [0, 100].

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.
        period: Lookback period in trading days. Default 14.

    Returns:
        pd.Series of RSI values in the range [0, 100]. Values are NaN
        for the first ``period`` rows during warm-up.
    """
    if _HAS_TA:
        return ta.rsi(close, length=period)
    # Manual Wilder-smoothed RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """Compute the MACD line and signal line.

    MACD line = EMA(close, fast) - EMA(close, slow).
    Signal line = EMA(MACD line, signal).

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.
        fast: Fast EMA span. Default 12.
        slow: Slow EMA span. Default 26.
        signal: Signal EMA span. Default 9.

    Returns:
        Tuple of ``(macd_line, macd_signal)`` — both pd.Series on the same
        raw-price-difference scale as close (not normalised). First
        ``slow + signal`` rows are effectively in warm-up.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, macd_signal


def _bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series]:
    """Compute Bollinger Band upper and lower envelopes.

    Upper = SMA(window) + num_std * std(window).
    Lower = SMA(window) - num_std * std(window).

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.
        window: Rolling window in trading days. Default 20.
        num_std: Number of standard deviations for the band width. Default 2.0.

    Returns:
        Tuple of ``(upper, lower)`` — both pd.Series on the raw price scale.
        First ``window`` values are NaN.
    """
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    return sma + num_std * std, sma - num_std * std


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Compute Average True Range (ATR) over ``period`` trading days.

    True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|).
    ATR = simple rolling mean of True Range over ``period`` days.
    Uses ``pandas_ta`` when available; falls back to a manual rolling-mean
    implementation.

    Args:
        high:  Raw High price series, positive floats, DatetimeIndex.
        low:   Raw Low price series, positive floats, same index.
        close: Raw Close price series, positive floats, same index.
        period: Lookback in trading days. Default 14.

    Returns:
        pd.Series of ATR values on the raw price scale (same units as
        close). First ``period`` values are NaN.
    """
    if _HAS_TA:
        return ta.atr(high, low, close, length=period)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _normalize_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Normalise volume by its rolling mean: Volume_t / mean(Volume, window).

    Args:
        volume: Raw volume series (share count), positive floats, DatetimeIndex.
        window: Rolling mean window in trading days. Default 20.

    Returns:
        pd.Series of dimensionless volume ratios (1.0 = average volume).
        First ``window`` values are NaN.
    """
    return volume / volume.rolling(window).mean()


def _normalize_close(close: pd.Series, window: int = 20) -> pd.Series:
    """Normalise close price by its rolling mean: Close_t / mean(Close, window).

    Args:
        close: Raw close price series, positive floats, DatetimeIndex.
        window: Rolling mean window in trading days. Default 20.

    Returns:
        pd.Series of dimensionless price ratios (1.0 = at rolling mean).
        First ``window`` values are NaN.
    """
    return close / close.rolling(window).mean()


def _overnight_gap(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight gap = log(Open_t / Close_{t-1}).

    Inputs:
        open_: raw Open price series, raw price scale, DatetimeIndex.
        close: raw Close price series, raw price scale, same index.

    Returns:
        pd.Series of log-ratios. Positive = gap up overnight; negative = gap
        down. First value is NaN (no prior close).
    """
    return np.log(open_ / close.shift(1))


def _rolling_skewness(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling skewness of log returns over ``window`` trading days.

    Inputs:
        close: raw Close price series, raw price scale, DatetimeIndex.
        window: lookback in trading days (default 20).

    Returns:
        pd.Series. Negative skew = crash-risk / left-tail dominance, relevant
        for Hawkes PINN jump intensity. First ``window`` values are NaN.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).skew()


def _vol_premium(realized_vol_ann: pd.Series, vix_raw: pd.Series) -> pd.Series:
    """Volatility risk premium = annualised realised vol minus implied vol.

    Inputs:
        realized_vol_ann: annualised realised volatility (decimal, e.g. 0.15
            for 15%), typically ``rolling_volatility_20``.
        vix_raw: raw VIX Close series in percentage-point scale (e.g. 15.0
            for 15% implied vol). DatetimeIndex, already aligned to price index.

    Returns:
        pd.Series. Positive = realised vol exceeds implied (fear underpriced);
        negative = implied vol elevated above realised (classic risk premium).
        Directly relevant to BS-PINN: the gap feeds back into the sigma term.
    """
    return realized_vol_ann - (vix_raw / 100)


def _tnx_level(tnx: pd.Series, window: int = 20) -> pd.Series:
    """Normalised TNX level: TNX_t divided by its rolling 20-day mean.

    Inputs:
        tnx: Raw TNX Close series (percentage-point scale, e.g. 4.5 = 4.5%
            annualised yield). DatetimeIndex aligned to the S&P 500 calendar.
        window: Rolling mean window in trading days. Default 20.

    Returns:
        pd.Series, same index as tnx, values > 0 (ratio of level to
        rolling mean). Analogous to ``vix_level``.
    """
    rolling_mean = tnx.rolling(window).mean()
    return tnx / rolling_mean


def _tnx_change(tnx: pd.Series) -> pd.Series:
    """Day-over-day log change in TNX: log(TNX_t / TNX_{t-1}).

    Inputs:
        tnx: Raw TNX Close series (percentage-point scale, positive floats).
            DatetimeIndex aligned to the S&P 500 calendar.

    Returns:
        pd.Series of log-changes, same index as tnx (first value is NaN).
        Analogous to ``vix_change``.
    """
    return np.log(tnx / tnx.shift(1))


def _vix_level(vix: pd.Series, window: int = 20) -> pd.Series:
    """vix_level = VIX_t / rolling_mean(VIX, window).

    Inputs:
        vix: raw VIX Close series, any positive float scale, DatetimeIndex.
        window: rolling mean window (default 20 trading days).

    Returns:
        pd.Series, same index as vix, values > 0. Ratio of 1.0 means VIX is
        at its recent average; > 1 means elevated fear.
    """
    return vix / vix.rolling(window).mean()


def _vix_change(vix: pd.Series) -> pd.Series:
    """vix_change = log(VIX_t / VIX_{t-1}).

    Inputs:
        vix: raw VIX Close series, positive floats, DatetimeIndex.

    Returns:
        pd.Series of log-differences. First value is NaN (no prior day).
        Positive = VIX spiked (rising fear); negative = VIX fell.
    """
    return np.log(vix / vix.shift(1))
