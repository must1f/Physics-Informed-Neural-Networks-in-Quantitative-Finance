"""Fetch OHLCV data from Yahoo Finance with disk caching."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

logger = get_logger(__name__)

_MIN_ROWS = 252  # at least one trading year
_MAX_GAP_DAYS = 5  # flag gaps larger than this
_REQUIRED_COLS = {"Open", "High", "Low", "Close", "Volume"}


class DataFetcher:
    """Fetch OHLCV data from Yahoo Finance with parquet caching."""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        """Initialise the fetcher and create the cache directory if absent.

        Args:
            cache_dir: Path to the directory used for parquet caching.
                Created (including parents) on first use. Relative paths
                are resolved from the current working directory.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch full OHLCV data for a ticker, using the parquet cache when available.

        On a cache miss, downloads from Yahoo Finance via ``yfinance``, validates
        the data, writes a parquet file, and returns the DataFrame. On a cache hit,
        reads and returns the cached file directly without any network call.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. ``"^GSPC"``).
            start:  ISO date string, inclusive (e.g. ``"2010-01-01"``).
            end:    ISO date string, exclusive (e.g. ``"2025-01-01"``).

        Returns:
            ``pd.DataFrame`` with a ``DatetimeIndex`` and five columns:
            ``Open``, ``High``, ``Low``, ``Close``, ``Volume`` — all on the
            raw price / volume scale, adjusted for splits and dividends
            (``auto_adjust=True`` in yfinance). At least ``_MIN_ROWS`` rows
            are guaranteed (raises otherwise).

        Raises:
            ValueError: If yfinance returns no data, any required column is
                missing or all-NaN, or fewer than ``_MIN_ROWS`` rows are
                present after download.
        """
        cache_path = self._cache_path(ticker, start, end)

        # 1. Check cache
        if cache_path.exists():
            logger.info("Cache hit: {}", cache_path.name)
            df = pd.read_parquet(cache_path)
            return df

        # 2. Download from Yahoo Finance
        logger.info("Fetching {} from yfinance ({} -> {})", ticker, start, end)
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise ValueError(f"No data returned for {ticker} ({start} – {end})")

        # Flatten MultiIndex columns if yfinance returns them
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[list(_REQUIRED_COLS)].copy()

        # 3. Validate
        self._validate(df, ticker)

        # 4. Save to cache
        df.to_parquet(cache_path)
        logger.info("Cached {} rows -> {}", len(df), cache_path.name)

        return df

    def fetch_close(self, ticker: str, start: str, end: str) -> pd.Series:
        """Fetch only the Close series for a scalar index (e.g. ^VIX).

        On a cache miss downloads via yfinance, validates row count, caches to
        parquet, and returns the series. On a cache hit reads from disk.

        Args:
            ticker: Yahoo Finance ticker symbol (e.g. ``"^VIX"``).
            start:  ISO date string, inclusive (e.g. ``"2010-01-01"``).
            end:    ISO date string, exclusive (e.g. ``"2024-12-31"``).

        Returns:
            ``pd.Series`` named ``"Close"`` with ``DatetimeIndex``, raw index
            values (not normalised, not log-transformed). At least
            ``_MIN_ROWS`` rows guaranteed. Cached under
            ``<cache_dir>/<safe_ticker>_close_<start>_<end>.parquet``.

        Raises:
            ValueError: If yfinance returns no data or fewer than
                ``_MIN_ROWS`` rows are present after download.
        """
        cache_path = self._cache_path(ticker, start, end, suffix="close")
        if cache_path.exists():
            logger.info("Cache hit: {}", cache_path.name)
            return pd.read_parquet(cache_path)["Close"]

        logger.info("Fetching {} close from yfinance ({} -> {})", ticker, start, end)
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

        if raw.empty:
            raise ValueError(f"No data returned for {ticker} ({start} – {end})")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        series = raw["Close"].dropna()
        if len(series) < _MIN_ROWS:
            raise ValueError(f"{ticker}: only {len(series)} rows (need >= {_MIN_ROWS})")

        pd.DataFrame({"Close": series}).to_parquet(cache_path)
        logger.info("Cached {} rows -> {}", len(series), cache_path.name)
        return series

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str, start: str, end: str, suffix: str = "") -> Path:
        """Build a filesystem-safe parquet cache path for a ticker/date range.

        Args:
            ticker: Raw ticker symbol — ``^`` and ``/`` are replaced with
                ``""`` and ``"_"`` respectively to keep the filename safe.
            start:  ISO date string (e.g. ``"2010-01-01"``).
            end:    ISO date string (e.g. ``"2025-01-01"``).
            suffix: Optional label inserted between the ticker and date parts
                (e.g. ``"close"`` → ``GSPC_close_2010-01-01_2025-01-01.parquet``).

        Returns:
            ``Path`` under ``self.cache_dir``, filesystem-safe on all
            platforms.
        """
        safe = ticker.replace("^", "").replace("/", "_")
        parts = [safe, start, end]
        if suffix:
            parts.insert(1, suffix)
        return self.cache_dir / f"{'_'.join(parts)}.parquet"

    def _validate(self, df: pd.DataFrame, ticker: str) -> None:
        """Validate a freshly-downloaded OHLCV DataFrame before caching.

        Checks for missing columns, all-NaN columns, minimum row count, and
        large calendar-day gaps in the index.

        Args:
            df: DataFrame with columns ``{"Open", "High", "Low", "Close",
                "Volume"}`` on the raw price / volume scale.
            ticker: Symbol string — used only in error and warning messages.

        Raises:
            ValueError: If any required column is absent, any column is
                entirely NaN, or the row count is below ``_MIN_ROWS``.

        Notes:
            Logs a warning (does not raise) for gaps exceeding
            ``_MAX_GAP_DAYS`` consecutive calendar days.
        """
        # Check required columns
        missing = _REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{ticker}: missing columns {missing}")

        # Check for NaN-only columns
        nan_cols = [c for c in df.columns if df[c].isna().all()]
        if nan_cols:
            raise ValueError(f"{ticker}: all-NaN columns {nan_cols}")

        # Minimum row count
        if len(df) < _MIN_ROWS:
            raise ValueError(
                f"{ticker}: only {len(df)} rows (need >= {_MIN_ROWS})"
            )

        # Large trading-day gaps
        if isinstance(df.index, pd.DatetimeIndex):
            gaps = df.index.to_series().diff().dt.days
            big = gaps[gaps > _MAX_GAP_DAYS]
            if not big.empty:
                logger.warning(
                    "{}: {} gap(s) > {} trading days — largest {} days",
                    ticker,
                    len(big),
                    _MAX_GAP_DAYS,
                    int(big.max()),
                )
