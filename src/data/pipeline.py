"""Data loading pipeline: fetch OHLCV + optional aux series, compute features."""
from __future__ import annotations

import pandas as pd

from src.data.features import compute_features
from src.data.fetcher import DataFetcher
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_features(
    config,
    cache_dir: str = "data/cache",
) -> pd.DataFrame:
    """Fetch market data and compute all features, including VIX when configured.

    Inputs:
        config: TrainingConfig instance. Reads ``config.tickers[0]``,
            ``config.start_date``, ``config.end_date``, and
            ``config.aux_tickers`` (list[str], may be empty or absent).
        cache_dir: local directory for parquet cache files.

    Returns:
        pd.DataFrame with DatetimeIndex and all engineered feature columns
        (16 base features + ``vix_level``/``vix_change``/``vol_premium`` when
        ``"^VIX"`` is in ``config.aux_tickers``, plus ``tnx_level``/
        ``tnx_change`` when ``"^TNX"`` is also present — 19 or 21 columns
        respectively). Warmup NaN rows already dropped.
    """
    fetcher = DataFetcher(cache_dir=cache_dir)
    ticker = config.tickers[0]

    raw = fetcher.fetch(ticker, config.start_date, config.end_date)
    logger.info("Loaded {} rows for {}", len(raw), ticker)

    vix_series: pd.Series | None = None
    tnx_series: pd.Series | None = None
    for aux in getattr(config, "aux_tickers", []):
        if aux == "^VIX":
            vix_series = fetcher.fetch_close(aux, config.start_date, config.end_date)
            logger.info("Loaded VIX series: {} rows", len(vix_series))
        elif aux == "^TNX":
            tnx_series = fetcher.fetch_close(aux, config.start_date, config.end_date)
            logger.info("Loaded TNX series: {} rows", len(tnx_series))

    df = compute_features(raw, vix=vix_series, tnx=tnx_series)
    logger.info("Feature matrix: {} rows x {} cols", *df.shape)
    return df
