# tests/data/test_pipeline.py
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.data.pipeline import load_features


def _make_config(aux_tickers=None):
    cfg = MagicMock()
    cfg.tickers = ["^GSPC"]
    cfg.start_date = "2015-01-01"
    cfg.end_date = "2020-12-31"
    cfg.aux_tickers = aux_tickers or []
    return cfg


def _make_ohlcv(n=300):
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    close = pd.Series(100.0 + np.arange(n) * 0.1, index=idx)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": 1_000_000,
    })


def _make_vix(n=300):
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    return pd.Series(15.0 + np.zeros(n), index=idx, name="Close")


def test_load_features_no_vix(tmp_path):
    ohlcv = _make_ohlcv()
    with patch("src.data.pipeline.DataFetcher") as MockFetcher:
        MockFetcher.return_value.fetch.return_value = ohlcv
        df = load_features(_make_config(), cache_dir=str(tmp_path))
    assert "log_return" in df.columns
    assert "vix_level" not in df.columns


def test_load_features_with_vix(tmp_path):
    ohlcv = _make_ohlcv()
    vix = _make_vix()
    with patch("src.data.pipeline.DataFetcher") as MockFetcher:
        instance = MockFetcher.return_value
        instance.fetch.return_value = ohlcv
        instance.fetch_close.return_value = vix
        df = load_features(_make_config(aux_tickers=["^VIX"]), cache_dir=str(tmp_path))
    assert "vix_level" in df.columns
    assert "vix_change" in df.columns


def test_load_features_unknown_aux_ignored(tmp_path):
    """Aux tickers that are not ^VIX are silently skipped."""
    ohlcv = _make_ohlcv()
    with patch("src.data.pipeline.DataFetcher") as MockFetcher:
        MockFetcher.return_value.fetch.return_value = ohlcv
        df = load_features(_make_config(aux_tickers=["^UNKNOWN"]), cache_dir=str(tmp_path))
    assert "vix_level" not in df.columns
