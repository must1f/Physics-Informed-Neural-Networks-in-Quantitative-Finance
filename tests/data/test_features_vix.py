# tests/data/test_features_vix.py
import numpy as np
import pandas as pd
import pytest
from src.data.features import compute_features


def _make_ohlcv(n=300):
    idx = pd.date_range("2015-01-02", periods=n, freq="B")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(42).normal(0, 1, n)), index=idx)
    return pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low":  close * 0.995,
        "Close": close,
        "Volume": 1_000_000,
    })


def _make_vix(idx):
    return pd.Series(15.0 + np.random.default_rng(7).normal(0, 2, len(idx)), index=idx, name="Close")


def test_vix_columns_added_when_supplied():
    ohlcv = _make_ohlcv()
    df = compute_features(ohlcv)
    vix = _make_vix(df.index)
    result = compute_features(ohlcv, vix=vix)
    assert "vix_level" in result.columns
    assert "vix_change" in result.columns


def test_no_vix_columns_without_vix():
    ohlcv = _make_ohlcv()
    result = compute_features(ohlcv)
    assert "vix_level" not in result.columns
    assert "vix_change" not in result.columns


def test_vix_level_is_ratio():
    """vix_level = VIX_t / rolling_mean(VIX, 20) — must be positive."""
    ohlcv = _make_ohlcv()
    df = compute_features(ohlcv)
    vix = _make_vix(df.index)
    result = compute_features(ohlcv, vix=vix)
    assert (result["vix_level"].dropna() > 0).all()


def test_vix_change_is_log_return():
    """vix_change = log(VIX_t / VIX_{t-1}) — constant VIX → change is 0."""
    ohlcv = _make_ohlcv()
    df = compute_features(ohlcv)
    vix = pd.Series([20.0] * len(df), index=df.index, name="Close")
    result = compute_features(ohlcv, vix=vix)
    assert np.allclose(result["vix_change"].dropna(), 0.0, atol=1e-9)


def test_vix_aligned_to_gspc_index():
    """Missing VIX dates must not produce NaN in output (ffill/bfill applied)."""
    ohlcv = _make_ohlcv(300)
    df = compute_features(ohlcv)
    vix_full = _make_vix(df.index)
    vix_sparse = vix_full.drop(vix_full.index[5:15])
    result = compute_features(ohlcv, vix=vix_sparse)
    assert result["vix_level"].isna().sum() == 0
    assert result["vix_change"].isna().sum() == 0


# ── TNX helpers ──────────────────────────────────────────────────────────────

def _make_tnx(idx):
    rng = np.random.default_rng(99)
    return pd.Series(
        np.clip(4.0 + rng.normal(0, 0.5, len(idx)), 0.5, 10.0),
        index=idx,
        name="Close",
    )


def test_tnx_columns_added_when_supplied():
    ohlcv = _make_ohlcv()
    df = compute_features(ohlcv)
    tnx = _make_tnx(df.index)
    result = compute_features(ohlcv, tnx=tnx)
    assert "tnx_level" in result.columns
    assert "tnx_change" in result.columns


def test_no_tnx_columns_without_tnx():
    ohlcv = _make_ohlcv()
    result = compute_features(ohlcv)
    assert "tnx_level" not in result.columns
    assert "tnx_change" not in result.columns


def test_tnx_level_is_positive():
    """tnx_level = TNX_t / rolling_mean(TNX, 20) — must always be > 0."""
    ohlcv = _make_ohlcv()
    df = compute_features(ohlcv)
    tnx = _make_tnx(df.index)
    result = compute_features(ohlcv, tnx=tnx)
    assert (result["tnx_level"].dropna() > 0).all()


def test_tnx_change_constant_yields_zero():
    """Constant TNX → log change is identically 0."""
    ohlcv = _make_ohlcv()
    df = compute_features(ohlcv)
    tnx = pd.Series([4.5] * len(df), index=df.index, name="Close")
    result = compute_features(ohlcv, tnx=tnx)
    assert np.allclose(result["tnx_change"].dropna(), 0.0, atol=1e-9)


def test_tnx_and_vix_independent():
    """Supplying both vix and tnx produces all five macro columns."""
    ohlcv = _make_ohlcv()
    df = compute_features(ohlcv)
    vix = _make_vix(df.index)
    tnx = _make_tnx(df.index)
    result = compute_features(ohlcv, vix=vix, tnx=tnx)
    for col in ("vix_level", "vix_change", "vol_premium", "tnx_level", "tnx_change"):
        assert col in result.columns, f"missing {col}"
