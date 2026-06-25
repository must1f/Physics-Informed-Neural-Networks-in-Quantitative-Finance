import numpy as np
import pandas as pd
import torch
import pytest
from pathlib import Path
from src.training.config import load_config
from src.training.runner import run_experiment

CONFIG_PATH = Path(__file__).parents[2] / "configs" / "dissertation.yaml"


@pytest.fixture
def tiny_df():
    rng = np.random.default_rng(0)
    n = 500
    prices = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    log_ret = np.log(prices[1:] / prices[:-1])
    log_ret = np.append(log_ret, log_ret[-1])
    sr = np.exp(log_ret) - 1
    s = pd.Series
    open_prices = prices * (1 + rng.normal(0, 0.001, n))
    return pd.DataFrame({
        "Close": prices,
        "log_return": log_ret,
        "simple_return": sr,
        "rolling_volatility_5":  s(log_ret).rolling(5).std().fillna(0).values,
        "rolling_volatility_20": s(log_ret).rolling(20).std().fillna(0).values,
        "momentum_5":  s(log_ret).rolling(5).sum().fillna(0).values,
        "momentum_20": s(log_ret).rolling(20).sum().fillna(0).values,
        "rsi_14":          np.full(n, 50.0),
        "macd":            np.zeros(n),
        "macd_signal":     np.zeros(n),
        "bollinger_upper": prices + 2,
        "bollinger_lower": prices - 2,
        "atr_14":             np.ones(n),
        "volume_normalized":  np.ones(n),
        "close_normalized":   (prices - prices.mean()) / prices.std(),
        # VIX/TNX macro features required by dissertation.yaml feature list
        "vix_level":   np.full(n, 1.0),
        "vix_change":  np.zeros(n),
        "vol_premium": s(log_ret).rolling(20).std().fillna(0).values - 0.15,
        "tnx_level":   np.full(n, 1.0),
        "tnx_change":  np.zeros(n),
        "overnight_gap": np.concatenate([[0.0], np.log(open_prices[1:] / prices[:-1])]),
    })


def test_ou_pinn_run_saves_checkpoint(tiny_df, tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.epochs = 2
    cfg.patience = 100
    result = run_experiment(cfg, "ou_pinn", tiny_df, checkpoint_dir=tmp_path, seed=42)
    assert result.best_val_loss is not None
    assert (tmp_path / "ou_pinn.pt").exists()


def test_gbm_ou_pinn_run_saves_checkpoint(tiny_df, tmp_path):
    cfg = load_config(CONFIG_PATH)
    cfg.epochs = 2
    cfg.patience = 100
    result = run_experiment(cfg, "gbm_ou_pinn", tiny_df, checkpoint_dir=tmp_path, seed=42)
    assert result.best_val_loss is not None
