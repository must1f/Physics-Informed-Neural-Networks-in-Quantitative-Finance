"""Smoke tests: run_experiment for representative models from each tier.

Each test runs 2 epochs on synthetic data. The goal is not loss reduction
but proving the full pipeline (data -> model -> loss -> trainer -> checkpoint)
does not crash for any model type.

Note: ``bs_pinn`` and ``global_pinn`` are excluded because the
``BlackScholesConstraint`` requires autograd-enriched metadata
(``inputs`` with ``requires_grad=True``, ``predictions_next``,
``price_mean/std``, ``target_mean/std``, ``volatilities``) that the
standard ``TimeSeriesDataset`` collate does not produce. This is a
known data-pipeline gap documented in the Phase 4 plan — not a trainer
bug.
"""
import pytest
import numpy as np
import pandas as pd

from src.training.config import TrainingConfig
from src.training.runner import run_experiment
from src.training.result import TrainingResult


def _synthetic_df(n=200):
    """Create a synthetic DataFrame matching the data pipeline output."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Close": close,
            "log_return": np.concatenate(
                [[0.0], np.log(close[1:] / close[:-1])]
            ),
            "simple_return": np.concatenate(
                [[0.0], close[1:] / close[:-1] - 1]
            ),
            "rolling_volatility_5": rng.normal(0.02, 0.005, n),
            "rolling_volatility_20": rng.normal(0.02, 0.005, n),
            "momentum_5": rng.normal(0, 0.01, n),
            "momentum_20": rng.normal(0, 0.01, n),
            "rsi_14": rng.uniform(30, 70, n),
            "macd": rng.normal(0, 0.5, n),
            "macd_signal": rng.normal(0, 0.5, n),
            "bollinger_upper": close + 2,
            "bollinger_lower": close - 2,
            "atr_14": rng.uniform(0.5, 2.0, n),
            "volume_normalized": rng.uniform(-1, 1, n),
            "close_normalized": rng.uniform(-1, 1, n),
        },
        index=dates,
    )


_CFG = TrainingConfig(
    epochs=2,
    batch_size=16,
    patience=100,
    hidden_dim=16,
    num_layers=1,
    sequence_length=20,
    warmup_epochs=1,
    warmup_strategy="linear",
)


@pytest.fixture
def df():
    return _synthetic_df()


# Tier 2 — Neural baselines
@pytest.mark.parametrize(
    "model_name",
    ["lstm", "gru", "bilstm", "attention_lstm", "transformer"],
)
def test_baseline_smoke(model_name, df, tmp_path):
    result = run_experiment(
        config=_CFG, model_name=model_name, dataframe=df,
        checkpoint_dir=tmp_path,
    )
    assert isinstance(result, TrainingResult)
    assert len(result.history) == 2


# Tier 3 — Core PINNs (excluding bs_pinn and global_pinn — see docstring)
@pytest.mark.parametrize(
    "model_name",
    ["baseline_pinn", "gbm_pinn", "ou_pinn", "gbm_ou_pinn"],
)
def test_core_pinn_smoke(model_name, df, tmp_path):
    result = run_experiment(
        config=_CFG, model_name=model_name, dataframe=df,
        checkpoint_dir=tmp_path,
    )
    assert isinstance(result, TrainingResult)
    assert len(result.history) == 2


# Tier 4 — Novel architectures
@pytest.mark.parametrize(
    "model_name",
    ["hawkes_pinn", "hawkes_ou_pinn", "stacked_pinn", "residual_pinn"],
)
def test_novel_pinn_smoke(model_name, df, tmp_path):
    result = run_experiment(
        config=_CFG, model_name=model_name, dataframe=df,
        checkpoint_dir=tmp_path,
    )
    assert isinstance(result, TrainingResult)
    assert len(result.history) == 2
