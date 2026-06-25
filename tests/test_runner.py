"""Tests for the experiment runner."""
import json
import pytest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.training.runner import run_experiment
from src.training.config import TrainingConfig
from src.training.result import TrainingResult


def _fake_dataframe(n=300):
    """Create a minimal DataFrame matching the data pipeline output."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Close": close,
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
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


class TestRunExperiment:
    def test_baseline_lstm_runs_to_completion(self, tmp_path):
        cfg = TrainingConfig(
            epochs=3, batch_size=16, patience=100,
            hidden_dim=16, num_layers=1, sequence_length=20,
        )
        df = _fake_dataframe(200)
        result = run_experiment(
            config=cfg, model_name="lstm", dataframe=df,
            checkpoint_dir=tmp_path,
        )
        assert isinstance(result, TrainingResult)
        assert result.model_name == "lstm"
        assert len(result.history) == 3
        assert result.checkpoint_path is not None
        assert Path(result.checkpoint_path).exists()
        assert result.test_metrics
        assert "rmse" in result.test_metrics
        assert "sharpe" in result.test_metrics

    def test_pinn_gbm_runs_to_completion(self, tmp_path):
        cfg = TrainingConfig(
            epochs=3, batch_size=16, patience=100,
            hidden_dim=16, num_layers=1, sequence_length=20,
            warmup_epochs=2, warmup_strategy="linear",
        )
        df = _fake_dataframe(200)
        result = run_experiment(
            config=cfg, model_name="gbm_pinn", dataframe=df,
            checkpoint_dir=tmp_path,
        )
        assert isinstance(result, TrainingResult)
        assert result.model_name == "gbm_pinn"
        assert len(result.history) == 3
        # Physics warmup: epoch 0 scale should be 0
        assert result.history[0].physics_scale == pytest.approx(0.0)

    def test_stacked_pinn_runs(self, tmp_path):
        cfg = TrainingConfig(
            epochs=2, batch_size=16, patience=100,
            hidden_dim=16, num_layers=1, sequence_length=20,
        )
        df = _fake_dataframe(200)
        result = run_experiment(
            config=cfg, model_name="stacked_pinn", dataframe=df,
            checkpoint_dir=tmp_path,
        )
        assert isinstance(result, TrainingResult)
        assert result.model_name == "stacked_pinn"

    def test_unknown_model_raises(self, tmp_path):
        cfg = TrainingConfig(epochs=1, sequence_length=20)
        df = _fake_dataframe(200)
        with pytest.raises(ValueError, match="Unknown model"):
            run_experiment(
                config=cfg, model_name="nonexistent", dataframe=df,
                checkpoint_dir=tmp_path,
            )

    def test_checkpoint_is_loadable(self, tmp_path):
        cfg = TrainingConfig(
            epochs=2, batch_size=16, patience=100,
            hidden_dim=16, num_layers=1, sequence_length=20,
        )
        df = _fake_dataframe(200)
        result = run_experiment(
            config=cfg, model_name="lstm", dataframe=df,
            checkpoint_dir=tmp_path,
        )
        ckpt = torch.load(result.checkpoint_path, weights_only=True)
        assert "model_state_dict" in ckpt
        assert "config" in ckpt

    def test_result_json_is_saved(self, tmp_path):
        cfg = TrainingConfig(
            epochs=2, batch_size=16, patience=100,
            hidden_dim=16, num_layers=1, sequence_length=20,
        )
        df = _fake_dataframe(200)
        result = run_experiment(
            config=cfg, model_name="lstm", dataframe=df,
            checkpoint_dir=tmp_path,
        )
        json_path = tmp_path / "lstm_result.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["model_name"] == "lstm"
