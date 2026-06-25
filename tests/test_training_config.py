"""Tests for TrainingConfig loading from dissertation.yaml."""
import pytest
from pathlib import Path

from src.training.config import TrainingConfig, load_config


def test_load_config_from_yaml():
    cfg = load_config(Path("configs/dissertation.yaml"))
    assert isinstance(cfg, TrainingConfig)
    assert cfg.epochs == 200
    assert cfg.batch_size == 64
    assert cfg.learning_rate == 0.001
    assert cfg.patience == 20
    assert cfg.gradient_clip == 0.5
    assert cfg.scheduler == "plateau"
    assert cfg.scheduler_patience == 10
    assert cfg.scheduler_factor == 0.5


def test_config_has_model_defaults():
    cfg = load_config(Path("configs/dissertation.yaml"))
    assert cfg.hidden_dim == 128
    assert cfg.num_layers == 2
    assert cfg.dropout == 0.2


def test_config_has_physics_settings():
    cfg = load_config(Path("configs/dissertation.yaml"))
    assert cfg.physics_lambdas == {
        "gbm": 0.1, "ou": 0.1, "bs": 0.01, "langevin": 0.05,
        "hawkes": 0.1, "hawkes_v2": 0.1,
    }
    assert cfg.warmup_epochs == 20
    assert cfg.warmup_strategy == "cosine"


def test_config_has_data_settings():
    cfg = load_config(Path("configs/dissertation.yaml"))
    assert cfg.sequence_length == 60
    assert cfg.split_ratios == (0.70, 0.15, 0.15)
    assert "^GSPC" in cfg.tickers


def test_config_override():
    cfg = load_config(
        Path("configs/dissertation.yaml"),
        overrides={"epochs": 10, "batch_size": 32},
    )
    assert cfg.epochs == 10
    assert cfg.batch_size == 32
    # non-overridden values unchanged
    assert cfg.learning_rate == 0.001
