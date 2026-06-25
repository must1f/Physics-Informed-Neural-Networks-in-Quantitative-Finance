"""Tests for training result containers."""
import pytest

from src.training.result import EpochMetrics, TrainingResult


def test_epoch_metrics_fields():
    m = EpochMetrics(
        epoch=1,
        train_loss=0.5,
        val_loss=0.6,
        train_breakdown={"data": 0.4, "gbm": 0.1, "total": 0.5},
        val_breakdown={"data": 0.5, "gbm": 0.1, "total": 0.6},
        physics_scale=0.5,
        lr=0.001,
    )
    assert m.epoch == 1
    assert m.train_loss == 0.5
    assert m.val_loss == 0.6


def test_training_result_best_epoch():
    r = TrainingResult(model_name="lstm", ticker="AAPL")
    r.history.append(EpochMetrics(epoch=0, train_loss=1.0, val_loss=0.8))
    r.history.append(EpochMetrics(epoch=1, train_loss=0.5, val_loss=0.4))
    r.history.append(EpochMetrics(epoch=2, train_loss=0.3, val_loss=0.5))
    assert r.best_epoch == 1
    assert r.best_val_loss == pytest.approx(0.4)


def test_training_result_to_dict():
    r = TrainingResult(model_name="gbm_pinn", ticker="AAPL")
    r.history.append(EpochMetrics(epoch=0, train_loss=1.0, val_loss=0.8))
    r.test_metrics = {"rmse": 0.05, "sharpe": 1.2}
    d = r.to_dict()
    assert d["model_name"] == "gbm_pinn"
    assert d["ticker"] == "AAPL"
    assert d["test_metrics"]["rmse"] == 0.05
    assert len(d["history"]) == 1


def test_training_result_empty_history():
    r = TrainingResult(model_name="lstm", ticker="AAPL")
    assert r.best_epoch is None
    assert r.best_val_loss is None
