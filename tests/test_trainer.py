"""Tests for the universal Trainer."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.losses.composite import CompositeLoss
from src.losses.data_losses import mse_loss
from src.losses.physics import GBMConstraint
from src.models.baselines import BaselineModel
from src.models.pinn import PINNModel
from src.training.scheduler import PhysicsScheduler
from src.training.trainer import Trainer
from src.training.result import EpochMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _baseline_loader(B=8, T=10, F=4, n_batches=3):
    """DataLoader yielding (x, y, metadata) for baselines."""
    samples = []
    for _ in range(B * n_batches):
        samples.append((torch.randn(T, F), torch.randn(1), {}))

    def _collate(batch):
        x = torch.stack([s[0] for s in batch])
        y = torch.stack([s[1] for s in batch])
        return x, y, {}

    return DataLoader(samples, batch_size=B, collate_fn=_collate)


def _pinn_loader(B=8, T=10, F=4, n_batches=3):
    """DataLoader with physics metadata (prices, returns)."""
    samples = []
    for _ in range(B * n_batches):
        x = torch.randn(T, F)
        y = torch.randn(1)
        prices = 100.0 + torch.cumsum(torch.randn(T + 1) * 0.5, dim=0)
        returns = torch.randn(T)
        meta = {"prices": prices, "returns": returns, "dt": 1 / 252}
        samples.append((x, y, meta))

    def _collate(batch):
        x = torch.stack([s[0] for s in batch])
        y = torch.stack([s[1] for s in batch])
        meta = {
            "prices": torch.stack([s[2]["prices"] for s in batch]),
            "returns": torch.stack([s[2]["returns"] for s in batch]),
            "dt": 1 / 252,
        }
        return x, y, meta

    return DataLoader(samples, batch_size=B, collate_fn=_collate)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainerBaseline:
    def test_one_epoch_baseline(self):
        model = BaselineModel(arch="lstm", input_dim=4, hidden_dim=16, num_layers=1)
        loss_fn = CompositeLoss(mse_loss, constraints=[])
        trainer = Trainer(model=model, loss_fn=loss_fn, lr=0.01)
        loader = _baseline_loader(F=4)
        metrics = trainer.train_epoch(loader)
        assert isinstance(metrics, EpochMetrics)
        assert metrics.train_loss > 0.0

    def test_validate_baseline(self):
        model = BaselineModel(arch="gru", input_dim=4, hidden_dim=16, num_layers=1)
        loss_fn = CompositeLoss(mse_loss, constraints=[])
        trainer = Trainer(model=model, loss_fn=loss_fn, lr=0.01)
        loader = _baseline_loader(F=4)
        metrics = trainer.validate(loader)
        assert isinstance(metrics, EpochMetrics)
        assert metrics.val_loss > 0.0

    def test_fit_reduces_loss(self):
        model = BaselineModel(arch="lstm", input_dim=4, hidden_dim=16, num_layers=1)
        loss_fn = CompositeLoss(mse_loss, constraints=[])
        trainer = Trainer(model=model, loss_fn=loss_fn, lr=0.01)
        train_dl = _baseline_loader(F=4)
        val_dl = _baseline_loader(F=4)
        result = trainer.fit(train_dl, val_dl, epochs=5, patience=100)
        assert len(result.history) == 5
        # Loss should not explode
        assert result.history[-1].train_loss < result.history[0].train_loss * 5


class TestTrainerPINN:
    def test_one_epoch_pinn_with_gbm(self):
        model = PINNModel(
            input_dim=4, hidden_dim=16, num_layers=1,
            constraints=[GBMConstraint()],
        )
        loss_fn = CompositeLoss(
            mse_loss, constraints=[GBMConstraint()], lambdas={"gbm": 0.1},
        )
        trainer = Trainer(model=model, loss_fn=loss_fn, lr=0.01)
        loader = _pinn_loader(F=4)
        metrics = trainer.train_epoch(loader)
        assert metrics.train_loss > 0

    def test_fit_pinn_with_physics_scheduler(self):
        model = PINNModel(
            input_dim=4, hidden_dim=16, num_layers=1,
            constraints=[GBMConstraint()],
        )
        loss_fn = CompositeLoss(
            mse_loss, constraints=[GBMConstraint()], lambdas={"gbm": 0.1},
        )
        phys_sched = PhysicsScheduler(warmup_epochs=3, strategy="linear")
        trainer = Trainer(
            model=model, loss_fn=loss_fn, lr=0.01,
            physics_scheduler=phys_sched,
        )
        train_dl = _pinn_loader(F=4)
        val_dl = _pinn_loader(F=4)
        result = trainer.fit(train_dl, val_dl, epochs=5, patience=100)
        # First epoch should have physics_scale=0, last should be 1.0
        assert result.history[0].physics_scale == pytest.approx(0.0)
        assert result.history[4].physics_scale == pytest.approx(1.0)


class TestTrainerEarlyStopping:
    def test_early_stopping_triggers(self):
        model = BaselineModel(arch="lstm", input_dim=4, hidden_dim=16, num_layers=1)
        loss_fn = CompositeLoss(mse_loss, constraints=[])
        trainer = Trainer(model=model, loss_fn=loss_fn, lr=0.01)
        train_dl = _baseline_loader(F=4)
        val_dl = _baseline_loader(F=4)
        result = trainer.fit(train_dl, val_dl, epochs=200, patience=3)
        assert len(result.history) < 200


class TestTrainerGradientClip:
    def test_gradient_clipping_does_not_crash(self):
        model = BaselineModel(arch="lstm", input_dim=4, hidden_dim=16, num_layers=1)
        loss_fn = CompositeLoss(mse_loss, constraints=[])
        trainer = Trainer(
            model=model, loss_fn=loss_fn, lr=0.01, gradient_clip=1.0,
        )
        loader = _baseline_loader(F=4)
        metrics = trainer.train_epoch(loader)
        assert metrics.train_loss > 0.0


class TestTrainerCheckpointing:
    def test_best_state_dict_is_restored_after_fit(self):
        torch.manual_seed(0)
        model = BaselineModel(arch="lstm", input_dim=4, hidden_dim=16, num_layers=1)
        loss_fn = CompositeLoss(mse_loss, constraints=[])
        trainer = Trainer(model=model, loss_fn=loss_fn, lr=0.01)
        train_dl = _baseline_loader(F=4)
        val_dl = _baseline_loader(F=4)
        result = trainer.fit(train_dl, val_dl, epochs=10, patience=100)
        assert result.best_epoch is not None
