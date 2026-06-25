"""F3 regression test: best-epoch selection must use data-only val loss.

Constructs a synthetic training history where the composite ``val_loss``
and the data-only ``val_breakdown["data"]`` disagree on which epoch is
"best". Asserts the data-only epoch wins.

This locks the fix that eliminates the arbitrary tie-breaking observed
in ``notebooks/4_core_pinns_extended.ipynb`` (2026-04-19 audit), where
multiple epochs tied on composite ``val_loss = data + Σλᵢ·physicsᵢ``
at the same plateau and best-epoch selection picked one essentially at
random. The data-only criterion breaks those ties deterministically.
"""
from src.training.result import EpochMetrics, TrainingResult


def test_best_epoch_prefers_data_loss_over_composite():
    # Epoch 1: high data, low physics → low composite.
    # Epoch 2: low data, high physics → high composite.
    # Correct choice under F3: epoch 2 (lowest data loss).
    history = [
        EpochMetrics(
            epoch=1,
            train_loss=0.0,
            val_loss=0.10,
            train_breakdown={"data": 0.20, "total": 0.10},
            val_breakdown={"data": 0.20, "physics_term": 0.10, "total": 0.10},
            physics_scale=1.0,
            lr=1e-3,
        ),
        EpochMetrics(
            epoch=2,
            train_loss=0.0,
            val_loss=0.15,
            train_breakdown={"data": 0.05, "total": 0.15},
            val_breakdown={"data": 0.05, "physics_term": 0.10, "total": 0.15},
            physics_scale=1.0,
            lr=1e-3,
        ),
    ]
    result = TrainingResult(model_name="test", history=history)
    assert result.best_epoch == 2
    assert result.best_val_loss == 0.05


def test_best_epoch_falls_back_to_val_loss_when_no_data_key():
    # Classical / non-PINN runs may have no ``"data"`` breakdown key.
    # Behaviour must remain: use composite ``val_loss`` directly.
    history = [
        EpochMetrics(
            epoch=1,
            train_loss=0.0,
            val_loss=0.30,
            train_breakdown={},
            val_breakdown={},
            physics_scale=1.0,
            lr=1e-3,
        ),
        EpochMetrics(
            epoch=2,
            train_loss=0.0,
            val_loss=0.20,
            train_breakdown={},
            val_breakdown={},
            physics_scale=1.0,
            lr=1e-3,
        ),
    ]
    result = TrainingResult(model_name="test", history=history)
    assert result.best_epoch == 2
    assert result.best_val_loss == 0.20


def test_best_epoch_none_on_empty_history():
    result = TrainingResult(model_name="test", history=[])
    assert result.best_epoch is None
    assert result.best_val_loss is None
