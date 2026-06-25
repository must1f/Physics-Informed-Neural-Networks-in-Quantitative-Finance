"""Tests for PhysicsScheduler — curriculum lambda warmup."""
import math
import pytest

from src.training.scheduler import PhysicsScheduler


class TestLinearWarmup:
    def test_zero_at_epoch_zero(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="linear")
        assert s.get_scale(0) == 0.0

    def test_one_at_warmup_end(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="linear")
        assert s.get_scale(20) == 1.0

    def test_half_at_midpoint(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="linear")
        assert s.get_scale(10) == pytest.approx(0.5)

    def test_clamped_above_warmup(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="linear")
        assert s.get_scale(100) == 1.0


class TestCosineWarmup:
    def test_zero_at_epoch_zero(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="cosine")
        assert s.get_scale(0) == pytest.approx(0.0, abs=1e-10)

    def test_one_at_warmup_end(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="cosine")
        assert s.get_scale(20) == pytest.approx(1.0, abs=1e-10)

    def test_monotonically_increasing(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="cosine")
        scales = [s.get_scale(e) for e in range(21)]
        for i in range(1, len(scales)):
            assert scales[i] >= scales[i - 1]


class TestStepWarmup:
    def test_zero_before_warmup(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="step")
        assert s.get_scale(0) == 0.0
        assert s.get_scale(19) == 0.0

    def test_one_at_warmup(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="step")
        assert s.get_scale(20) == 1.0

    def test_one_after_warmup(self):
        s = PhysicsScheduler(warmup_epochs=20, strategy="step")
        assert s.get_scale(50) == 1.0


def test_zero_warmup_always_returns_one():
    s = PhysicsScheduler(warmup_epochs=0, strategy="linear")
    assert s.get_scale(0) == 1.0


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="Unknown"):
        PhysicsScheduler(warmup_epochs=10, strategy="invalid")
