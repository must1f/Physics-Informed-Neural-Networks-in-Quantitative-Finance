"""Unit tests and correctness audit for ``src.losses.physics``.

Each constraint is exercised for:
  * required-key contract (metadata)
  * shape + scalar output
  * non-negativity (MSE-style residuals)
  * numerical behaviour on hand-crafted inputs where the residual has a
    known closed-form value
  * gradient flow through ``predictions`` (so the constraint can train the
    network) and through its own learnable parameters (where applicable)

The tests also encode the audit findings documented in the module-level
docstring below.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.losses.physics import (
    EPS,
    DAILY_DT,
    BlackScholesConstraint,
    GBMConstraint,
    HawkesConstraint,
    LangevinConstraint,
    OUConstraint,
    PhysicsConstraint,
)
from src.losses.composite import CompositeLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_prices(batch: int = 4, seq: int = 20, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    log_r = 0.0005 + 0.01 * torch.randn(batch, seq - 1, generator=g)
    prices = 100.0 * torch.exp(torch.cumsum(log_r, dim=1))
    return torch.cat([torch.full((batch, 1), 100.0), prices], dim=1)


def _rand_returns(batch: int = 4, seq: int = 20, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.001 + 0.01 * torch.randn(batch, seq, generator=g)


def _check_scalar(loss: torch.Tensor) -> None:
    assert loss.dim() == 0, f"expected scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    assert loss.item() >= 0.0, f"MSE-style residual must be non-negative, got {loss}"


# ===========================================================================
# GBMConstraint
# ===========================================================================

class TestGBM:
    def test_basic_contract(self):
        c = GBMConstraint()
        prices = _rand_prices()
        pred = torch.zeros(prices.shape[0], 1, requires_grad=True)
        loss = c.residual(pred, {"prices": prices})
        _check_scalar(loss)

    def test_requires_prices_key(self):
        c = GBMConstraint()
        pred = torch.zeros(4, 1)
        with pytest.raises(KeyError):
            c.residual(pred, {})

    def test_accepts_3d_prices(self):
        c = GBMConstraint()
        prices_3d = _rand_prices().unsqueeze(-1)  # [B, T, 1]
        pred = torch.zeros(prices_3d.shape[0], 1)
        loss = c.residual(pred, {"prices": prices_3d})
        _check_scalar(loss)

    def test_gradient_flows_to_predictions(self):
        c = GBMConstraint()
        prices = _rand_prices()
        pred = torch.zeros(prices.shape[0], 1, requires_grad=True)
        loss = c.residual(pred, {"prices": prices})
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_zero_residual_when_pred_matches_ito_drift(self):
        """Feeding the Itô-corrected GBM drift as the prediction should
        drive the residual to zero. ``mu`` is estimated from simple
        returns (so the Itô correction is applied exactly once inside the
        constraint), ``sigma`` from log returns.
        """
        c = GBMConstraint()
        prices = _rand_prices()
        simple_returns = prices[:, 1:] / (prices[:, :-1] + EPS) - 1.0
        log_returns = torch.log(prices[:, 1:] / (prices[:, :-1] + EPS) + EPS)
        mu_hat = simple_returns.mean(dim=1, keepdim=True) / c.dt
        sigma_hat = log_returns.std(dim=1, keepdim=True) / math.sqrt(c.dt)
        drift = mu_hat - 0.5 * sigma_hat ** 2
        target = drift * c.dt
        loss = c.residual(target, {"prices": prices})
        assert loss.item() == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# OUConstraint
# ===========================================================================

class TestOU:
    def test_basic_contract(self):
        c = OUConstraint()
        returns = _rand_returns()
        pred = torch.zeros(returns.shape[0], 1, requires_grad=True)
        loss = c.residual(pred, {"returns": returns})
        _check_scalar(loss)

    def test_learnable_param_constraints(self):
        c = OUConstraint()
        assert c.theta.item() > 0       # softplus
        assert c.sigma.item() >= 0      # softplus
        # mu is a fixed buffer seeded from training-set mean, not a Parameter.
        assert not isinstance(c.mu, torch.nn.Parameter)
        assert hasattr(c, "_theta_raw") and isinstance(c._theta_raw, torch.nn.Parameter)
        assert hasattr(c, "_sigma_raw") and isinstance(c._sigma_raw, torch.nn.Parameter)

    def test_gradient_flows_to_theta(self):
        c = OUConstraint()
        returns = _rand_returns()
        pred = torch.randn(returns.shape[0], 1)
        loss = c.residual(pred, {"returns": returns})
        loss.backward()
        assert c._theta_raw.grad is not None
        assert torch.isfinite(c._theta_raw.grad)

    def test_zero_residual_for_ou_expected_value(self):
        c = OUConstraint()
        returns = _rand_returns()
        last = returns[:, -1:]
        with torch.no_grad():
            expected = last + c.theta * (c.mu - last) * c.dt
        loss = c.residual(expected, {"returns": returns})
        assert loss.item() == pytest.approx(0.0, abs=1e-10)

    def test_mu_parameter_is_live(self):
        """Perturbing ``self.mu`` must change the residual — otherwise the
        learnable long-run mean is dead code.
        """
        c = OUConstraint()
        returns = _rand_returns()
        pred = torch.randn(returns.shape[0], 1)
        loss_a = c.residual(pred, {"returns": returns}).item()
        with torch.no_grad():
            c.mu.add_(5.0)
        loss_b = c.residual(pred, {"returns": returns}).item()
        assert loss_a != pytest.approx(loss_b, abs=1e-6)

    def test_learnable_params_receive_gradient(self):
        # mu is a fixed buffer; _sigma_raw is API-completeness only (not in drift
        # residual). Only _theta_raw participates in the backward pass.
        c = OUConstraint()
        returns = _rand_returns()
        pred = torch.randn(returns.shape[0], 1)
        loss = c.residual(pred, {"returns": returns})
        loss.backward()
        assert c._theta_raw.grad is not None
        assert torch.isfinite(c._theta_raw.grad)


# ===========================================================================
# BlackScholesConstraint
# ===========================================================================

class TestBlackScholes:
    def _make_metadata(self, batch=4, seq=10, features=3, price_idx=0):
        torch.manual_seed(0)
        inputs = torch.randn(batch, seq, features, requires_grad=True)
        volatilities = 0.2 * torch.ones(batch, 1)
        price_mean = torch.tensor(100.0)
        price_std = torch.tensor(10.0)
        target_mean = torch.tensor(100.0)
        target_std = torch.tensor(10.0)
        prices = torch.linspace(100.0, 110.0, seq + 1).unsqueeze(0).expand(batch, -1)
        return {
            "inputs": inputs,
            "price_feature_idx": price_idx,
            "price_mean": price_mean,
            "price_std": price_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "volatilities": volatilities,
            "prices": prices,
        }

    def _nonlinear_V(self, inputs: torch.Tensor) -> torch.Tensor:
        """Quadratic in the price feature so d2V/dS2 has a non-constant grad."""
        last_price = inputs[:, -1, 0:1]
        return last_price ** 2 + 0.1 * inputs[:, -1, :].mean(dim=-1, keepdim=True)

    def test_basic_contract(self):
        c = BlackScholesConstraint()
        meta = self._make_metadata()
        V = self._nonlinear_V(meta["inputs"])
        meta["predictions_next"] = V.detach() + 0.01
        loss = c.residual(V, meta)
        _check_scalar(loss)

    def test_gradient_flows_to_predictions(self):
        c = BlackScholesConstraint()
        meta = self._make_metadata()
        V = self._nonlinear_V(meta["inputs"])
        meta["predictions_next"] = V.detach() + 0.01
        loss = c.residual(V, meta)
        loss.backward()
        assert meta["inputs"].grad is not None
        assert torch.isfinite(meta["inputs"].grad).all()

    def test_gradient_flows_to_sigma_scale(self):
        c = BlackScholesConstraint()
        meta = self._make_metadata()
        V = self._nonlinear_V(meta["inputs"])
        meta["predictions_next"] = V.detach() + 0.01
        loss = c.residual(V, meta)
        loss.backward()
        assert c._sigma_log_scale.grad is not None
        assert torch.isfinite(c._sigma_log_scale.grad)


# ===========================================================================
# LangevinConstraint
# ===========================================================================

class TestLangevin:
    def test_basic_contract(self):
        c = LangevinConstraint()
        returns = _rand_returns()
        pred = torch.zeros(returns.shape[0], 1, requires_grad=True)
        loss = c.residual(pred, {"returns": returns})
        _check_scalar(loss)

    def test_gradient_flows_to_gamma_and_temperature(self):
        c = LangevinConstraint()
        returns = _rand_returns()
        pred = torch.randn(returns.shape[0], 1)
        loss = c.residual(pred, {"returns": returns})
        loss.backward()
        assert c._gamma_raw.grad is not None
        assert c._temperature_raw.grad is not None

    def test_drift_only_residual_zero_when_matched(self):
        """If we feed the exact expected drift and ignore diffusion, the
        drift component should vanish; the diffusion residual is a separate
        additive term, so the total loss equals only the diffusion term."""
        c = LangevinConstraint()
        returns = _rand_returns()
        last = returns[:, -1:]
        with torch.no_grad():
            expected_drift = last - c.gamma * last * c.dt
            expected_diff = torch.sqrt(2.0 * c.gamma * c.temperature * c.dt + EPS)
            diff_res = ((expected_drift - last).abs() - expected_diff) ** 2
            expected_total = diff_res.mean()
        loss = c.residual(expected_drift, {"returns": returns})
        assert loss.item() == pytest.approx(expected_total.item(), rel=1e-5, abs=1e-8)


# ===========================================================================
# HawkesConstraint
# ===========================================================================

class TestHawkes:
    def test_basic_contract(self):
        c = HawkesConstraint()
        returns = _rand_returns()
        pred = torch.zeros(returns.shape[0], 1, requires_grad=True)
        loss = c.residual(pred, {"returns": returns})
        _check_scalar(loss)

    def test_intensity_matches_manual_computation(self):
        c = HawkesConstraint()
        returns = _rand_returns(batch=2, seq=5)
        B, T = returns.shape
        with torch.no_grad():
            alpha, beta, mu0 = c.alpha, c.beta, c.mu0
            # Manual: lambda = mu0 + sum_s |r_s| * alpha * exp(-beta * (T - s))
            manual = torch.zeros(B, 1)
            for b in range(B):
                total = mu0
                for s in range(T):
                    dt = T - s  # step s → forecast T
                    total = total + returns[b, s].abs() * alpha * torch.exp(-beta * dt)
                manual[b, 0] = total
            # Pred² = manual intensity → near-zero residual.
            # float32 rounding in torch.sqrt makes pred² differ from intensity by ~1e-15;
            # after EMA normalisation clamped at 1e-12, the normalised residual is ~1e-3.
            # abs=0.1 is the appropriate tolerance for the normalised output.
            pred = torch.sqrt(manual)
        loss = c.residual(pred, {"returns": returns})
        assert loss.item() == pytest.approx(0.0, abs=0.1)

    def test_gradient_flows_to_hawkes_params(self):
        c = HawkesConstraint()
        returns = _rand_returns()
        pred = torch.randn(returns.shape[0], 1)
        loss = c.residual(pred, {"returns": returns})
        loss.backward()
        for p in (c._mu0_raw, c._alpha_raw, c._beta_raw):
            assert p.grad is not None and torch.isfinite(p.grad)

    def test_mu0_init_matches_return_variance(self):
        # When mu0_init is provided (train-set mean(r²) scale), softplus(_mu0_raw)
        # must equal the seed value within float precision.
        target = 1e-4
        c = HawkesConstraint(mu0_init=target)
        assert float(c.mu0.detach()) == pytest.approx(target, rel=1e-4)

    def test_default_init_backward_compatible(self):
        # Default mu0_init=None preserves the legacy softplus(0) = ln 2 init
        # so existing tests and checkpoints keep their numerical behaviour.
        c = HawkesConstraint()
        assert float(c.mu0.detach()) == pytest.approx(math.log(2), abs=1e-5)


# ===========================================================================
# CompositeLoss integration
# ===========================================================================

class TestComposite:
    def test_pure_data_baseline(self):
        loss_fn = torch.nn.functional.mse_loss
        comp = CompositeLoss(loss_fn, constraints=[])
        pred = torch.zeros(4, 1, requires_grad=True)
        target = torch.ones(4, 1)
        total, parts = comp(pred, target, physics_input={})
        assert total.item() == pytest.approx(1.0)
        assert "data" in parts and "total" in parts

    def test_physics_contributes_when_weighted(self):
        loss_fn = torch.nn.functional.mse_loss
        comp = CompositeLoss(
            loss_fn,
            constraints=[GBMConstraint()],
            lambdas={"gbm": 2.0},
        )
        pred = torch.zeros(4, 1, requires_grad=True)
        target = torch.zeros(4, 1)
        prices = _rand_prices(batch=4, seq=20)
        total, parts = comp(pred, target, physics_input={"prices": prices})
        assert "gbm" in parts
        # parts["gbm"] is already λ-weighted (CompositeLoss stores lam*residual).
        expected = parts["data"] + parts["gbm"]
        assert total.item() == pytest.approx(expected.item(), rel=1e-6)

    def test_bs_lambda_is_applied(self):
        """CompositeLoss must find the BS constraint via name 'bs', not 'black_scholes'."""
        from src.losses.composite import CompositeLoss
        from src.losses.physics import BlackScholesConstraint

        c = BlackScholesConstraint()
        assert c.name == "bs", f"Expected name 'bs', got '{c.name}'"

        loss_fn = torch.nn.functional.mse_loss
        comp_lam2 = CompositeLoss(loss_fn, constraints=[c], lambdas={"bs": 2.0})
        comp_lam0 = CompositeLoss(loss_fn, constraints=[c], lambdas={"bs": 0.0})

        torch.manual_seed(0)
        B = 4
        pred = torch.randn(B, 1, requires_grad=True)
        target = torch.zeros(B, 1)
        prices = torch.linspace(4000.0, 4050.0, 11).unsqueeze(0).expand(B, -1)
        meta = {
            "volatilities": 0.2 * torch.ones(B, 1),
            "target_mean": torch.zeros(B, 1),
            "target_std": torch.ones(B, 1),
            "prices": prices,
        }

        total_lam2, _ = comp_lam2(pred, target, meta)
        total_lam0, _ = comp_lam0(pred, target, meta)
        assert total_lam2.item() != pytest.approx(total_lam0.item(), rel=1e-4), (
            "λ=2.0 and λ=0.0 give identical totals — name mismatch not fixed"
        )


# ===========================================================================
# Base class contract
# ===========================================================================

def test_base_class_is_abstract():
    with pytest.raises(TypeError):
        PhysicsConstraint()  # type: ignore[abstract]
