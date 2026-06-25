import torch
import pytest
from src.losses.physics import OUConstraint


def test_ou_mu_init_default_is_zero():
    c = OUConstraint()
    assert abs(float(c.mu.detach()) - 0.0) < 1e-7


def test_ou_mu_init_positive_drift():
    c = OUConstraint(mu_init=0.0005)
    assert abs(float(c.mu.detach()) - 0.0005) < 1e-7


def test_ou_residual_shape_and_nonneg():
    c = OUConstraint(mu_init=0.0005)
    preds = torch.randn(8, 1)
    returns = torch.randn(8, 60)
    loss = c.residual(preds, {"returns": returns})
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_ou_mu_init_shifts_residual():
    """Residual must differ when mu_init changes the OU expected value."""
    torch.manual_seed(0)
    preds = torch.randn(8, 1)
    returns = torch.zeros(8, 60)  # last return = 0; OU expected = theta*(mu - 0)*dt
    loss_zero = OUConstraint(mu_init=0.0).residual(preds, {"returns": returns})
    loss_pos  = OUConstraint(mu_init=0.5).residual(preds, {"returns": returns})
    assert abs(loss_zero.item() - loss_pos.item()) > 1e-6, (
        "mu_init=0.5 should shift the OU expected value and change the residual"
    )


def test_ou_mu_init_negative():
    """mu_init accepts negative values (valid for short-biased assets)."""
    c = OUConstraint(mu_init=-0.001)
    assert abs(float(c.mu.detach()) - (-0.001)) < 1e-7


from src.losses.physics import BlackScholesConstraint, LangevinConstraint


def _check_scalar(loss):
    assert loss.ndim == 0
    assert loss.item() >= 0.0


class TestBlackScholes:
    """Tests for the physical-measure drift form of BlackScholesConstraint."""

    def _meta(self, batch: int = 4, seq: int = 10) -> dict:
        # Realistic ascending prices so simple_returns are well-conditioned.
        prices = torch.linspace(4000.0, 4100.0, seq + 1).unsqueeze(0).expand(batch, -1)
        return {
            "volatilities": 0.2 * torch.ones(batch, 1),
            "target_mean": torch.zeros(batch, 1),
            "target_std": torch.ones(batch, 1),
            "prices": prices,
        }

    def test_basic_contract(self):
        c = BlackScholesConstraint()
        pred = torch.zeros(4, 1, requires_grad=True)
        loss = c.residual(pred, self._meta())
        _check_scalar(loss)

    def test_requires_volatilities_key(self):
        c = BlackScholesConstraint()
        pred = torch.zeros(4, 1)
        with pytest.raises(KeyError):
            c.residual(pred, {"target_mean": torch.zeros(4, 1), "target_std": torch.ones(4, 1)})

    def test_gradient_flows_to_predictions(self):
        """Gradient must reach predictions so the LSTM learns."""
        c = BlackScholesConstraint()
        pred = torch.randn(4, 1, requires_grad=True)
        loss = c.residual(pred, self._meta())
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()
        assert pred.grad.abs().sum().item() > 0

    def test_gradient_flows_to_sigma_scale(self):
        """Learnable log-scale correction must receive gradients."""
        c = BlackScholesConstraint()
        pred = torch.randn(4, 1)
        loss = c.residual(pred, self._meta())
        loss.backward()
        assert c._sigma_log_scale.grad is not None
        assert torch.isfinite(c._sigma_log_scale.grad)

    def test_zero_residual_when_pred_matches_physical_drift(self):
        """Feeding the exact physical-measure drift must zero the residual."""
        c = BlackScholesConstraint()
        meta = self._meta()
        # Compute mu_hat the same way residual() does.
        prices = meta["prices"]
        simple_returns = prices[:, 1:] / (prices[:, :-1] + 1e-8) - 1.0
        mu_hat = simple_returns.mean(dim=1, keepdim=True) / c.dt
        sigma = 0.2  # matches _meta(); _sigma_log_scale=0 → sigma_eff = sigma
        # target_mean=0, target_std=1 so de-normalised == normalised.
        expected = (mu_hat - 0.5 * sigma ** 2) * c.dt
        pred = expected.detach().clone()
        loss = c.residual(pred, meta)
        assert loss.item() == pytest.approx(0.0, abs=1e-8)

    def test_lambda_sensitivity_via_composite(self):
        """CompositeLoss with λ=2 must give strictly more physics weight than λ=0."""
        from src.losses.composite import CompositeLoss
        loss_fn = torch.nn.functional.mse_loss
        c = BlackScholesConstraint()
        comp_high = CompositeLoss(loss_fn, constraints=[c], lambdas={"bs": 2.0})
        comp_zero = CompositeLoss(loss_fn, constraints=[c], lambdas={"bs": 0.0})
        pred = torch.randn(4, 1, requires_grad=True)
        target = torch.zeros(4, 1)
        meta = self._meta()
        total_high, _ = comp_high(pred, target, meta)
        total_zero, _ = comp_zero(pred, target, meta)
        assert total_high.item() > total_zero.item(), (
            "λ=2 total must exceed λ=0 total when BS residual is non-zero"
        )

    def test_scalar_target_stats(self):
        """De-normalisation must work when target_mean/std are 0-d tensors (scalar output)."""
        c = BlackScholesConstraint()
        pred = torch.randn(4, 1, requires_grad=True)
        prices = torch.linspace(4000.0, 4050.0, 11).unsqueeze(0).expand(4, -1)
        meta = {
            "volatilities": 0.2 * torch.ones(4, 1),
            "target_mean": torch.tensor(0.0),
            "target_std": torch.tensor(1.0),
            "prices": prices,
        }
        loss = c.residual(pred, meta)
        _check_scalar(loss)

    def test_1d_volatilities_input(self):
        """residual must handle volatilities of shape [batch] (not just [batch, 1])."""
        c = BlackScholesConstraint()
        pred = torch.randn(4, 1)
        prices = torch.linspace(4000.0, 4050.0, 11).unsqueeze(0).expand(4, -1)
        meta = {
            "volatilities": 0.2 * torch.ones(4),  # shape [4], not [4, 1]
            "target_mean": torch.zeros(4, 1),
            "target_std": torch.ones(4, 1),
            "prices": prices,
        }
        loss = c.residual(pred, meta)
        _check_scalar(loss)


from src.losses.physics import LangevinConstraint


def test_langevin_mu_data_default_is_zero():
    c = LangevinConstraint()
    assert abs(float(c.mu_data.detach()) - 0.0) < 1e-7


def test_langevin_mu_data_nonzero():
    c = LangevinConstraint(mu_init=0.0005)
    assert abs(float(c.mu_data.detach()) - 0.0005) < 1e-7


def test_langevin_residual_shape_and_nonneg():
    c = LangevinConstraint(mu_init=0.0005)
    preds = torch.randn(8, 1)
    returns = torch.randn(8, 60)
    loss = c.residual(preds, {"returns": returns})
    assert loss.ndim == 0
    assert loss.item() >= 0.0


def test_langevin_mu_init_shifts_residual():
    """Drift residual must differ when mu_data shifts the equilibrium."""
    torch.manual_seed(0)
    preds = torch.randn(8, 1)
    returns = torch.zeros(8, 60)
    loss_zero = LangevinConstraint(mu_init=0.0).residual(preds, {"returns": returns})
    loss_pos  = LangevinConstraint(mu_init=0.5).residual(preds, {"returns": returns})
    assert abs(loss_zero.item() - loss_pos.item()) > 1e-6


from src.losses.physics import HawkesConstraint


class TestHawkesConstraint:
    """Tests for HawkesConstraint including EMA normalisation."""

    def _make_inputs(self, batch: int = 8, seq_len: int = 60):
        torch.manual_seed(0)
        preds = torch.randn(batch, 1) * 0.01  # realistic return scale
        returns = torch.randn(batch, seq_len) * 0.01
        return preds, {"returns": returns}

    def test_residual_scalar_nonneg(self):
        c = HawkesConstraint(mu0_init=1e-4)
        preds, meta = self._make_inputs()
        loss = c.residual(preds, meta)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_ema_scale_initialised_after_first_call(self):
        """_ema_scale must be None before first call, float after."""
        c = HawkesConstraint(mu0_init=1e-4)
        assert c._ema_scale is None
        preds, meta = self._make_inputs()
        c.residual(preds, meta)
        assert c._ema_scale is not None
        assert c._ema_scale > 0.0

    def test_normalised_loss_order_of_magnitude(self):
        """After warm-up, normalised loss must stay O(1) not O(1e-12)."""
        c = HawkesConstraint(mu0_init=1e-4)
        preds, meta = self._make_inputs()
        c.residual(preds, meta)  # warm-up: seeds _ema_scale
        # Second call with different predictions — EMA scale is now informed
        preds2 = torch.randn(8, 1) * 0.02
        loss2 = c.residual(preds2, meta)
        assert 1e-3 <= loss2.item() <= 1e3, (
            f"Expected normalised loss O(1) after warm-up, got {loss2.item():.3e}"
        )

    def test_ema_scale_updates_across_calls(self):
        """_ema_scale must change (EMA update) on successive calls."""
        c = HawkesConstraint(mu0_init=1e-4)
        preds, meta = self._make_inputs()
        c.residual(preds, meta)
        scale_after_first = c._ema_scale
        preds2 = torch.randn(8, 1) * 0.02
        c.residual(preds2, meta)
        assert c._ema_scale != scale_after_first

    def test_gradient_flows_to_predictions(self):
        """EMA normalisation must not break gradient flow."""
        c = HawkesConstraint(mu0_init=1e-4)
        preds = torch.randn(8, 1, requires_grad=True) * 0.01
        preds.retain_grad()  # non-leaf tensor: retain_grad() needed to populate .grad
        returns = torch.randn(8, 60) * 0.01
        loss = c.residual(preds, {"returns": returns})
        loss.backward()
        assert preds.grad is not None
        assert torch.isfinite(preds.grad).all()
        assert preds.grad.abs().sum().item() > 0

    def test_ema_clamp_prevents_division_by_zero(self):
        """Zero raw_loss must not produce NaN (clamp to 1e-12)."""
        c = HawkesConstraint(mu0_init=1e-4)
        preds = torch.zeros(4, 1)
        returns = torch.zeros(4, 60)
        loss1 = c.residual(preds, {"returns": returns})
        loss2 = c.residual(preds, {"returns": returns})
        assert torch.isfinite(loss2)
