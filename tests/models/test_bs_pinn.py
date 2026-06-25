import torch
import pytest
from src.models.pinn import PINNModel
from src.losses.physics import BlackScholesConstraint


def test_bs_pinn_forward_does_not_mutate_input_requires_grad():
    """After fix, forward() must NOT set requires_grad on the input tensor."""
    c = BlackScholesConstraint()
    model = PINNModel(input_dim=5, hidden_dim=32, num_layers=1, constraints=[c])
    model.eval()
    x = torch.randn(2, 10, 5)
    assert not x.requires_grad
    with torch.no_grad():
        meta = {
            "target_mean": torch.zeros(2, 1),
            "target_std": torch.ones(2, 1),
        }
        pred, enriched = model(x, meta)
    assert not x.requires_grad, "forward() must not mutate input requires_grad"
    assert pred.shape == (2, 1)


def test_bs_pinn_forward_injects_volatilities():
    """_build_physics_metadata must add 'volatilities' from vol_head."""
    c = BlackScholesConstraint()
    model = PINNModel(input_dim=5, hidden_dim=32, num_layers=1, constraints=[c])
    model.eval()
    x = torch.randn(2, 10, 5)
    meta = {
        "target_mean": torch.zeros(2, 1),
        "target_std": torch.ones(2, 1),
    }
    pred, enriched = model(x, meta)
    assert "volatilities" in enriched
    assert enriched["volatilities"].shape == (2, 1)
    assert (enriched["volatilities"] > 0).all()  # softplus > 0


def test_bs_pinn_forward_does_not_inject_predictions_next():
    """New BS constraint does not need predictions_next — must not be injected."""
    c = BlackScholesConstraint()
    model = PINNModel(input_dim=5, hidden_dim=32, num_layers=1, constraints=[c])
    model.eval()
    x = torch.randn(2, 10, 5)
    meta = {
        "target_mean": torch.zeros(2, 1),
        "target_std": torch.ones(2, 1),
    }
    pred, enriched = model(x, meta)
    assert "predictions_next" not in enriched


def test_vol_head_receives_gradients_during_training():
    """vol_head must receive gradients in training mode so BS constraint is live."""
    from src.losses.physics import BlackScholesConstraint
    from src.losses.composite import CompositeLoss
    import torch.nn.functional as F

    c = BlackScholesConstraint()
    model = PINNModel(input_dim=5, hidden_dim=32, num_layers=1, constraints=[c])
    model.train()

    x = torch.randn(2, 10, 5)
    y = torch.zeros(2, 1)
    meta = {
        "target_mean": torch.zeros(2, 1),
        "target_std": torch.ones(2, 1),
    }

    pred, enriched = model(x, meta)
    loss_fn = CompositeLoss(F.mse_loss, constraints=[c], lambdas={"bs": 1.0})
    total, _ = loss_fn(pred, y, enriched)
    total.backward()

    vol_head_grad = model.vol_head.weight.grad
    assert vol_head_grad is not None
    assert torch.isfinite(vol_head_grad).all()
    assert vol_head_grad.abs().sum().item() > 0
