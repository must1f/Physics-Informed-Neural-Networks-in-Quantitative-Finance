import pytest
from src.models.registry import build_model


def test_build_ou_pinn_default_mu():
    m = build_model("ou_pinn", input_dim=14, hidden_dim=64, num_layers=2, dropout=0.1)
    ou = m.constraints[0]
    assert abs(float(ou.mu.detach()) - 0.0) < 1e-7


def test_build_ou_pinn_custom_mu():
    m = build_model("ou_pinn", input_dim=14, hidden_dim=64, num_layers=2, dropout=0.1, mu_init=0.0005)
    ou = m.constraints[0]
    assert abs(float(ou.mu.detach()) - 0.0005) < 1e-7


def test_build_gbm_ou_pinn_custom_mu():
    m = build_model("gbm_ou_pinn", input_dim=14, hidden_dim=64, num_layers=2, dropout=0.1, mu_init=0.0003)
    # constraints = [GBMConstraint, OUConstraint] — OU is at index 1
    ou = m.constraints[1]
    assert abs(float(ou.mu.detach()) - 0.0003) < 1e-7


def test_build_gbm_pinn_unaffected():
    # gbm_pinn has no OUConstraint — mu_init accepted but does nothing
    m = build_model("gbm_pinn", input_dim=14, hidden_dim=64, num_layers=2, dropout=0.1, mu_init=0.9999)
    assert len(m.constraints) == 1


def test_build_hawkes_ou_pinn_custom_mu():
    m = build_model("hawkes_ou_pinn", input_dim=14, hidden_dim=64, num_layers=2, dropout=0.1, mu_init=0.0002)
    # constraints = [HawkesConstraint, OUConstraint] — OU is at index 1
    ou = m.constraints[1]
    assert abs(float(ou.mu.detach()) - 0.0002) < 1e-7


def test_build_hawkes_pinn_custom_mu0():
    # hawkes_mu0_init threads into HawkesConstraint so softplus(_mu0_raw) ==
    # requested scale, matching train-set mean(r²) at init.
    m = build_model(
        "hawkes_pinn",
        hawkes_mu0_init=1e-4,
        input_dim=14, hidden_dim=32, num_layers=1,
    )
    hc = m.constraints[0]
    assert abs(float(hc.mu0.detach()) - 1e-4) < 1e-8
