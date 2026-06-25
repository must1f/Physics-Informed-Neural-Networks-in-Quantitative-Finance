"""Losses and physics-informed constraints for FINN v2.

Public API
----------
Data losses (thin wrappers around ``torch.nn.functional``):
    mse_loss, mae_loss, huber_loss

Physics constraints (strategy-pattern, all subclass ``PhysicsConstraint``):
    GBMConstraint          — GBM Itô-corrected drift residual (RQ1)
    OUConstraint           — Ornstein-Uhlenbeck mean-reversion residual (RQ2)
    BlackScholesConstraint — BS physical-measure drift constraint (RQ3)
    LangevinConstraint     — overdamped Langevin double-well residual (RQ4)
    HawkesConstraint       — self-exciting Hawkes intensity residual v1 (RQ4)
    HawkesConstraintV2     — Hawkes QMLE NLL + structural match, stationarity
                             by construction (RQ4 improved formulation)

Composite:
    CompositeLoss          — weighted sum of data loss + physics residuals
"""

from src.losses.data_losses import huber_loss, mae_loss, mse_loss
from src.losses.physics import (
    BlackScholesConstraint,
    GBMConstraint,
    HawkesConstraint,
    HawkesConstraintV2,
    LangevinConstraint,
    OUConstraint,
    PhysicsConstraint,
)
from src.losses.composite import CompositeLoss

__all__ = [
    # data losses
    "mse_loss",
    "mae_loss",
    "huber_loss",
    # physics constraints
    "PhysicsConstraint",
    "GBMConstraint",
    "OUConstraint",
    "BlackScholesConstraint",
    "LangevinConstraint",
    "HawkesConstraint",
    "HawkesConstraintV2",
    # composite
    "CompositeLoss",
]
