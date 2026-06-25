"""Model layer — baselines, PINNs, and the registry factory.

Public API
----------
build_model(name, mu_init, hawkes_mu0_init, **kwargs) -> nn.Module
    Preferred entry-point: constructs any of the 19 registered models by
    name with fresh constraint instances.

list_models() -> list[str]
    Returns all registered model names.

Classes exported for type annotations and direct construction in tests:
    BasePINN, BaselineModel, HawkesDecoupledPINN, PINNModel,
    ResidualPINN, StackedPINN.
"""

from src.models.base_pinn import BasePINN
from src.models.baselines import BaselineModel
from src.models.hawkes_decoupled_pinn import HawkesDecoupledPINN
from src.models.pinn import PINNModel
from src.models.registry import build_model, list_models
from src.models.residual_pinn import ResidualPINN
from src.models.stacked_pinn import StackedPINN

__all__ = [
    "BasePINN",
    "BaselineModel",
    "HawkesDecoupledPINN",
    "PINNModel",
    "ResidualPINN",
    "StackedPINN",
    "build_model",
    "list_models",
]
