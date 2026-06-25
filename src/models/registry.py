"""Model registry — maps 19 model names to pre-configured constructors.

Each ``build_model`` call creates **fresh** constraint instances so that
models built from the same registry entry never share mutable state.
Classical models (random_walk, historical_mean, persistence, garch) are
registered here for name validation only; run_experiment dispatches them
to _run_classical_experiment before build_model is ever called.
"""

from __future__ import annotations

from functools import partial
from typing import Callable

import torch.nn as nn

from src.losses.physics import (
    BlackScholesConstraint,
    GBMConstraint,
    HawkesConstraint,
    HawkesConstraintV2,
    LangevinConstraint,
    OUConstraint,
    PhysicsConstraint,
)
from src.models.baselines import BaselineModel
from src.models.classical import GARCHModel, HistoricalMeanModel, PersistenceModel, RandomWalkModel
from src.models.hawkes_decoupled_pinn import HawkesDecoupledPINN
from src.models.pinn import PINNModel
from src.models.residual_pinn import ResidualPINN
from src.models.stacked_pinn import StackedPINN

# Constraint factories — called each time to avoid shared mutable state.
# Each lambda accepts ``mu_init`` (training-set mean log-return) so that
# OUConstraint and LangevinConstraint start with a physically calibrated
# long-run mean instead of zero. Hawkes factories additionally accept
# ``mu0_init`` (training-set mean(r²)) so HawkesConstraint.mu0 is seeded
# on the return-variance scale rather than the legacy softplus(0) ≈ 0.69.
# Non-hawkes factories are only called with ``mu_init``; ``mu0_init`` is
# threaded through ``build_model`` only for names containing "hawkes".
_CONSTRAINT_FACTORIES: dict[str, Callable[..., list[PhysicsConstraint]]] = {
    "baseline_pinn":  lambda mu_init=0.0: [],
    "gbm_pinn":       lambda mu_init=0.0: [GBMConstraint()],
    "ou_pinn":        lambda mu_init=0.0: [OUConstraint(mu_init=mu_init)],
    "bs_pinn":        lambda mu_init=0.0: [BlackScholesConstraint()],
    "gbm_ou_pinn":    lambda mu_init=0.0: [GBMConstraint(), OUConstraint(mu_init=mu_init)],
    "global_pinn":    lambda mu_init=0.0: [GBMConstraint(), OUConstraint(mu_init=mu_init),
                                           BlackScholesConstraint(), LangevinConstraint(mu_init=mu_init)],
    "hawkes_pinn":    lambda mu_init=0.0, mu0_init=None: [HawkesConstraint(mu0_init=mu0_init)],
    "hawkes_ou_pinn": lambda mu_init=0.0, mu0_init=None: [HawkesConstraint(mu0_init=mu0_init),
                                                          OUConstraint(mu_init=mu_init)],
    # Hawkes v2 — NLL + stationarity reparam + decoupled variance head.
    # Paired with HawkesDecoupledPINN (two-head model) in MODEL_REGISTRY.
    # K=1: single-scale Hawkes; K=2: heterogeneous-Hawkes (short/long memory).
    "hawkes_v2_pinn":            lambda mu_init=0.0, mu0_init=None:
        [HawkesConstraintV2(mu0_init=mu0_init, num_components=1)],
    "hawkes_v2_multiscale_pinn": lambda mu_init=0.0, mu0_init=None:
        [HawkesConstraintV2(mu0_init=mu0_init, num_components=2)],
    # RQ5 novel architectures — GBM + OU pairing (drift + mean-reversion).
    # Matches gbm_ou_pinn's physics so the RQ5 comparison isolates the
    # architecture effect (dual-encoder vs single-encoder, residual-correction
    # vs direct) with physics held constant.
    "stacked_pinn":   lambda mu_init=0.0: [GBMConstraint(), OUConstraint(mu_init=mu_init)],
    "residual_pinn":  lambda mu_init=0.0: [GBMConstraint(), OUConstraint(mu_init=mu_init)],
}

MODEL_REGISTRY: dict[str, type | partial] = {
    # ── Neural baselines ──
    "lstm":           partial(BaselineModel, arch="lstm"),
    "gru":            partial(BaselineModel, arch="gru"),
    "bilstm":         partial(BaselineModel, arch="bilstm"),
    "attention_lstm": partial(BaselineModel, arch="attention_lstm"),
    "transformer":    partial(BaselineModel, arch="transformer"),
    # ── Core PINNs (built via _CONSTRAINT_FACTORIES in build_model) ──
    "baseline_pinn":  PINNModel,
    "gbm_pinn":       PINNModel,
    "ou_pinn":        PINNModel,
    "bs_pinn":        PINNModel,
    "gbm_ou_pinn":    PINNModel,
    "global_pinn":    PINNModel,
    "hawkes_pinn":    PINNModel,
    "hawkes_ou_pinn": PINNModel,
    # ── Hawkes v2 (decoupled mean + log-variance heads) ──
    "hawkes_v2_pinn":            HawkesDecoupledPINN,
    "hawkes_v2_multiscale_pinn": HawkesDecoupledPINN,
    # ── Novel architectures (RQ5) ──
    "stacked_pinn":   StackedPINN,
    "residual_pinn":  ResidualPINN,
    # ── Classical baselines (registered for name validation; dispatched in runner) ──
    "random_walk":     RandomWalkModel,
    "historical_mean": HistoricalMeanModel,
    "persistence":     PersistenceModel,
    "garch":           GARCHModel,
}


def build_model(
    name: str,
    mu_init: float = 0.0,
    hawkes_mu0_init: float | None = None,
    **kwargs,
) -> nn.Module:
    """Construct a model by registry name with fresh constraint instances.

    Args:
        name: key in :data:`MODEL_REGISTRY` (e.g. ``"lstm"``,
            ``"gbm_pinn"``, ``"stacked_pinn"``).
        mu_init: seed-varying bootstrap mean over training-split log-returns
            (produced by :func:`src.training.runner._bootstrap_mean`).
            Forwarded to ``OUConstraint`` and ``LangevinConstraint`` as
            their equilibrium-mean initialisation. Each experiment seed
            must produce a distinct ``mu_init`` — otherwise every seed
            starts from the same physics attractor and collapses to
            identical predictions (2026-04-19 audit, F4). Silently ignored
            for models without OU / Langevin constraints (``gbm_pinn``,
            ``bs_pinn``, ``hawkes_pinn``, ``baseline_pinn``). Default
            ``0.0`` preserves previous behaviour for unit tests.
        hawkes_mu0_init: training-set ``mean(r²)`` forwarded to
            ``HawkesConstraint`` so its baseline intensity ``mu0`` is
            seeded on the return-variance scale (matches ``pred² ∼ 1e-4``
            at init). Only threaded into the two Hawkes factories
            (``hawkes_pinn``, ``hawkes_ou_pinn``, ``hawkes_v2_pinn``,
            ``hawkes_v2_multiscale_pinn``); silently ignored by other
            factories. Default ``None`` preserves the legacy
            ``softplus(0) ≈ 0.69`` initialisation.
        **kwargs: forwarded to the model constructor (typically
            ``input_dim``, ``hidden_dim``, ``num_layers``, ``dropout``).

    Returns:
        An initialised ``nn.Module`` ready for training, with fresh
        constraint instances carrying ``mu_init`` / ``hawkes_mu0_init``
        where applicable.

    Raises:
        ValueError: if *name* is not in :data:`MODEL_REGISTRY`.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name!r}. Available: {list(MODEL_REGISTRY)}"
        )
    if name in _CONSTRAINT_FACTORIES:
        factory = _CONSTRAINT_FACTORIES[name]
        factory_kwargs: dict = {"mu_init": mu_init}
        if "hawkes" in name:
            factory_kwargs["mu0_init"] = hawkes_mu0_init
        kwargs.setdefault("constraints", factory(**factory_kwargs))
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return all registered model names in insertion order.

    Returns:
        List of strings, each a valid *name* argument to :func:`build_model`.
        Includes neural baselines, PINN variants, novel architectures, and
        classical baselines (19 entries total).
    """
    return list(MODEL_REGISTRY)
