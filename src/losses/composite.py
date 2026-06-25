"""
Composite loss — combines a data loss with zero or more physics constraints.

When no constraints are provided, this reduces to a pure data loss (baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.losses.physics import PhysicsConstraint
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CompositeLoss(nn.Module):
    """Weighted sum of a data loss and physics-informed constraint residuals.

    Args:
        data_loss_fn: Callable ``(pred, target) -> scalar`` for the data term.
        constraints: List of :class:`PhysicsConstraint` modules.  Pass an
            empty list for a pure data-driven baseline.
        lambdas: Optional mapping ``{constraint.name: weight}``.  Any
            constraint not present in the dict defaults to ``1.0``.
    """

    def __init__(
        self,
        data_loss_fn,
        constraints: list[PhysicsConstraint] | None = None,
        lambdas: dict[str, float] | None = None,
    ) -> None:
        """Initialise CompositeLoss.

        Args:
            data_loss_fn: Callable ``(pred, target) -> scalar Tensor``.
                Typically :func:`mse_loss`, :func:`mae_loss`, or
                :func:`huber_loss`.  Must accept tensors of the same shape
                as the model output and return a scalar.
            constraints: List of :class:`PhysicsConstraint` modules to add
                as weighted residual terms.  Pass ``None`` or ``[]`` for a
                pure data-driven baseline.
            lambdas: Mapping ``{constraint.name: weight}`` controlling the
                contribution of each physics term in the total loss.  Keys
                not present default to ``1.0``.  Typical values used in the
                dissertation are ``{"gbm": 0.01, "ou": 0.01, "bs": 0.01}``.
        """
        super().__init__()
        self.data_loss_fn = data_loss_fn
        self.constraints = nn.ModuleList(constraints or [])
        self._lambdas: dict[str, float] = lambdas or {}

    def _lambda(self, name: str) -> float:
        """Return the physics weight for constraint *name*, defaulting to 1.0.

        Args:
            name: ``constraint.name`` string (e.g. ``"gbm"``, ``"ou"``).

        Returns:
            float weight λ used to scale the constraint residual in the total loss.
        """
        return self._lambdas.get(name, 1.0)

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        physics_input: dict | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute total loss and per-term breakdown.

        Args:
            pred: Model predictions.
            target: Ground-truth targets.
            physics_input: Metadata dict forwarded to each constraint's
                ``residual()`` method.  Ignored when *constraints* is empty.

        Returns:
            ``(total_loss, breakdown)`` where *breakdown* contains keys
            ``"data"``, ``"total"``, one entry per constraint name holding
            the **λ-weighted** physics contribution (``λ * residual``), and
            one ``"{name}_unweighted"`` entry holding the raw residual.
            Storing the weighted value makes ``physics_ratio`` in the training
            history reflect the actual loss contribution at each λ.

        Target injection
        ----------------
        Before iterating constraints, ``target`` is added to
        ``physics_input`` under key ``"target"`` (only if not already
        present, so callers can override). This lets NLL-style constraints
        such as :class:`HawkesConstraintV2` access the observed next-step
        return for quasi-likelihood residuals without widening the
        :class:`PhysicsConstraint.residual` signature. Legacy constraints
        that ignore ``"target"`` are unaffected.
        """
        data_loss = self.data_loss_fn(pred, target)
        breakdown: dict[str, Tensor] = {"data": data_loss}

        total = data_loss

        if physics_input is None:
            physics_input = {}
        # Inject the observed target so NLL-style constraints (Hawkes v2)
        # can compute log-likelihood residuals; setdefault preserves any
        # caller-supplied override.
        physics_input = dict(physics_input)
        physics_input.setdefault("target", target)

        for constraint in self.constraints:
            lam = self._lambda(constraint.name)
            phys_loss = constraint.residual(pred, physics_input)
            weighted = lam * phys_loss
            total = total + weighted
            breakdown[constraint.name] = weighted
            breakdown[f"{constraint.name}_unweighted"] = phys_loss

        breakdown["total"] = total
        return total, breakdown
