"""Abstract base class for all PINN variants (Template Method pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from src.losses.physics import PhysicsConstraint


class BasePINN(nn.Module, ABC):
    """Abstract contract every PINN variant in the registry must satisfy.

    Implements the Template Method pattern: subclasses supply ``_encode``
    and ``_predict``; the base class orchestrates ``forward``, inference,
    physics-metadata enrichment, and diagnostics. Physics constraints are
    injected via composition (strategy pattern) rather than inheritance.

    Inputs (to ``forward`` / ``predict``):
        x: ``[batch, seq_len, n_features]`` float tensor of normalised
            feature windows (z-scored log returns + engineered features,
            as produced by ``src/data`` pipeline). ``requires_grad`` only
            when the caller needs autograd-based physics residuals (e.g.
            Black–Scholes); otherwise no grad is required.
        metadata: optional dict passed through to ``CompositeLoss``. May
            carry keys such as ``"t"`` (time-to-maturity), ``"S"`` (spot
            price tensor, raw scale), ``"r"``, ``"sigma"`` — required keys
            depend on which ``PhysicsConstraint`` instances are attached.

    Returns (from ``forward``):
        Tuple ``(pred, enriched_metadata)`` where ``pred`` has shape
        ``[batch, 1]`` on the same scale as the training target
        (normalised next-step log return by default) and
        ``enriched_metadata`` is the input dict augmented with any
        tensors the physics constraints need (e.g. price gradients).

    Learnable parameters:
        Only those contributed by registered ``constraints`` (stored in
        ``self.constraints: nn.ModuleList``). The base class itself
        introduces no learnable weights; subclass encoders/heads add
        their own.
    """

    def __init__(self, constraints: list[PhysicsConstraint]) -> None:
        """Register physics constraints as child modules.

        Args:
            constraints: list of ``PhysicsConstraint`` strategies. Each
                may hold its own learnable parameters (e.g. softplus-
                constrained volatility, unconstrained drift) and exposes
                a ``.name`` plus ``.forward(pred, metadata) -> Tensor``
                residual on a scale defined by that constraint.
        """
        super().__init__()
        self.constraints = nn.ModuleList(constraints)

    # ── Abstract hooks (subclass MUST implement) ───────────────────

    @abstractmethod
    def _encode(self, x: Tensor) -> Tensor:
        """Encode a feature window into a hidden representation.

        Args:
            x: ``[batch, seq_len, n_features]`` normalised features.

        Returns:
            Hidden tensor whose shape is subclass-defined (commonly
            ``[batch, hidden_dim]`` for sequence encoders that pool, or
            ``[batch, seq_len, hidden_dim]`` when the head attends).
        """

    @abstractmethod
    def _predict(self, hidden: Tensor) -> Tensor:
        """Map hidden representation to the forecast head.

        Args:
            hidden: tensor produced by ``_encode``.

        Returns:
            ``[batch, 1]`` prediction on the normalised target scale
            (typically z-scored next-step log return).
        """

    # ── Concrete methods (inherited by all subclasses) ─────────────

    def forward(
        self, x: Tensor, metadata: dict | None = None,
    ) -> tuple[Tensor, dict]:
        """Run encode → predict → physics-metadata enrichment.

        Args:
            x: ``[batch, seq_len, n_features]`` normalised features.
                Must have ``requires_grad=True`` if any attached
                constraint computes autograd derivatives (e.g. the
                Black–Scholes constraint needs ∂V/∂S, ∂V/∂t).
            metadata: optional dict of auxiliary tensors required by the
                attached constraints (keys depend on constraint set;
                e.g. Black–Scholes requires ``"S"``, ``"t"``, ``"r"``,
                ``"sigma"``). ``None`` is treated as ``{}``.

        Returns:
            ``(pred, enriched_metadata)``:
              * ``pred``: ``[batch, 1]`` on the normalised target scale.
              * ``enriched_metadata``: input dict plus any tensors
                injected by ``_build_physics_metadata`` (e.g. ``"V"``,
                ``"dVdS"``, ``"dVdt"``, ``"d2VdS2"`` on the raw price
                scale for BS-style constraints).
        """
        hidden = self._encode(x)
        pred = self._predict(hidden)
        enriched = self._build_physics_metadata(
            x, hidden, pred, metadata if metadata is not None else {},
        )
        return pred, enriched

    def predict(self, x: Tensor) -> Tensor:
        """Inference-only forward pass (no grad, no physics metadata).

        Args:
            x: ``[batch, seq_len, n_features]`` normalised features.
                Gradient tracking is disabled via ``torch.no_grad``.

        Returns:
            ``[batch, 1]`` prediction on the normalised target scale.
            De-normalise upstream if raw-price output is required.
        """
        with torch.no_grad():
            hidden = self._encode(x)
            return self._predict(hidden)

    def diagnostics(self) -> dict:
        """Snapshot constraint names and their current scalar parameters.

        Returns:
            Dict of the form::

                {
                    "constraints": ["bs", "no_arbitrage", ...],
                    "bs_params": {"sigma": 0.23, "r": 0.05, ...},
                    ...
                }

            Each ``<name>_params`` entry contains the current ``.item()``
            value of every learnable scalar in that constraint (post-
            activation, e.g. softplus-transformed σ). Intended for
            logging / TensorBoard; not part of the training signal.
        """
        info: dict = {"constraints": [c.name for c in self.constraints]}
        for c in self.constraints:
            info[f"{c.name}_params"] = {
                name: (
                    param.item() if param.numel() == 1
                    else param.detach().cpu().tolist()
                )
                for name, param in c.named_parameters()
            }
        return info

    def _build_physics_metadata(
        self, x: Tensor, hidden: Tensor, pred: Tensor, metadata: dict,
    ) -> dict:
        """Hook for subclasses to inject physics tensors into metadata.

        Default implementation is a no-op pass-through. Subclasses that
        need autograd derivatives (e.g. Black–Scholes variants) override
        this to compute ∂V/∂S, ∂V/∂t, ∂²V/∂S² from ``pred`` w.r.t. the
        raw-scale inputs recovered from ``x`` / ``metadata`` and add
        them under the keys consumed by the attached constraints.

        Args:
            x: original feature tensor passed to ``forward``.
            hidden: output of ``_encode`` (subclass-defined shape).
            pred: ``[batch, 1]`` output of ``_predict``.
            metadata: caller-supplied dict (never ``None`` here).

        Returns:
            Dict to forward to ``CompositeLoss``. Same object as
            ``metadata`` by default; subclasses may return a new dict.
        """
        return metadata
