"""Residual PINN — base LSTM + bounded GRU correction network."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.losses.physics import PhysicsConstraint
from src.models.base_pinn import BasePINN


class ResidualPINN(BasePINN):
    """Base LSTM forecaster + bounded GRU correction, with physics constraints.

    Implements the forecast decomposition
    ``ŷ = base(x) + max_correction · tanh(correction(x))``: the LSTM
    produces the primary prediction, and a parallel GRU (different
    architecture for representational diversity) emits a correction
    term hard-clipped to ``[−max_correction, +max_correction]`` so it
    cannot dominate the base signal. Physics constraints attached via
    :class:`BasePINN` act on the combined prediction.

    Inputs (via :meth:`BasePINN.forward`):
        x: ``[batch, seq_len, input_dim]`` normalised features.
        metadata: passed straight through (``_build_physics_metadata``
            is inherited as a no-op); required keys depend on the
            attached constraints.

    Returns (from ``forward``):
        ``(pred, metadata)`` with ``pred`` of shape ``[batch, 1]`` on
        the normalised target scale.

    Learnable parameters:
        * ``base_encoder``: ``nn.LSTM(input_dim → hidden_dim)``.
        * ``base_head``: ``nn.Linear(hidden_dim, 1)`` — primary forecast
          on the normalised target scale.
        * ``correction_encoder``: ``nn.GRU(input_dim → hidden_dim//2)``.
        * ``correction_head``: MLP ``H/2 → H/4 → 1`` with two ``Tanh``
          activations; the final ``tanh`` bounds the correction to
          ``[−1, 1]`` before multiplication by ``max_correction``.
        * plus any parameters owned by the attached ``constraints``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_correction: float = 0.1,
        constraints: list[PhysicsConstraint] | None = None,
    ) -> None:
        """Construct a residual PINN with base LSTM and bounded GRU correction.

        Args:
            input_dim: number of features per timestep (``F``).
            hidden_dim: hidden size ``H`` of the base LSTM; the
                correction GRU uses ``H // 2`` and its MLP head
                narrows to ``H // 4``.
            num_layers: stacked depth for both encoders.
            dropout: inter-layer dropout applied by both RNNs.
            max_correction: hard bound on the correction term (raw
                normalised-target units). Final prediction lies within
                ``base_pred ± max_correction``.
            constraints: physics constraints registered as child
                modules via :class:`BasePINN`; empty/``None`` gives a
                pure residual baseline with no physics loss.
        """
        super().__init__(constraints or [])
        self.max_correction = max_correction

        # Base encoder (LSTM)
        self.base_encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )
        self.base_head = nn.Linear(hidden_dim, 1)

        # Correction encoder (GRU — different architecture for diversity)
        self.correction_encoder = nn.GRU(
            input_dim, hidden_dim // 2, num_layers, batch_first=True, dropout=dropout,
        )
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Tanh(),  # bounds output to [-1, 1]
        )

    def _encode(self, x: Tensor) -> Tensor:
        """Run both encoders in parallel and concatenate their final states.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features.

        Returns:
            ``[batch, hidden_dim + hidden_dim//2]`` concatenation of
            the LSTM's top-layer final hidden state (first
            ``hidden_dim`` columns) and the GRU's top-layer final
            hidden state (remaining ``hidden_dim//2`` columns). The
            split index is recovered from ``base_head.in_features`` in
            :meth:`_predict`.
        """
        _, (h_base, _) = self.base_encoder(x)
        _, h_corr = self.correction_encoder(x)
        return torch.cat([h_base[-1], h_corr[-1]], dim=-1)  # [B, H + H//2]

    def _predict(self, hidden: Tensor) -> Tensor:
        """Combine base forecast with the bounded correction term.

        Splits the packed hidden state back into base / correction
        halves, projects each through its head, and returns
        ``base_pred + max_correction · tanh_mlp(h_corr)``.

        Args:
            hidden: ``[batch, hidden_dim + hidden_dim//2]`` from
                :meth:`_encode`.

        Returns:
            ``[batch, 1]`` forecast on the normalised target scale,
            guaranteed to lie within ``base_pred ± max_correction``.
        """
        split = self.base_head.in_features
        h_base = hidden[:, :split]
        h_corr = hidden[:, split:]

        base_pred = self.base_head(h_base)                              # [B, 1]
        correction = self.correction_head(h_corr) * self.max_correction # [B, 1]
        return base_pred + correction
