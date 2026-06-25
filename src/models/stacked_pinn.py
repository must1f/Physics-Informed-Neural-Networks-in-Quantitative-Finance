"""Stacked PINN — parallel LSTM + GRU encoders with attention fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses.physics import PhysicsConstraint
from src.models.base_pinn import BasePINN


class StackedPINN(BasePINN):
    """Parallel LSTM + GRU encoders fused via learned two-way attention.

    Implements ``fused = α · h_lstm + (1 − α) · h_gru`` where
    ``α, 1−α = softmax(MLP([h_lstm ; h_gru]))``. Both encoders consume
    the same input and produce their top-layer final hidden state; a
    small attention MLP scores each encoder and the softmax weights
    form a convex combination of the two representations. The fused
    vector feeds a single linear head. Physics constraints from
    :class:`BasePINN` act on the combined prediction.

    The most recent softmax weights are cached (detached) on
    ``self.last_attention_weights`` as ``[batch, 2]`` for logging /
    interpretability; they do not participate in gradient flow.

    Inputs (via :meth:`BasePINN.forward`):
        x: ``[batch, seq_len, input_dim]`` normalised features.
        metadata: passed through unchanged (inherited no-op
            ``_build_physics_metadata``); required keys depend on the
            attached constraints.

    Returns (from ``forward``):
        ``(pred, metadata)`` with ``pred`` of shape ``[batch, 1]`` on
        the normalised target scale.

    Learnable parameters:
        * ``lstm``: ``nn.LSTM(input_dim → hidden_dim)``.
        * ``gru``:  ``nn.GRU(input_dim → hidden_dim)``.
        * ``attn``: MLP ``2H → H → 2`` (Tanh activation), softmaxed
          over the last axis to produce the two-encoder weights
          (unconstrained pre-softmax logits).
        * ``prediction_head``: ``nn.Linear(hidden_dim, 1)``.
        * plus any parameters owned by the attached ``constraints``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        constraints: list[PhysicsConstraint] | None = None,
    ) -> None:
        """Construct a stacked PINN with parallel LSTM + GRU and attention fusion.

        Args:
            input_dim: number of features per timestep (``F``).
            hidden_dim: hidden size ``H`` of each encoder and of the
                fused representation fed to ``prediction_head``.
            num_layers: stacked depth for both encoders.
            dropout: inter-layer dropout applied by both RNNs.
            constraints: physics constraints registered as child
                modules via :class:`BasePINN`; empty/``None`` gives a
                pure dual-encoder baseline with no physics loss.
        """
        super().__init__(constraints or [])

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )

        # Attention fusion: score each encoder's output
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

        self.prediction_head = nn.Linear(hidden_dim, 1)
        self.last_attention_weights: Tensor | None = None

    def _encode(self, x: Tensor) -> Tensor:
        """Run both encoders and fuse their final states via softmax attention.

        Steps:
            1. Take the top-layer final hidden state of each encoder
               (``h_lstm``, ``h_gru`` — both ``[batch, hidden_dim]``).
            2. Concatenate to ``[batch, 2·hidden_dim]`` and score via
               the attention MLP to obtain logits ``[batch, 2]``.
            3. Softmax over the two-encoder axis → weights summing to 1.
            4. Return the weighted sum of the two hidden states.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features.

        Returns:
            ``[batch, hidden_dim]`` fused representation. Side effect:
            ``self.last_attention_weights`` is updated to the detached
            ``[batch, 2]`` softmax tensor (column 0 = LSTM weight,
            column 1 = GRU weight).
        """
        _, (h_lstm, _) = self.lstm(x)
        _, h_gru = self.gru(x)

        h_lstm = h_lstm[-1]  # [B, H]
        h_gru = h_gru[-1]   # [B, H]

        # Attention weights over the two encoders
        combined = torch.cat([h_lstm, h_gru], dim=-1)          # [B, 2H]
        weights = F.softmax(self.attn(combined), dim=-1)       # [B, 2]
        self.last_attention_weights = weights.detach()

        # Weighted fusion
        stacked = torch.stack([h_lstm, h_gru], dim=1)          # [B, 2, H]
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)   # [B, H]
        return fused

    def _predict(self, hidden: Tensor) -> Tensor:
        """Project the fused hidden state to a scalar forecast.

        Args:
            hidden: ``[batch, hidden_dim]`` attention-fused state from
                :meth:`_encode`.

        Returns:
            ``[batch, 1]`` prediction on the normalised target scale
            (z-scored next-step log return).
        """
        return self.prediction_head(hidden)
