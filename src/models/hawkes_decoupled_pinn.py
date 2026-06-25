"""Hawkes v2 — decoupled mean / log-variance heads for NLL training."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.losses.physics import PhysicsConstraint
from src.models.base_pinn import BasePINN


class HawkesDecoupledPINN(BasePINN):
    """Recurrent encoder with separate mean and log-variance heads.

    Designed to pair with :class:`~src.losses.physics.HawkesConstraintV2`.
    The mean head predicts the signed next-step return; the log-variance
    head predicts ``log σ̂²_{t+1}``. Keeping them independent breaks the
    structural sign-flip degeneracy of the legacy single-head Hawkes PINN
    (which forced the same scalar to satisfy both ``pred² ≈ λ`` and
    ``pred ≈ r_{t+1}``, producing a double-well optimum at ±√λ).

    The class is Hawkes-v2 agnostic: it merely exposes ``log_var`` via
    ``metadata["log_var"]`` so *any* variance-consuming constraint can
    plug in. The attached constraints list decides whether the variance
    head is used physics-informedly or ignored.

    Inputs (via :meth:`BasePINN.forward`):
        x: ``[batch, seq_len, input_dim]`` normalised feature windows
            produced by the ``src.data`` pipeline.
        metadata: dict passed through to ``CompositeLoss``. Must already
            carry ``"returns"`` (log-return history) when
            ``HawkesConstraintV2`` is attached. ``"target"`` is injected
            automatically by ``CompositeLoss.forward``.

    Returns (from ``forward``):
        ``(mean_pred, enriched_metadata)`` where
          * ``mean_pred``: ``[batch, 1]`` z-scored next-step log-return
            (the primary training target, consumed by ``mse_loss`` in
            ``CompositeLoss``).
          * ``enriched_metadata``: input dict with an additional
            ``"log_var"`` key holding ``[batch, 1]`` log-variance (the
            log-variance head output, **no activation** — constraints
            exponentiate internally and clamp for numerical safety).

    Learnable parameters:
        * ``_encoder``: ``nn.LSTM`` or ``nn.GRU`` stack.
        * ``mean_head``: ``nn.Linear(hidden_dim, 1)`` — signed return
          prediction on the z-scored target scale.
        * ``log_var_head``: ``nn.Linear(hidden_dim, 1)`` — raw
          log-variance output; unconstrained so the variance head can
          represent small (≈ log 1e-4 ≈ −9.2) values without saturating.
        * plus any parameters owned by attached ``constraints`` (e.g.
          ``HawkesConstraintV2`` contributes ``_mu0_raw``,
          ``_branching_raw``, ``_beta_raw``).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        encoder: str = "lstm",
        constraints: list[PhysicsConstraint] | None = None,
    ) -> None:
        """Construct a dual-head PINN with shared recurrent encoder.

        Args:
            input_dim: number of features per timestep ``F``.
            hidden_dim: encoder hidden size ``H``; both heads project
                from this to scalars.
            num_layers: stacked encoder depth.
            dropout: inter-layer dropout applied by the RNN.
            encoder: ``"lstm"`` or ``"gru"``; any other value raises
                ``ValueError``. Shared by both heads (single encoder
                feeds mean and log-variance projections).
            constraints: physics constraints registered as child
                modules via :class:`BasePINN`. When
                :class:`HawkesConstraintV2` is attached the log-variance
                head is trained via the NLL + structural-match residual;
                otherwise the head still runs but contributes no signal.
        """
        super().__init__(constraints or [])
        self.encoder_type = encoder

        if encoder == "lstm":
            self._encoder = nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout,
            )
        elif encoder == "gru":
            self._encoder = nn.GRU(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown encoder '{encoder}'. Choose 'lstm' or 'gru'.")

        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_var_head = nn.Linear(hidden_dim, 1)

    def _encode(self, x: Tensor) -> Tensor:
        """Run the recurrent encoder and pool to the last-layer final state.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features.

        Returns:
            ``[batch, hidden_dim]`` final hidden state of the top RNN
            layer (``h_n[-1]``). Cell state (LSTM) is discarded.
        """
        if self.encoder_type == "lstm":
            _, (h_n, _) = self._encoder(x)
        else:
            _, h_n = self._encoder(x)
        return h_n[-1]

    def _predict(self, hidden: Tensor) -> Tensor:
        """Project the pooled hidden state to the signed mean forecast.

        Args:
            hidden: ``[batch, hidden_dim]`` from :meth:`_encode`.

        Returns:
            ``[batch, 1]`` mean prediction on the z-scored target scale.
            Trained by ``mse_loss`` against ``y`` in ``CompositeLoss``
            and simultaneously regularised by the NLL term of
            :class:`HawkesConstraintV2` through the shared hidden state.
        """
        return self.mean_head(hidden)

    def _build_physics_metadata(
        self, x: Tensor, hidden: Tensor, pred: Tensor, metadata: dict,
    ) -> dict:
        """Inject ``log_var`` from the log-variance head into metadata.

        Args:
            x: ``[batch, seq_len, input_dim]`` (unused; BasePINN hook
                signature compatibility).
            hidden: ``[batch, hidden_dim]`` pooled encoder state from
                :meth:`_encode`.
            pred: ``[batch, 1]`` mean-head output (unused — a separate
                head drives ``log_var``).
            metadata: caller-supplied dict; never mutated in place.

        Returns:
            Shallow copy of *metadata* with an added
            ``"log_var"`` key holding ``[batch, 1]`` raw log-variance.
            No activation is applied — the constraint exponentiates and
            clamps internally so the head can represent negative log
            variances (typical on z-scored daily returns: ``log σ² ≈ −9``).
        """
        enriched = dict(metadata)
        enriched["log_var"] = self.log_var_head(hidden)
        return enriched
