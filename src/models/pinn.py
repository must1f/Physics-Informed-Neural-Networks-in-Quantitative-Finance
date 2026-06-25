"""Core PINN — single encoder + physics constraints via composition."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.losses.physics import PhysicsConstraint
from src.models.base_pinn import BasePINN


class PINNModel(BasePINN):
    """Core PINN — one recurrent encoder + composable physics constraints.

    The physics variant is determined entirely by the *constraints* list
    passed at construction (composition, not inheritance), which lets a
    single class cover 8 registry entries::

        constraints=[]                              → baseline_pinn
        constraints=[GBMConstraint()]               → gbm_pinn
        constraints=[OUConstraint()]                → ou_pinn
        constraints=[BlackScholesConstraint()]      → bs_pinn
        constraints=[GBM(), OU()]                   → gbm_ou_pinn
        constraints=[GBM(), OU(), BS(), Langevin()] → global_pinn
        constraints=[HawkesConstraint()]            → hawkes_pinn
        constraints=[Hawkes(), OU()]                → hawkes_ou_pinn

    Inputs (via :meth:`BasePINN.forward`):
        x: ``[batch, seq_len, input_dim]`` normalised features.
        metadata: any constraint-specific tensors (e.g. ``"target_mean"``,
            ``"target_std"``). The BS constraint additionally consumes the
            ``"volatilities"`` key injected by this class.

    Returns (from ``forward``):
        ``(pred, enriched_metadata)`` with ``pred`` of shape
        ``[batch, 1]`` on the normalised target scale.

    Learnable parameters:
        * ``_encoder``: LSTM or GRU stack (unconstrained).
        * ``prediction_head``: ``nn.Linear(hidden_dim, 1)``.
        * ``vol_head`` (only when BS constraint is attached):
          ``nn.Linear(hidden_dim, 1)``, passed through ``softplus`` to
          guarantee σ > 0 on the raw volatility scale.
        * plus any parameters owned by the attached ``constraints``
          (e.g. drift μ, mean-reversion θ, intensity baseline λ₀).
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
        """Construct a core PINN with the given encoder and physics constraints.

        Args:
            input_dim: number of features per timestep (``F``).
            hidden_dim: encoder hidden size ``H``; also the dim fed to
                ``prediction_head`` and, when present, ``vol_head``.
            num_layers: stacked encoder depth.
            dropout: inter-layer dropout applied by the RNN.
            encoder: ``"lstm"`` or ``"gru"`` — anything else raises
                ``ValueError``.
            constraints: physics constraints registered as child
                ``nn.Module``s via :class:`BasePINN`; ``None`` or ``[]``
                yields a plain recurrent baseline (baseline_pinn).
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

        self.prediction_head = nn.Linear(hidden_dim, 1)

        # Volatility head: feeds σ to BlackScholesConstraint risk-neutral drift.
        self._has_bs = any(c.name == "bs" for c in self.constraints)
        if self._has_bs:
            self.vol_head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: Tensor, metadata: dict | None = None,
    ) -> tuple[Tensor, dict]:
        """Encode → predict → physics-metadata enrichment.

        Pass-through to :meth:`BasePINN.forward`. No gradient manipulation
        is performed on the input; the BS constraint uses a simple drift
        formula and does not require autograd through the encoder.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features on the
                trainer's device.
            metadata: dict produced by :func:`collate_fn`. When the BS
                constraint is active, must carry ``target_mean`` and
                ``target_std`` (scaler stats on the normalised target scale).

        Returns:
            ``(pred, enriched)`` identical in shape/meaning to
            :meth:`BasePINN.forward`. ``enriched`` additionally carries
            every key injected by :meth:`_build_physics_metadata`.
        """
        return super().forward(x, metadata)

    def _encode(self, x: Tensor) -> Tensor:
        """Run the recurrent encoder and pool to the last-layer final state.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features.

        Returns:
            ``[batch, hidden_dim]`` final hidden state of the top RNN
            layer (``h_n[-1]``). Cell state is discarded.
        """
        if self.encoder_type == "lstm":
            _, (h_n, _) = self._encoder(x)
        else:
            _, h_n = self._encoder(x)
        return h_n[-1]  # [B, H]

    def _predict(self, hidden: Tensor) -> Tensor:
        """Project the pooled hidden state to a scalar forecast.

        Args:
            hidden: ``[batch, hidden_dim]`` from :meth:`_encode`.

        Returns:
            ``[batch, 1]`` prediction on the normalised target scale
            (z-scored next-step log return).
        """
        return self.prediction_head(hidden)

    def _build_physics_metadata(
        self, x: Tensor, hidden: Tensor, pred: Tensor, metadata: dict,
    ) -> dict:
        """Inject BS metadata when the BS constraint is active.

        No-op pass-through unless a constraint named ``"bs"`` is registered.
        When active, returns a shallow copy of *metadata* with two keys added:

            * ``"volatilities"``: ``[batch, 1]`` σ > 0 via
              ``softplus(vol_head(hidden))`` — consumed by
              :class:`BlackScholesConstraint` as σ in the Itô correction.
            * ``"prices"`` (when absent from *metadata*): ``[batch, seq_len]``
              raw close prices recovered by inverting the StandardScaler
              transform on the price feature column, using ``price_mean``,
              ``price_std``, and ``price_feature_idx`` from *metadata*.
              When ``prices`` is already present (supplied by the dataset),
              it is kept as-is — no re-computation.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features. Used to
               recover raw prices when the dataset did not supply them directly.
            hidden: ``[batch, hidden_dim]`` pooled encoder state.
            pred: ``[batch, 1]`` scalar forecast (unused).
            metadata: caller-supplied dict; never mutated in place.

        Returns:
            Original *metadata* when no BS constraint is attached; new dict
            with ``"volatilities"`` and (if needed) ``"prices"`` added.
        """
        if not self._has_bs:
            return metadata

        enriched = dict(metadata)
        enriched["volatilities"] = F.softplus(self.vol_head(hidden))

        if "prices" not in enriched:
            # Recover raw prices from the normalised feature column.
            # price_raw = x_norm * price_std + price_mean (inverse StandardScaler).
            idx = int(enriched.get("price_feature_idx", 0))
            p_mean = enriched.get("price_mean", 0.0)
            p_std = enriched.get("price_std", 1.0)
            if isinstance(p_mean, float):
                p_mean = x.new_tensor(p_mean)
            if isinstance(p_std, float):
                p_std = x.new_tensor(p_std)
            p_mean = p_mean.view(-1, 1)
            p_std = p_std.view(-1, 1)
            enriched["prices"] = x[:, :, idx] * p_std + p_mean  # [B, seq_len]

        return enriched
