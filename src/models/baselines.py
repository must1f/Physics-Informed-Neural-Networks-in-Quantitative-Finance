"""Neural baselines — one parametric class, five encoder architectures."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# Private helper encoders
# ═══════════════════════════════════════════════════════════════════════════


class _AttentionLSTMEncoder(nn.Module):
    """LSTM + Bahdanau additive attention pooled over all timesteps.

    Computes attention scores ``e_t = v^T tanh(W h_t)`` across the LSTM
    outputs, softmaxes them over the time axis, and returns the
    attention-weighted sum of hidden states. The most recent softmax
    vector is cached on ``self.last_attention_weights`` (detached) for
    interpretability / figures — not used in the training graph.

    Learnable parameters:
        * ``lstm``: standard stacked ``nn.LSTM`` weights/biases.
        * ``W``: ``nn.Linear(hidden_dim, hidden_dim)`` — unconstrained.
        * ``v``: ``nn.Linear(hidden_dim, 1, bias=False)`` — unconstrained.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float,
    ) -> None:
        """Build attention-LSTM encoder.

        Args:
            input_dim: number of features per timestep (``F``).
            hidden_dim: LSTM hidden size ``H`` (also attention dim).
            num_layers: stacked LSTM depth.
            dropout: inter-layer LSTM dropout in ``[0, 1)``.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
        )
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.last_attention_weights: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Attention-pool an LSTM run over the input window.

        Args:
            x: ``[batch, seq_len, input_dim]`` float tensor of
                normalised features.

        Returns:
            Context vector of shape ``[batch, hidden_dim]`` — the
            softmax-weighted sum of per-timestep LSTM states. Side
            effect: ``self.last_attention_weights`` is set to the
            detached ``[batch, seq_len]`` softmax tensor.
        """
        output, _ = self.lstm(x)                        # [B, T, H]
        energy = torch.tanh(self.W(output))             # [B, T, H]
        scores = self.v(energy).squeeze(-1)             # [B, T]
        weights = torch.softmax(scores, dim=1)          # [B, T]
        self.last_attention_weights = weights.detach()
        context = torch.bmm(weights.unsqueeze(1), output).squeeze(1)  # [B, H]
        return context


class _TransformerEncoder(nn.Module):
    """Input projection + learnable positional encoding + causal transformer.

    Projects ``input_dim`` features to ``hidden_dim``, adds a learnable
    positional embedding (sliced to the actual sequence length), and
    runs a stack of ``TransformerEncoderLayer`` blocks with a causal
    (square subsequent) attention mask so timestep ``t`` cannot attend
    to ``t+1…T-1``. The representation at the final timestep is
    returned as the pooled summary.

    Fixed hyperparameters: ``nhead=4``, ``dim_feedforward=4*hidden_dim``.

    Learnable parameters:
        * ``input_proj``: ``nn.Linear(input_dim, hidden_dim)``.
        * ``pos_encoding``: ``[1, max_seq_len, hidden_dim]`` parameter,
          unconstrained (init ``randn``).
        * ``transformer``: stacked encoder layer weights/biases.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int = 512,
    ) -> None:
        """Build causal transformer encoder.

        Args:
            input_dim: number of features per timestep (``F``).
            hidden_dim: model dimension ``H`` (== d_model).
            num_layers: number of stacked encoder layers.
            dropout: dropout rate inside each encoder layer.
            max_seq_len: upper bound on sequence length for which
                positional embeddings are allocated (``seq_len ≤ max_seq_len``
                is required at forward time).
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run the causal transformer and return the last-token summary.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features;
                ``seq_len`` must be ``≤ max_seq_len``.

        Returns:
            ``[batch, hidden_dim]`` representation at timestep
            ``seq_len - 1`` after causal self-attention.
        """
        T = x.size(1)
        x = self.input_proj(x) + self.pos_encoding[:, :T, :]   # [B, T, H]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        out = self.transformer(x, mask=mask)                    # [B, T, H]
        return out[:, -1, :]                                    # [B, H]


# ═══════════════════════════════════════════════════════════════════════════
# Public class
# ═══════════════════════════════════════════════════════════════════════════


class BaselineModel(nn.Module):
    """Unified neural baseline — one class, five interchangeable encoders.

    Acts as the non-physics control arm for the dissertation's PINN
    comparisons. The encoder is chosen by the ``arch`` string at
    construction (strategy pattern); a shared linear head produces the
    forecast. There are no physics constraints and no metadata output,
    which keeps this class decoupled from :class:`BasePINN`.

    Input:
        ``x``: ``[batch, seq_len, input_dim]`` float tensor of
        normalised features (z-scored log returns + engineered
        features). ``requires_grad`` is not required.

    Output:
        ``[batch, 1]`` prediction on the raw log-return scale.
        Targets are never z-scored in this project's pipeline
        (``TimeSeriesDataset`` sets ``target_mean=0.0``,
        ``target_std=1.0`` — identity stats).

    Learnable parameters:
        * ``encoder``: weights of the selected architecture (LSTM,
          GRU, BiLSTM, AttentionLSTM, or causal Transformer).
        * ``prediction_head``: ``nn.Linear(hidden_dim, 1)`` for all
          archs except BiLSTM, where it is ``nn.Linear(2*hidden_dim, 1)``
          — the full concatenated forward+backward representation is fed
          directly to the head with no intermediate projection
          (Fischer & Krauss, 2018).
    """

    VALID_ARCHS = ("lstm", "gru", "bilstm", "attention_lstm", "transformer")

    def __init__(
        self,
        arch: str,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """Construct a baseline model with the given encoder architecture.

        Args:
            arch: encoder architecture; one of :attr:`VALID_ARCHS`.
            input_dim: number of input features per timestep ``F`` — must
                match the feature dimension of the input tensor
                ``[batch, seq_len, F]`` supplied at forward time.
            hidden_dim: encoder hidden size ``H``; also the input dimension
                of ``prediction_head`` (``2H`` for ``bilstm``).
            num_layers: number of stacked encoder layers.
            dropout: inter-layer dropout in ``[0, 1)``; applied by the
                underlying RNN / transformer encoder layer.
        """
        super().__init__()
        if arch not in self.VALID_ARCHS:
            raise ValueError(
                f"Unknown arch '{arch}'. Choose from {self.VALID_ARCHS}"
            )
        self.arch = arch
        self.encoder = self._build_encoder(arch, input_dim, hidden_dim, num_layers, dropout)

        # BiLSTM concatenates forward + backward final states → 2H input to head
        head_in = hidden_dim * 2 if arch == "bilstm" else hidden_dim
        self.prediction_head = nn.Linear(head_in, 1)

    # ── Encoder factory ────────────────────────────────────────────

    @staticmethod
    def _build_encoder(
        arch: str, input_dim: int, hidden_dim: int, num_layers: int, dropout: float,
    ) -> nn.Module:
        """Build the encoder module selected by *arch* (strategy factory).

        Args:
            arch: one of :attr:`BaselineModel.VALID_ARCHS`.
            input_dim: features per timestep.
            hidden_dim: encoder hidden size ``H``.
            num_layers: stacked depth.
            dropout: inter-layer dropout (applied by the underlying
                RNN/transformer).

        Returns:
            An ``nn.Module`` that consumes ``[batch, seq_len, input_dim]``
            and exposes the hidden state needed by :meth:`_encode`
            (full sequence for LSTM/GRU/BiLSTM; already-pooled
            ``[batch, H]`` for ``attention_lstm`` and ``transformer``).
        """
        if arch == "lstm":
            return nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        if arch == "gru":
            return nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        if arch == "bilstm":
            return nn.LSTM(
                input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout, bidirectional=True,
            )
        if arch == "attention_lstm":
            return _AttentionLSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        # arch == "transformer"
        return _TransformerEncoder(input_dim, hidden_dim, num_layers, dropout)

    # ── Forward ────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """Encode the window and project to a scalar forecast.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features.

        Returns:
            ``[batch, 1]`` prediction on the raw log-return scale.
            Unlike :class:`BasePINN` subclasses, no metadata dict is
            returned — this model has no physics residuals.
        """
        hidden = self._encode(x)
        return self.prediction_head(hidden)

    def _encode(self, x: Tensor) -> Tensor:
        """Reduce a windowed input to a single ``[batch, hidden_dim]`` vector.

        Dispatch rules:
            * ``attention_lstm`` / ``transformer``: encoder already
              returns ``[batch, H]``; pass through.
            * ``bilstm``: concatenate final forward + backward hidden
              states → returns ``[batch, 2H]`` (no projection — the full
              bidirectional representation is fed directly to the head).
            * ``lstm`` / ``gru``: take the last layer's final hidden
              state ``h_n[-1]`` of shape ``[batch, H]``.

        Args:
            x: ``[batch, seq_len, input_dim]`` normalised features.

        Returns:
            ``[batch, hidden_dim]`` pooled representation for lstm/gru/
            attention_lstm/transformer; ``[batch, 2*hidden_dim]`` for
            bilstm.
        """
        if self.arch in ("attention_lstm", "transformer"):
            return self.encoder(x)                             # [B, H]
        if self.arch == "bilstm":
            _, (h_n, _) = self.encoder(x)
            h_fwd = h_n[-2]                                    # [B, H]
            h_bwd = h_n[-1]                                    # [B, H]
            return torch.cat([h_fwd, h_bwd], dim=-1)           # [B, 2H]
        if self.arch == "lstm":
            _, (h_n, _) = self.encoder(x)
            return h_n[-1]                                     # [B, H]
        # gru
        _, h_n = self.encoder(x)
        return h_n[-1]                                         # [B, H]
