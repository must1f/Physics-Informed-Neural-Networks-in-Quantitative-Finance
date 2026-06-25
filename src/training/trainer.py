"""Universal trainer — one class for baselines and PINNs.

Detects whether the model is a :class:`BasePINN` (``forward`` returns
``(pred, enriched_metadata)``) or a plain ``nn.Module`` (``forward``
returns ``pred``) and adapts the forward call. The same ``fit`` loop
handles early stopping, best-model checkpointing, LR scheduling via
``ReduceLROnPlateau``, gradient clipping, and optional curriculum-based
physics-lambda warmup through :class:`PhysicsScheduler`.
"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.losses.composite import CompositeLoss
from src.models.base_pinn import BasePINN
from src.training.result import EpochMetrics, TrainingResult
from src.training.scheduler import PhysicsScheduler
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """Universal training loop for baselines and PINNs.

    Args:
        model: Any ``nn.Module``. If it is a :class:`BasePINN` subclass,
            ``forward(x, metadata)`` is called; otherwise ``forward(x)``.
        loss_fn: :class:`CompositeLoss` instance combining data loss and
            (optionally) physics constraint residuals.
        lr: Initial learning rate for ``Adam``.
        weight_decay: L2 regularisation coefficient.
        gradient_clip: Max gradient norm for ``clip_grad_norm_``. ``None``
            disables clipping.
        device: PyTorch device string (``"cpu"`` / ``"cuda"``). Auto-
            detected when ``None``.
        scheduler_patience: Epochs without improvement before
            ``ReduceLROnPlateau`` reduces LR.
        scheduler_factor: Factor by which LR is reduced on plateau.
        physics_scheduler: Optional :class:`PhysicsScheduler` for
            curriculum-based physics warmup. Ignored for baselines.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: CompositeLoss,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        gradient_clip: float | None = None,
        device: str | torch.device | None = None,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        physics_scheduler: PhysicsScheduler | None = None,
    ) -> None:
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.gradient_clip = gradient_clip
        self.physics_scheduler = physics_scheduler
        self._is_pinn = isinstance(model, BasePINN)

        self.optimizer = Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min",
            patience=scheduler_patience, factor=scheduler_factor,
        )

        self._current_epoch = 0
        self._best_val_loss = float("inf")
        self._best_state: dict[str, Any] | None = None

    # ── Forward dispatch ──────────────────────────────────────────────

    def _forward(
        self, x: torch.Tensor, metadata: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Dispatch forward call based on model type.

        Returns:
            ``(pred, metadata)`` — for PINNs the metadata may be enriched
            by ``_build_physics_metadata``; for baselines it is passed
            through unchanged.
        """
        if self._is_pinn:
            pred, enriched = self.model(x, metadata)
            return pred, enriched
        pred = self.model(x)
        return pred, metadata

    def _physics_scale(self) -> float:
        """Current curriculum warmup multiplier (0.0 → 1.0)."""
        if self.physics_scheduler is None:
            return 1.0
        return self.physics_scheduler.get_scale(self._current_epoch)

    # ── Epoch-level methods ───────────────────────────────────────────

    def train_epoch(self, dataloader: DataLoader) -> EpochMetrics:
        """Run one training epoch over *dataloader*.

        Args:
            dataloader: Yields ``(x, y, metadata)`` triples (see
                :func:`src.data.dataset.collate_fn`).

        Returns:
            :class:`EpochMetrics` with averaged ``train_loss`` and
            per-term ``train_breakdown``. ``val_loss`` is left at 0.0
            (filled by :meth:`validate`).
        """
        self.model.train()
        total_loss = 0.0
        total_breakdown: dict[str, float] = {}
        n_batches = 0
        scale = self._physics_scale()
        attn_sum = 0.0
        attn_count = 0

        for x, y, metadata in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            metadata = self._meta_to_device(metadata)

            pred, enriched = self._forward(x, metadata)
            loss, breakdown = self.loss_fn(pred, y, enriched)

            # Apply physics scale to the physics portion
            if scale < 1.0 and self._is_pinn:
                data_loss = breakdown.get("data", loss)
                physics_total = loss - data_loss
                loss = data_loss + scale * physics_total

            self.optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip,
                )

            self.optimizer.step()

            total_loss += loss.item()
            for k, v in breakdown.items():
                val = v.item() if torch.is_tensor(v) else float(v)
                total_breakdown[k] = total_breakdown.get(k, 0.0) + val
            n_batches += 1

            # StackedPINN exposes softmax attention weights [B, 2] after
            # each forward pass (col 0 = LSTM, col 1 = GRU). Accumulate the
            # batch-mean LSTM weight so EpochMetrics.attention_weight
            # records the per-epoch α consumed by nb8's logging cell.
            attn_w = getattr(self.model, "last_attention_weights", None)
            if torch.is_tensor(attn_w) and attn_w.ndim == 2 and attn_w.shape[1] == 2:
                attn_sum += float(attn_w[:, 0].mean().item())
                attn_count += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_breakdown = {k: v / max(n_batches, 1) for k, v in total_breakdown.items()}
        avg_attn = (attn_sum / attn_count) if attn_count > 0 else float("nan")

        return EpochMetrics(
            epoch=self._current_epoch,
            train_loss=avg_loss,
            val_loss=0.0,
            train_breakdown=avg_breakdown,
            physics_scale=scale,
            lr=self.optimizer.param_groups[0]["lr"],
            attention_weight=avg_attn,
        )

    def validate(self, dataloader: DataLoader) -> EpochMetrics:
        """Run a validation pass (no parameter updates).

        Autograd is kept **enabled** when the model carries a Black–Scholes
        constraint, because :meth:`BlackScholesConstraint.residual` calls
        ``autograd.grad(V, inputs, create_graph=True)`` to recover ∂V/∂S.
        Under the global ``torch.no_grad()`` used by every other validation
        path, ``V`` has no ``grad_fn`` and the autograd call fails with
        ``"element 0 of tensors does not require grad and does not have a
        grad_fn"``. Validation still performs **no parameter updates**:
        we never call ``loss.backward()`` here, and losses are summed via
        ``.item()`` so no autograd graph is retained across batches.

        Non-BS models keep the ``torch.no_grad()`` fast-path — they pay
        no memory / compute tax for this change.

        Args:
            dataloader: Yields ``(x, y, metadata)`` triples.

        Returns:
            :class:`EpochMetrics` with averaged ``val_loss`` (composite
            data + Σλᵢ·physicsᵢ) and per-term ``val_breakdown``. The
            breakdown's ``"data"`` key holds the unweighted MSE on the
            prediction target and is the model-selection metric used by
            :meth:`fit` for best-epoch checkpointing (F3, 2026-04-19).
            ``train_loss`` is left at 0.0.
        """
        self.model.eval()
        total_loss = 0.0
        total_breakdown: dict[str, float] = {}
        n_batches = 0

        needs_grad = bool(getattr(self.model, "_has_bs", False))
        ctx = torch.enable_grad() if needs_grad else torch.no_grad()

        with ctx:
            for x, y, metadata in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                metadata = self._meta_to_device(metadata)

                pred, enriched = self._forward(x, metadata)
                loss, breakdown = self.loss_fn(pred, y, enriched)

                total_loss += loss.item()
                for k, v in breakdown.items():
                    val = v.item() if torch.is_tensor(v) else float(v)
                    total_breakdown[k] = total_breakdown.get(k, 0.0) + val
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_breakdown = {k: v / max(n_batches, 1) for k, v in total_breakdown.items()}

        return EpochMetrics(
            epoch=self._current_epoch,
            train_loss=0.0,
            val_loss=avg_loss,
            val_breakdown=avg_breakdown,
            physics_scale=self._physics_scale(),
            lr=self.optimizer.param_groups[0]["lr"],
        )

    # ── Full training run ─────────────────────────────────────────────

    def fit(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epochs: int = 200,
        patience: int = 20,
    ) -> TrainingResult:
        """Full training loop with early stopping and best-model checkpointing.

        **Best-epoch selection (F3, 2026-04-19).** The checkpointed
        "best" weights minimise the **data-only** validation loss
        (``val_breakdown["data"]``), not the composite ``val_loss``.
        Physics constraints still contribute to the training gradient,
        but they do not confound model selection — so the restored
        weights are the ones that best fit the prediction target,
        independent of how heavily they are regularised toward physics
        priors. Early stopping uses the same data-only criterion. The
        LR scheduler continues to react to composite ``val_loss``
        because scheduler adaptation and checkpoint selection answer
        different questions.

        Args:
            train_dl: Training :class:`DataLoader` yielding
                ``(x, y, metadata)`` triples.
            val_dl: Validation :class:`DataLoader`.
            epochs: Maximum number of epochs.
            patience: Stop after this many consecutive epochs without
                an improvement in data-only val loss.

        Returns:
            :class:`TrainingResult` with full epoch history. Model
            weights are restored to the best-validation checkpoint
            before returning.
        """
        result = TrainingResult(model_name="", ticker="")
        no_improve = 0

        for epoch in range(epochs):
            self._current_epoch = epoch

            train_metrics = self.train_epoch(train_dl)
            val_metrics = self.validate(val_dl)

            combined = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics.train_loss,
                val_loss=val_metrics.val_loss,
                train_breakdown=train_metrics.train_breakdown,
                val_breakdown=val_metrics.val_breakdown,
                physics_scale=train_metrics.physics_scale,
                lr=train_metrics.lr,
                attention_weight=train_metrics.attention_weight,
            )
            result.history.append(combined)

            # LR scheduling — stays on composite val_loss: the scheduler
            # reacts to training dynamics (which legitimately include the
            # physics term) and is a different decision from model selection.
            self.lr_scheduler.step(val_metrics.val_loss)

            # Best-model checkpointing — data-only criterion (F3, 2026-04-19).
            # Composite val_loss lets many epochs tie on the plateau, making
            # tiebreaks random. See src/training/result.py::_selection_metric
            # for the full rationale.
            data_val = val_metrics.val_breakdown.get("data", val_metrics.val_loss)
            if data_val < self._best_val_loss:
                self._best_val_loss = data_val
                self._best_state = copy.deepcopy(self.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1

            logger.info(
                "Epoch {}/{} — train={:.6f} val={:.6f} scale={:.2f} lr={:.2e}",
                epoch + 1, epochs,
                train_metrics.train_loss, val_metrics.val_loss,
                train_metrics.physics_scale, train_metrics.lr,
            )

            # Early stopping
            if no_improve >= patience:
                logger.info(
                    "Early stopping at epoch {} (patience={})",
                    epoch + 1, patience,
                )
                break

        # Restore best weights
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return result

    # ── Helpers ────────────────────────────────────────────────────────

    def _meta_to_device(self, metadata: dict) -> dict:
        """Move all tensor values in *metadata* to ``self.device``.

        Non-tensor values (scalars, strings, arrays) are passed through
        unchanged so callers do not need to pre-filter the metadata dict.

        Args:
            metadata: Arbitrary key-value dict from the dataloader collate
                function. Tensor values are moved; all other values are
                copied as-is.

        Returns:
            New dict with the same keys as *metadata*; tensor values are
            on ``self.device``, all other values are unchanged.
        """
        out: dict = {}
        for k, v in metadata.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out
