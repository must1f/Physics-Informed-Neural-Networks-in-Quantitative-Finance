"""Data containers for training history and results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EpochMetrics:
    """Metrics captured after one training epoch.

    Stores both aggregate scalars (``train_loss``, ``val_loss``) and
    per-term breakdowns (``train_breakdown``, ``val_breakdown``) so
    downstream consumers can plot data-loss vs physics-loss curves.
    ``physics_scale`` records the :class:`PhysicsScheduler` multiplier
    that was active during this epoch (0.0 → 1.0 over warmup).
    ``attention_weight`` records the batch-mean LSTM softmax weight
    from :class:`StackedPINN.last_attention_weights` (column 0 of the
    two-encoder softmax); ``NaN`` for models that don't expose
    ``last_attention_weights`` (every non-stacked model).
    """
    epoch: int
    train_loss: float
    val_loss: float = 0.0
    train_breakdown: dict[str, float] = field(default_factory=dict)
    val_breakdown: dict[str, float] = field(default_factory=dict)
    physics_scale: float = 1.0
    lr: float = 0.0
    attention_weight: float = float("nan")

    def to_dict(self) -> dict[str, Any]:
        """Serialise this epoch's metrics to a plain JSON-compatible dict.

        Returns:
            dict with keys: ``epoch`` (int), ``train_loss`` (float),
            ``val_loss`` (float), ``train_breakdown`` (dict[str, float]),
            ``val_breakdown`` (dict[str, float]), ``physics_scale`` (float
            in ``[0.0, 1.0]``), ``lr`` (float), ``attention_weight``
            (float, ``NaN`` for non-stacked models).
        """
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_breakdown": self.train_breakdown,
            "val_breakdown": self.val_breakdown,
            "physics_scale": self.physics_scale,
            "lr": self.lr,
            "attention_weight": self.attention_weight,
        }


def _selection_metric(m: "EpochMetrics") -> float:
    """Model-selection criterion: data-only val loss with composite fallback.

    Returns ``val_breakdown["data"]`` when present (standard PINN /
    neural run — the unweighted MSE on the prediction target) and
    otherwise falls back to the aggregate ``val_loss`` (classical runs
    with empty breakdown). The data-only value is preferred because the
    composite ``val_loss = data + Σλᵢ·physicsᵢ`` lets many epochs tie
    on the loss plateau, making tiebreaks effectively random. See
    :attr:`TrainingResult.best_epoch` for the full rationale (F3,
    2026-04-19).

    Args:
        m: A single :class:`EpochMetrics` record from ``TrainingResult.history``.

    Returns:
        float: ``val_breakdown["data"]`` if present, otherwise ``val_loss``.
    """
    return m.val_breakdown.get("data", m.val_loss)


@dataclass
class TrainingResult:
    """Complete output from a single training run.

    Holds the full epoch-by-epoch ``history``, optional ``test_metrics``
    (populated after the run by the evaluation layer), and the path to
    the saved best-model checkpoint. The ``best_epoch`` and
    ``best_val_loss`` properties scan ``history`` for the epoch that
    minimises the **data-only** validation loss (see
    :func:`_selection_metric`).

    Attributes:
        model_name: Registry key identifying the trained model (e.g.
            ``"gbm_pinn"``).
        ticker: Ticker symbol this run was trained on (metadata only;
            not used in training logic).
        history: One :class:`EpochMetrics` per completed epoch. Empty
            for classical baselines which have no epoch loop.
        test_metrics: Evaluation metrics computed on the held-out test
            split. Keys: ``"rmse"``, ``"mae"``, ``"r_squared"``,
            ``"directional_accuracy"``, ``"sharpe"``, ``"sortino"``,
            ``"max_drawdown"``, ``"calmar"``. Values are plain floats.
        checkpoint_path: Absolute path to the saved ``.pt`` (neural) or
            pickled (classical) checkpoint, or ``None`` before saving.
        test_preds: 1-D array of shape ``(T,)`` — model predictions on
            the test split, on log-return scale. ``None`` until populated
            by the evaluation layer.
        test_actual: 1-D array of shape ``(T,)`` — ground-truth
            log-returns for the test split. Aligns index-for-index with
            ``test_preds``.
        equity_curve: Cumulative product of ``(1 + strategy_returns)``
            on the test split, shape ``(T,)``. Starts at 1.0.
        buy_hold_curve: Cumulative product of ``(1 + test_actual)``,
            shape ``(T,)``. Buy-and-hold benchmark; starts at 1.0.
    """
    model_name: str
    ticker: str = ""
    history: list[EpochMetrics] = field(default_factory=list)
    test_metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    # Populated by classical runner and optionally by neural runner for plotting.
    test_preds: np.ndarray | None = None
    test_actual: np.ndarray | None = None
    equity_curve: np.ndarray | None = None
    buy_hold_curve: np.ndarray | None = None

    @property
    def best_epoch(self) -> int | None:
        """Epoch index minimising **data-only** validation loss.

        Selection criterion is ``val_breakdown["data"]`` — the
        unweighted MSE on the prediction target, *excluding* physics
        residuals. This prevents the confounding observed in
        ``notebooks/4_core_pinns_extended.ipynb`` where many epochs tied
        on the composite ``val_loss`` (data + Σλᵢ·physicsᵢ) and
        best-epoch selection became effectively random. For classical
        runs with an empty breakdown, falls back to composite
        ``val_loss`` to preserve legacy behaviour.

        Returns
        -------
        int or None
            Epoch number of the best checkpoint, or ``None`` if
            ``history`` is empty.
        """
        if not self.history:
            return None
        return min(self.history, key=_selection_metric).epoch

    @property
    def history_df(self):
        """Training history as a :class:`pandas.DataFrame`.

        Flattens ``self.history`` (a ``list[EpochMetrics]``) into one row
        per epoch. Carries native EpochMetrics columns plus a derived
        ``physics_ratio`` column: fraction of total training loss attributable
        to physics constraints, computed as
        ``(total - data) / |total|`` from ``train_breakdown``.
        ``NaN`` for baseline (non-PINN) models where breakdown has no "total"
        key. nb8's Cell 5 logging cell reads ``history_df['attention_weight']``
        — ``NaN`` on every model except :class:`StackedPINN`. Empty DataFrame
        when ``self.history`` is empty (e.g. failed run).
        """
        import pandas as pd
        if not self.history:
            return pd.DataFrame()
        df = pd.DataFrame(self.history)
        if "train_breakdown" in df.columns:
            def _ratio(bd: dict) -> float:
                """Compute physics fraction from a single epoch's train_breakdown.

                Args:
                    bd: ``train_breakdown`` dict from one :class:`EpochMetrics`.
                        Expected keys: ``"data"`` (unweighted MSE) and ``"total"``
                        (data + Σλᵢ·physicsᵢ). Both are scalars (float).

                Returns:
                    ``(total - data) / |total|`` — the fraction of the epoch's
                    total training loss attributable to physics constraints.
                    ``NaN`` when either key is absent or ``total`` is zero
                    (baseline / non-PINN models have no "total" key).
                """
                if not isinstance(bd, dict):
                    return float("nan")
                data = bd.get("data")
                total = bd.get("total")
                if data is None or total is None or total == 0:
                    return float("nan")
                return float(total - data) / float(abs(total))
            df["physics_ratio"] = df["train_breakdown"].apply(_ratio)
        return df

    @property
    def best_val_loss(self) -> float | None:
        """Lowest **data-only** validation loss across all epochs.

        Matches the criterion used by :attr:`best_epoch`. Falls back to
        composite ``val_loss`` when ``val_breakdown["data"]`` is
        unavailable (classical / non-PINN runs).
        """
        if not self.history:
            return None
        return min(_selection_metric(m) for m in self.history)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-compatible).

        Arrays (test_preds, test_actual, equity_curve, buy_hold_curve) are
        converted to Python lists so the dict is directly JSON-serialisable
        without a ``default=str`` fallback. None values are preserved as None.
        """
        def _to_list(x: np.ndarray | None) -> list | None:
            return None if x is None else np.asarray(x).tolist()

        return {
            "model_name": self.model_name,
            "ticker": self.ticker,
            "history": [m.to_dict() for m in self.history],
            "test_metrics": self.test_metrics,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "checkpoint_path": self.checkpoint_path,
            "test_preds": _to_list(self.test_preds),
            "test_actual": _to_list(self.test_actual),
            "equity_curve": _to_list(self.equity_curve),
            "buy_hold_curve": _to_list(self.buy_hold_curve),
        }


@dataclass
class WalkForwardResult:
    """Aggregated output from a full walk-forward evaluation of one model.

    Produced by :func:`src.training.walk_forward.aggregate_walk_forward`
    after all fold×seed :class:`TrainingResult` artefacts have been written
    to disk. Fold metrics are seed-averaged first, then aggregated across
    folds for the grand mean and std.

    Attributes:
        model_name: Registry key (e.g. ``"gbm_pinn"``).
        fold_metrics: One dict per fold. Each dict maps metric name →
            seed-averaged value for that fold (e.g. ``{"rmse": 0.012,
            "sharpe": 1.1, ...}``). Length equals ``len(test_years)``.
        mean_metrics: Grand mean across folds for each metric.
        std_metrics: Grand std (ddof=1) across folds for each metric.
            Used directly in the dissertation results table (mean ± std).
        test_years: Calendar years used as test windows (e.g. ``[2018,
            ..., 2023]``). Aligns with ``fold_metrics`` by index.
        seeds: Seeds used in each fold (e.g. ``[42, 123, 456]``).
        test_preds: Concatenated OOS predictions across all folds (seed-0
            only). Populated by ``aggregate_walk_forward`` from per-fold
            JSONs. Persisted separately to ``wf_preds.json``; not included
            in ``wf_summary.json`` to keep it small.
        test_actual: Concatenated OOS actuals across all folds. Shape and
            ordering match ``test_preds``.

    Examples:
        # In a notebook cell — after run_walk_forward():
        from src.training.walk_forward import aggregate_walk_forward
        wf = aggregate_walk_forward("results/walk_forward", "gbm_pinn")
        print(f"Sharpe: {wf.mean_metrics['sharpe']:.3f} ± {wf.std_metrics['sharpe']:.3f}")
    """
    model_name: str
    fold_metrics: list[dict[str, float]]
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    test_years: list[int]
    seeds: list[int]
    test_preds: list[float] | None = None
    test_actual: list[float] | None = None

    @property
    def n_folds(self) -> int:
        """Number of folds in this walk-forward result.

        Returns:
            int: Length of ``fold_metrics`` list, equal to ``len(test_years)``.
        """
        return len(self.fold_metrics)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-compatible).

        Returns:
            dict with keys: ``model_name`` (str), ``fold_metrics``
            (list[dict]), ``mean_metrics`` (dict), ``std_metrics`` (dict),
            ``test_years`` (list[int]), ``seeds`` (list[int]).
        """
        return {
            "model_name": self.model_name,
            "fold_metrics": self.fold_metrics,
            "mean_metrics": self.mean_metrics,
            "std_metrics": self.std_metrics,
            "test_years": self.test_years,
            "seeds": self.seeds,
        }
