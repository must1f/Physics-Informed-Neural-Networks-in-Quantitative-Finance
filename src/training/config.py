"""Training configuration — single source of truth from dissertation.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WalkForwardConfig:
    """Walk-forward cross-validation settings.

    Controls the expanding-window fold structure used by
    :func:`src.training.walk_forward.run_walk_forward`.

    Attributes:
        test_years: Calendar years used as test windows, one per fold.
            Each fold trains on all data up to (test_year - 1) minus the
            val buffer. Default covers 2018–2023 (6 folds, one per year).
        val_months: Number of calendar months immediately before the test
            year reserved for validation. Default 4 (~84 trading days).
            The walk-forward runner enforces a minimum of 4 to prevent
            look-ahead; values below 4 are silently raised to 4.
    """
    test_years: list[int] = field(default_factory=lambda: [2018, 2019, 2020, 2021, 2022, 2023])
    val_months: int = 4


@dataclass
class TrainingConfig:
    """All experiment hyperparameters in one place.

    Loaded from ``configs/dissertation.yaml`` via :func:`load_config`.
    Field groups mirror the YAML top-level sections (``training``,
    ``model_defaults``, ``physics``, ``data``, ``evaluation``, ``seeds``,
    ``walk_forward``).

    Every training run in the project — notebook, backend, or CLI — should
    receive a ``TrainingConfig`` rather than consuming bare globals or
    hard-coded constants.

    The ``aux_tickers`` field (e.g. ``["^VIX", "^TNX"]``) is read by
    :func:`src.data.pipeline.load_features` to fetch optional macro series
    passed to :func:`src.data.features.compute_features`. Without it the
    VIX/TNX feature columns are never produced, causing a ``KeyError`` when
    the runner tries to index them from the feature DataFrame.

    Attributes:
        epochs: Maximum training epochs per run. Range: [1, ∞).
        batch_size: Mini-batch size for ``DataLoader``. Range: [1, ∞).
        learning_rate: Initial learning rate for ``Adam``. Range: (0, 1).
        weight_decay: L2 regularisation coefficient for ``Adam``. Range: [0, 1).
        patience: Early-stopping patience in epochs (no improvement in
            data-only val loss). Range: [1, ∞).
        gradient_clip: Max gradient norm for ``clip_grad_norm_``. Range: (0, ∞).
        scheduler: LR scheduler type (currently only ``"plateau"`` is wired up
            in :class:`~src.training.trainer.Trainer`).
        scheduler_patience: Epochs before ``ReduceLROnPlateau`` fires. Range: [1, ∞).
        scheduler_factor: LR reduction factor on plateau. Range: (0, 1).
        hidden_dim: LSTM/GRU hidden state width. Range: [1, ∞).
        num_layers: Number of stacked LSTM/GRU layers. Range: [1, ∞).
        dropout: Recurrent dropout probability. Range: [0.0, 1.0).
        encoder: Encoder type passed to the model registry (``"lstm"`` or
            ``"gru"``).
        physics_lambdas: Per-constraint weighting dict. Keys: ``"gbm"``,
            ``"ou"``, ``"bs"``, ``"langevin"``, ``"hawkes"``. Values in (0, ∞);
            ``bs`` defaults to 0.01 (smaller because the BS residual scale is
            already normalised by option price).
        warmup_epochs: Epochs for curriculum-based physics warmup. ``0``
            disables warmup (scale always 1.0). Range: [0, ∞).
        warmup_strategy: Warmup schedule shape passed to
            :class:`~src.training.scheduler.PhysicsScheduler`. One of
            ``"linear"``, ``"cosine"``, ``"step"``.
        tickers: Primary ticker symbols for data download.
        aux_tickers: Optional auxiliary tickers (e.g. ``["^VIX", "^TNX"]``)
            fetched to build macro feature columns. Empty list disables macro
            features.
        start_date: Data download start date (ISO 8601, inclusive).
        end_date: Data download end date (ISO 8601, inclusive).
        sequence_length: Look-back window in trading days fed to the encoder.
            Range: [1, ∞).
        split_ratios: ``(train, val, test)`` proportions. Must sum to 1.0.
        features: Explicit feature column list passed to
            :class:`~src.data.dataset.TimeSeriesDataset`. Empty list = use all
            available columns.
        transaction_cost: Round-trip cost fraction per trade (e.g. 0.001 =
            10 bp). Range: [0.0, 1.0).
        risk_free_rate: Annualised risk-free rate for Sharpe/Sortino
            computation. Range: [0.0, 1.0).
        trading_days_per_year: Annualisation factor (252 for equities).
        seeds: Random seeds for multi-seed reproducibility. Each seed
            generates one independent training run per fold.
        walk_forward: Nested :class:`WalkForwardConfig` with fold structure.
    """
    # Training loop
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    patience: int = 20
    gradient_clip: float = 0.5

    # LR scheduler
    scheduler: str = "plateau"
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5

    # Model defaults
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    encoder: str = "lstm"

    # Physics
    physics_lambdas: dict[str, float] = field(default_factory=lambda: {
        "gbm": 0.1, "ou": 0.1, "bs": 0.01, "langevin": 0.05, "hawkes": 0.1,
    })
    warmup_epochs: int = 20
    warmup_strategy: str = "cosine"

    # Data
    tickers: list[str] = field(default_factory=lambda: ["AAPL"])
    aux_tickers: list[str] = field(default_factory=list)
    start_date: str = "2015-01-01"
    end_date: str = "2024-12-31"
    sequence_length: int = 60
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    features: list[str] = field(default_factory=list)

    # Evaluation
    transaction_cost: float = 0.001
    risk_free_rate: float = 0.02
    trading_days_per_year: int = 252

    # Seeds
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])

    # Walk-forward evaluation
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


def load_config(
    path: Path,
    overrides: dict[str, Any] | None = None,
) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file with optional overrides.

    Flattens the nested YAML sections (``training``, ``model_defaults``,
    ``physics``, ``data``, ``evaluation``, ``seeds``, ``walk_forward``) into
    a flat dict, constructs nested config objects (e.g. ``WalkForwardConfig``)
    where needed, applies any caller-supplied overrides, then filters to only
    the fields declared on ``TrainingConfig`` before constructing the instance.

    Args:
        path: Absolute path to the YAML config file (e.g.
            ``configs/dissertation.yaml``).
        overrides: Optional ``{field_name: value}`` dict applied after YAML
            parsing; values must match the declared field type.

    Returns:
        A fully populated ``TrainingConfig`` instance. The ``walk_forward``
        field is a ``WalkForwardConfig`` object; all other fields are scalars,
        lists, tuples, or dicts as declared on ``TrainingConfig``.
    """
    raw = yaml.safe_load(path.read_text())

    flat: dict[str, Any] = {}
    # Flatten nested YAML structure
    if "training" in raw:
        for k, v in raw["training"].items():
            flat[k] = v
    if "model_defaults" in raw:
        for k, v in raw["model_defaults"].items():
            flat[k] = v
    if "physics" in raw:
        p = raw["physics"]
        flat["physics_lambdas"] = p.get("lambdas", {})
        flat["warmup_epochs"] = p.get("warmup_epochs", 20)
        flat["warmup_strategy"] = p.get("warmup_strategy", "cosine")
    if "data" in raw:
        d = raw["data"]
        flat["tickers"] = d.get("tickers", [])
        flat["aux_tickers"] = d.get("aux_tickers", [])
        flat["start_date"] = d.get("start_date", "2015-01-01")
        flat["end_date"] = d.get("end_date", "2024-12-31")
        flat["sequence_length"] = d.get("sequence_length", 60)
        ratios = d.get("split_ratios", [0.70, 0.15, 0.15])
        flat["split_ratios"] = tuple(ratios)
    if "features" in raw:
        flat["features"] = raw["features"]
    if "evaluation" in raw:
        e = raw["evaluation"]
        flat["transaction_cost"] = e.get("transaction_cost", 0.001)
        flat["risk_free_rate"] = e.get("risk_free_rate", 0.02)
        flat["trading_days_per_year"] = e.get("trading_days_per_year", 252)
    if "seeds" in raw:
        flat["seeds"] = raw["seeds"]

    wf_raw = raw.get("walk_forward", {})
    flat["walk_forward"] = WalkForwardConfig(**wf_raw) if wf_raw else WalkForwardConfig()

    if overrides:
        flat.update(overrides)

    # Only pass known fields to the dataclass
    valid = {f.name for f in TrainingConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in flat.items() if k in valid}
    return TrainingConfig(**filtered)
