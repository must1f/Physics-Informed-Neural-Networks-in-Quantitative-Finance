"""Experiment entry point — config to trained model in one call.

Wires together the data pipeline (:mod:`src.data`), model registry
(:mod:`src.models.registry`), composite loss (:mod:`src.losses`), and
trainer (:mod:`src.training.trainer`) into a single
:func:`run_experiment` function that takes a :class:`TrainingConfig`,
a model name, and a feature DataFrame and returns a
:class:`TrainingResult` with full epoch history plus a saved checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import numpy as np

from src.data.dataset import TimeSeriesDataset, collate_fn
from src.losses.composite import CompositeLoss
from src.losses.data_losses import mse_loss
from src.models.base_pinn import BasePINN
from src.models.classical import (
    GARCHModel,
    HistoricalMeanModel,
    PersistenceModel,
    RandomWalkModel,
    is_classical,
)
from src.models.registry import build_model
from src.training.config import TrainingConfig
from src.training.result import TrainingResult
from src.training.scheduler import PhysicsScheduler
from src.training.trainer import Trainer
from src.utils.logger import get_logger
from src.utils.reproducibility import seed_everything

logger = get_logger(__name__)


def _bootstrap_mean(ret: np.ndarray, seed: int) -> float:
    """Seeded bootstrap estimate of ``mean(ret)``.

    Draws ``len(ret)`` samples from ``ret`` with replacement using a
    dedicated ``np.random.default_rng(seed)`` and returns the sample
    mean as a Python ``float``. Used to seed
    :class:`~src.losses.physics.OUConstraint` and
    :class:`~src.losses.physics.LangevinConstraint` ``mu_init`` so
    that multi-seed training receives a **different initial drift
    prior per seed** — the full-sample mean is seed-independent
    (the train/val/test split is deterministic) and therefore
    produces identical physics attractors for every seed, collapsing
    all runs to the same constant prediction (see the 2026-04-19
    audit of ``notebooks/4_core_pinns_extended.ipynb``).

    The bootstrap is a consistent estimator of the sample mean: for
    S&P 500 daily log-returns (σ ≈ 0.012, n ≈ 2 567) the per-seed
    standard error is ≈ 2.4 × 10⁻⁴, small relative to the true
    positive drift (~+5 × 10⁻⁴/day). So every seed still receives a
    physically plausible drift prior, just a slightly different one.

    Args:
        ret: 1-D ``np.ndarray`` of training-split log-returns (any dtype;
            cast to ``float64`` internally). Flattened if multi-dimensional.
            An empty array is tolerated and returns ``0.0``.
        seed: Experiment seed. The same seed always produces the same
            bootstrap sample (reproducibility guarantee).

    Returns:
        float: Bootstrap sample mean of *ret*. ``0.0`` if *ret* is empty.
    """
    arr = np.asarray(ret, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    return float(rng.choice(arr, size=arr.size, replace=True).mean())


def run_experiment(
    config: TrainingConfig,
    model_name: str,
    dataframe: pd.DataFrame,
    checkpoint_dir: Path | str = "results/checkpoints",
    seed: int = 42,
    ticker: str = "",
) -> TrainingResult:
    """Run a complete training experiment from config to checkpoint.

    Args:
        config: Training hyperparameters (loaded via :func:`load_config`
            or constructed directly).
        model_name: Key in ``MODEL_REGISTRY`` (e.g. ``"lstm"``,
            ``"gbm_pinn"``, ``"stacked_pinn"``).
        dataframe: Feature DataFrame from the data pipeline. Must
            contain all feature columns plus ``"Close"`` and
            ``"log_return"``.
        checkpoint_dir: Directory for saving model checkpoints and
            result JSON. Created if it does not exist.
        seed: Random seed for reproducibility.
        ticker: Ticker symbol (metadata only — stored in the result
            and checkpoint, not used for training logic).

    Returns:
        :class:`TrainingResult` with full training history,
        ``checkpoint_path`` pointing to the saved ``.pt`` file, and
        an adjacent ``{model_name}_result.json``.

    Raises:
        ValueError: If *model_name* is not in the registry.

    Notes
    -----
    For models with ``OUConstraint`` or ``LangevinConstraint`` (``ou_pinn``,
    ``gbm_ou_pinn``, ``global_pinn``, ``hawkes_ou_pinn``, and any
    ``stacked_pinn`` / ``residual_pinn`` stack containing those
    constraints), the constraint equilibrium mean is seeded from a
    **seed-varying bootstrap mean** over training-split log-returns
    (see :func:`_bootstrap_mean`). The deterministic full-sample mean
    was previously used but caused every multi-seed run to converge to
    the same attractor — the 2026-04-19 audit of
    ``notebooks/4_core_pinns_extended.ipynb`` found ``global_pinn``
    producing identical RMSE/DA/Sharpe across seeds {42, 123, 7}.
    Same seed still produces the same ``mu_init`` (reproducibility
    preserved); different seeds now produce physically plausible but
    distinct drift priors.

    For models with ``HawkesConstraint`` (``hawkes_pinn``, ``hawkes_ou_pinn``),
    the baseline intensity ``mu0`` is seeded from the **deterministic**
    training-split ``mean(r²)`` — it is a physical intensity *scale*,
    not a drift prior, so seed-varying would be inappropriate. This
    matches the empirical return-variance scale (``pred² ∼ σ² ∼ 1e-4``
    for daily returns). Without this seed, the legacy
    ``softplus(0) ≈ 0.69`` dominates ``pred²`` by ~3 orders of
    magnitude and saturates the physics term.
    """
    # Validate model name early — before expensive dataset construction
    from src.models.registry import MODEL_REGISTRY
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    # Classical baselines bypass the neural training loop entirely.
    if is_classical(model_name):
        return _run_classical_experiment(
            config, model_name, dataframe,
            Path(checkpoint_dir), seed, ticker,
        )

    seed_everything(seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── Build datasets ──────────────────────────────────────────────
    train_ds, val_ds, test_ds = TimeSeriesDataset.from_dataframe(
        dataframe,
        seq_len=config.sequence_length,
        target_col="log_return",
        split_ratios=config.split_ratios,
        feature_cols=config.features if config.features else None,
    )
    # num_workers=2 + pin_memory=True: overlaps CPU feature prep with GPU
    # compute. Measured ~3x throughput on Colab L4 vs defaults at batch=256.
    # persistent_workers=True keeps workers alive across epochs (saves the
    # ~1s per-epoch process-spawn cost that dominates short epochs).
    _dl_kwargs = dict(
        shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_dl = DataLoader(train_ds, batch_size=config.batch_size, **_dl_kwargs)
    val_dl = DataLoader(val_ds, batch_size=config.batch_size, **_dl_kwargs)

    # ── Seed OU / Langevin constraints with a SEED-VARYING bootstrap mean.
    # The full-sample mean is seed-independent (the train split is
    # deterministic and unshuffled), which collapsed multi-seed runs of
    # global_pinn onto the same physics attractor and produced identical
    # RMSE/DA/Sharpe across seeds {42, 123, 7} (see the 2026-04-19 audit
    # of notebooks/4_core_pinns_extended.ipynb). A bootstrap over `ret`
    # keyed by the experiment seed injects the variation the optimiser
    # needs while remaining a consistent estimator of the empirical
    # drift. See :func:`_bootstrap_mean` for the full rationale.
    ret = train_ds.metadata.get("returns")
    ret_arr = np.asarray(ret).reshape(-1) if ret is not None else np.array([])
    _ou_mu_init: float = _bootstrap_mean(ret_arr, seed=seed)

    # ── Seed HawkesConstraint.mu0 from train-set mean(r²) ──────────
    # Hawkes μ₀ is a physical intensity *scale* (not a drift prior) and
    # must track the empirical return-variance ~σ². Keeping this on the
    # deterministic full-sample mean(r²) is correct — F4 only
    # seed-varies OU/Langevin drift means.
    # Legacy softplus(0) ≈ 0.69 baseline intensity dominates pred² ∼ 1e-4
    # by ~3 orders; seeding at mean(r²) ≈ σ² puts initial λ on the same
    # scale so the physics residual is balanced from epoch 0.
    _hawkes_mu0_init: float | None = None
    if ret_arr.size > 0:
        _hawkes_mu0_init = float(np.mean(ret_arr ** 2))

    # ── Build model ─────────────────────────────────────────────────
    n_features = train_ds.features.shape[1]
    model = build_model(
        model_name,
        mu_init=_ou_mu_init,
        hawkes_mu0_init=_hawkes_mu0_init,
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    is_pinn = isinstance(model, BasePINN)

    # ── Build loss ──────────────────────────────────────────────────
    constraints = list(model.constraints) if is_pinn else []
    loss_fn = CompositeLoss(
        data_loss_fn=mse_loss,
        constraints=constraints,
        lambdas=config.physics_lambdas if is_pinn else None,
    )

    # ── Physics scheduler (PINNs with constraints only) ─────────────
    physics_scheduler = None
    if is_pinn and len(constraints) > 0:
        physics_scheduler = PhysicsScheduler(
            warmup_epochs=config.warmup_epochs,
            strategy=config.warmup_strategy,
        )

    # ── Train ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
        scheduler_patience=config.scheduler_patience,
        scheduler_factor=config.scheduler_factor,
        physics_scheduler=physics_scheduler,
    )

    logger.info(
        "[REAL TRAINING] {} on {} (epochs={}, seed={})",
        model_name, ticker or "data", config.epochs, seed,
    )

    result = trainer.fit(
        train_dl=train_dl,
        val_dl=val_dl,
        epochs=config.epochs,
        patience=config.patience,
    )
    result.model_name = model_name
    result.ticker = ticker

    # ── Save checkpoint ─────────────────────────────────────────────
    ckpt_path = checkpoint_dir / f"{model_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "config": {
                "input_dim": n_features,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "dropout": config.dropout,
                "epochs_trained": len(result.history),
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
            },
            "seed": seed,
            "ticker": ticker,
        },
        ckpt_path,
    )
    result.checkpoint_path = str(ckpt_path)

    # ── Evaluate on held-out test set ───────────────────────────────
    from src.evaluation.evaluator import evaluate_on_test
    (
        result.test_metrics,
        result.test_preds,
        result.test_actual,
        result.equity_curve,
        result.buy_hold_curve,
    ) = evaluate_on_test(
        model=model,
        test_ds=test_ds,
        device=trainer.device,
    )

    # ── Save result JSON ────────────────────────────────────────────
    json_path = checkpoint_dir / f"{model_name}_result.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

    logger.info(
        "[DONE] {} — best_epoch={}, best_val_loss={:.6f}",
        model_name, result.best_epoch, result.best_val_loss or float("nan"),
    )

    return result


def _run_classical_experiment(
    config: TrainingConfig,
    model_name: str,
    dataframe: pd.DataFrame,
    checkpoint_dir: Path,
    seed: int,
    ticker: str,
) -> TrainingResult:
    """Fit a classical baseline and return a fully-populated TrainingResult.

    Splits ``dataframe['log_return']`` by ``config.split_ratios`` (train /
    val / test). Val split is excluded from fitting — only training returns
    are used. Generates test predictions, builds sign-based equity curves,
    computes all evaluation metrics on the sign-based strategy returns
    ``sign(test_preds) * test_actual`` (not on buy-and-hold), and pickles the
    fitted model as a checkpoint.

    Args:
        config: :class:`~src.training.config.TrainingConfig` — only
            ``split_ratios`` is consumed here.
        model_name: One of ``"random_walk"``, ``"historical_mean"``,
            ``"persistence"``, ``"garch"``.
        dataframe: Feature DataFrame with a ``"log_return"`` column
            containing daily log-returns (any index). Only the
            ``"log_return"`` column is used; other columns are ignored.
        checkpoint_dir: Directory for pickle checkpoint and result JSON.
            Created if absent. Must be a :class:`pathlib.Path`.
        seed: Unused (classical models are deterministic). Kept for API
            parity with the neural runner so callers can loop uniformly.
        ticker: Ticker symbol stored on the result and log lines only.

    Returns:
        :class:`~src.training.result.TrainingResult` with populated
        ``test_metrics``, ``test_preds``, ``test_actual``,
        ``equity_curve`` (sign-strategy cumulative product),
        ``buy_hold_curve``, and ``checkpoint_path``.
        ``history`` is empty — classical models have no epoch loop.

    Side effects:
        Writes ``{checkpoint_dir}/{model_name}.pt`` (pickled model),
        ``{model_name}_result.json`` (result dump), and — if the fitted
        model exposes a ``.params()`` method (currently only
        :class:`~src.models.classical.GARCHModel`) —
        ``{model_name}_params.json`` (fitted-parameter sidecar for
        dissertation-appendix reporting).
    """
    import pickle
    from src.evaluation.metrics import compute_all_metrics

    returns = dataframe["log_return"].dropna().to_numpy(dtype=float)
    n = len(returns)
    train_end = int(n * config.split_ratios[0])
    val_end = train_end + int(n * config.split_ratios[1])

    train_returns = returns[:train_end]
    test_returns = returns[val_end:]

    _model_map = {
        "random_walk":     RandomWalkModel,
        "historical_mean": HistoricalMeanModel,
        "persistence":     PersistenceModel,
        "garch":           GARCHModel,
    }
    model = _model_map[model_name]()
    model.fit(train_returns)

    last_train_return = float(train_returns[-1]) if len(train_returns) else 0.0
    test_preds = model.predict(test_returns, last_train_return)
    test_actual = test_returns

    # Sign-based strategy: long when pred > 0, short when pred < 0, flat when pred == 0.
    strategy_returns = np.sign(test_preds) * test_actual
    equity_curve = np.cumprod(1.0 + strategy_returns)
    buy_hold_curve = np.cumprod(1.0 + test_actual)

    # Financial metrics MUST be computed on the strategy returns, not on
    # test_actual — otherwise every model (including the zero-forecast
    # random_walk) would report the same buy-and-hold Sharpe.
    test_metrics = compute_all_metrics(test_preds, test_actual, strategy_returns)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{model_name}.pt"
    with open(ckpt_path, "wb") as f:
        pickle.dump(model, f)

    # Optional sidecar: any classical model that exposes a .params()
    # method (currently only GARCHModel) gets its fitted parameters
    # written next to the checkpoint for dissertation-appendix reporting.
    if hasattr(model, "params") and callable(model.params):
        params_path = checkpoint_dir / f"{model_name}_params.json"
        params_path.write_text(json.dumps(model.params(), indent=2, default=str))

    result = TrainingResult(model_name=model_name, ticker=ticker)
    result.checkpoint_path = str(ckpt_path)
    result.test_metrics = test_metrics
    result.test_preds = test_preds
    result.test_actual = test_actual
    result.equity_curve = equity_curve
    result.buy_hold_curve = buy_hold_curve

    json_path = checkpoint_dir / f"{model_name}_result.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))

    logger.info(
        "[REAL TRAINING] {} (classical) on {} — DA={:.3f}, Sharpe={:.3f}",
        model_name, ticker or "data",
        test_metrics.get("directional_accuracy", float("nan")),
        test_metrics.get("sharpe", float("nan")),
    )

    return result
