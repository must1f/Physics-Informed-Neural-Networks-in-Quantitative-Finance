"""Walk-forward orchestration for multi-fold, multi-seed evaluation.

Coordinates WalkForwardSplitter, run_experiment, and artefact persistence
to produce statistically valid comparisons across 21 model variants.
"""
from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splitter import WalkForwardSplitter
from src.training.config import TrainingConfig
from src.training.result import TrainingResult, WalkForwardResult
from src.training.runner import run_experiment
from src.utils.logger import get_logger

logger = get_logger(__name__)

_METRIC_KEYS = (
    "rmse", "mae", "r_squared", "directional_accuracy",
    "sharpe", "sortino", "max_drawdown", "calmar",
)


def run_walk_forward(
    config: TrainingConfig,
    model_name: str,
    dataframe: pd.DataFrame,
    base_dir: str | Path = "results/walk_forward",
    ticker: str = "",
) -> WalkForwardResult:
    """Run all folds and seeds for one model; persist per-fold artefacts.

    Iterates over :class:`~src.data.splitter.WalkForwardSplitter` using the
    fold structure from ``config.walk_forward``. For each fold, all seeds in
    ``config.seeds`` are run via :func:`~src.training.runner.run_experiment`.

    **Crash-resume:** if ``{base_dir}/{model_name}/fold_{n}/seed_{s}/{model_name}_result.json``
    already exists, that fold×seed is skipped. Re-running after a Colab crash
    resumes from the next incomplete fold×seed pair.

    **Split ratios:** each fold concatenates train/val/test slices and passes
    proportional ``split_ratios`` to ``run_experiment`` via
    ``dataclasses.replace(config, split_ratios=...)``. This avoids any
    changes to the runner's interface.

    Args:
        config: Full ``TrainingConfig`` with a ``walk_forward`` attribute
            (``WalkForwardConfig``) containing ``test_years`` and ``val_months``.
            Also supplies ``config.seeds`` for multi-seed runs.
        model_name: Registry key, e.g. ``"gbm_pinn"``. Must be in ``MODEL_REGISTRY``.
        dataframe: Full feature DataFrame with ``DatetimeIndex`` and
            ``"log_return"`` column. Should span 2010-01-01 to 2023-12-31+
            to cover all default folds.
        base_dir: Root directory for walk-forward artefacts. Per-fold results
            land under ``{base_dir}/{model_name}/fold_{n}/seed_{s}/``.
        ticker: Ticker symbol stored in result JSONs (metadata only).

    Returns:
        :class:`~src.training.result.WalkForwardResult` with fold-level
        seed-averaged metrics, grand mean, and grand std across folds.
        Also writes ``{base_dir}/{model_name}/wf_summary.json``.

    Examples:
        # In a notebook cell:
        from src.training.walk_forward import run_walk_forward
        from src.training.config import load_config
        from pathlib import Path

        config = load_config(Path("configs/dissertation.yaml"))
        result = run_walk_forward(config, "gbm_pinn", df, base_dir="results/walk_forward")
        print(result.mean_metrics)
        all_wf_results["gbm_pinn"] = result
    """
    base_dir = Path(base_dir)
    wf_config = config.walk_forward
    seeds = config.seeds

    splitter = WalkForwardSplitter(
        dataframe,
        test_years=wf_config.test_years,
        val_months=wf_config.val_months,
    )

    for fold_idx, train_df, val_df, test_df in splitter:
        n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
        n_total = n_train + n_val + n_test
        fold_config = dataclasses.replace(
            config,
            split_ratios=(n_train / n_total, n_val / n_total, n_test / n_total),
        )
        fold_df = pd.concat([train_df, val_df, test_df])

        for seed in seeds:
            ckpt_dir = base_dir / model_name / f"fold_{fold_idx}" / f"seed_{seed}"
            sentinel = ckpt_dir / f"{model_name}_result.json"
            if sentinel.exists():
                logger.info(
                    "[SKIP] {} fold={} seed={} — artefacts already exist",
                    model_name, fold_idx, seed,
                )
                continue

            logger.info(
                "[RUN] {} fold={} (test_year={}) seed={}",
                model_name, fold_idx, wf_config.test_years[fold_idx], seed,
            )
            fold_result = run_experiment(
                fold_config, model_name, fold_df,
                checkpoint_dir=ckpt_dir,
                seed=seed,
                ticker=ticker,
            )
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            sentinel.write_text(json.dumps(fold_result.to_dict()))

    return aggregate_walk_forward(base_dir, model_name, seeds=seeds, test_years=wf_config.test_years)


def aggregate_walk_forward(
    base_dir: str | Path,
    model_name: str,
    seeds: list[int] | None = None,
    test_years: list[int] | None = None,
) -> WalkForwardResult:
    """Read persisted fold artefacts and compute grand mean ± std across folds.

    For each fold, averages metric values across seeds (simple mean). Then
    computes grand mean and std (ddof=1) across folds. Writes result as
    ``{base_dir}/{model_name}/wf_summary.json``.

    Args:
        base_dir: Root walk-forward artefact directory (same value passed to
            :func:`run_walk_forward`).
        model_name: Registry key (e.g. ``"gbm_pinn"``). Used to locate
            ``{base_dir}/{model_name}/fold_*/seed_*/{model_name}_result.json``.
        seeds: Seed list used during training. If ``None``, inferred from
            directory structure. Stored in result metadata only.
        test_years: Test year list. If ``None``, inferred from fold directory
            names (sorted). Stored in result metadata only.

    Returns:
        :class:`~src.training.result.WalkForwardResult` with:
        ``fold_metrics`` (seed-averaged per fold), ``mean_metrics`` (grand mean),
        ``std_metrics`` (grand std, ddof=1), ``test_years``, ``seeds``.
        All ``nanmean``/``nanstd`` calls over all-NaN slices (metrics absent from
        test fixtures) are silenced via ``warnings.catch_warnings`` to avoid
        ``RuntimeWarning: Mean of empty slice`` noise in logs.

    Raises:
        FileNotFoundError: If no fold directories exist under
            ``{base_dir}/{model_name}``.

    Examples:
        # In a notebook cell — after all folds are complete:
        from src.training.walk_forward import aggregate_walk_forward

        wf = aggregate_walk_forward("results/walk_forward", "gbm_pinn")
        print(f"Sharpe: {wf.mean_metrics['sharpe']:.3f} ± {wf.std_metrics['sharpe']:.3f}")
        all_wf_results["gbm_pinn"] = wf
    """
    base_dir = Path(base_dir)
    model_dir = base_dir / model_name

    fold_dirs = sorted(model_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold directories found under {model_dir}. "
            "Run run_walk_forward() first."
        )

    fold_metrics: list[dict[str, float]] = []
    all_preds:    list[float] = []
    all_actuals:  list[float] = []

    for fold_dir in fold_dirs:
        seed_metrics: list[dict[str, float]] = []
        fold_preds:   list[float] | None = None
        fold_actuals: list[float] | None = None

        for result_json in sorted(fold_dir.glob(f"seed_*/{model_name}_result.json")):
            data = json.loads(result_json.read_text())
            seed_metrics.append(data.get("test_metrics", {}))
            # Collect predictions from the first available seed per fold.
            if fold_preds is None and data.get("test_preds") and data.get("test_actual"):
                fold_preds   = data["test_preds"]
                fold_actuals = data["test_actual"]

        if not seed_metrics:
            continue
        fold_mean: dict[str, float] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for key in _METRIC_KEYS:
                vals = [m.get(key, float("nan")) for m in seed_metrics]
                fold_mean[key] = float(np.nanmean(vals))
        fold_metrics.append(fold_mean)

        if fold_preds is not None:
            all_preds.extend(fold_preds)
            all_actuals.extend(fold_actuals)

    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for key in _METRIC_KEYS:
            vals = np.array([f.get(key, float("nan")) for f in fold_metrics])
            mean_metrics[key] = float(np.nanmean(vals))
            std_metrics[key] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0

    if seeds is None:
        seed_dirs = sorted(fold_dirs[0].glob("seed_*"))
        seeds = [int(d.name.split("_")[1]) for d in seed_dirs]
    if test_years is None:
        test_years = [int(d.name.split("_")[1]) for d in fold_dirs]

    preds_out   = all_preds   if all_preds   else None
    actuals_out = all_actuals if all_actuals else None

    result = WalkForwardResult(
        model_name=model_name,
        fold_metrics=fold_metrics,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        test_years=test_years,
        seeds=seeds,
        test_preds=preds_out,
        test_actual=actuals_out,
    )

    summary_path = model_dir / "wf_summary.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2))
    logger.info("[DONE] {} walk-forward summary written to {}", model_name, summary_path)

    if preds_out is not None:
        preds_path = model_dir / "wf_preds.json"
        preds_path.write_text(json.dumps(
            {"model_name": model_name, "test_preds": preds_out, "test_actual": actuals_out},
            separators=(",", ":"),
        ))
        logger.info("[DONE] {} walk-forward predictions written to {}", model_name, preds_path)

    return result


def load_wf_preds(
    base_dir: str | Path,
    model_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Load concatenated OOS predictions from a persisted ``wf_preds.json``.

    Reads ``{base_dir}/{model_name}/wf_preds.json`` written by
    :func:`aggregate_walk_forward`. Returns ``(None, None)`` when the file
    does not exist so callers can skip missing models gracefully.

    Args:
        base_dir: Root walk-forward artefact directory (same value passed to
            :func:`run_walk_forward`), e.g. ``"results/walk_forward"`` or
            ``"/content/drive/MyDrive/finn_results/walk_forward"``.
        model_name: Registry key (e.g. ``"gbm_pinn"``). Used to locate
            ``{base_dir}/{model_name}/wf_preds.json``.

    Returns:
        ``(test_preds, test_actual)`` — both 1-D ``np.ndarray`` of shape
        ``(T,)`` on log-return scale, or ``(None, None)`` if the file is
        absent.
    """
    preds_path = Path(base_dir) / model_name / "wf_preds.json"
    if not preds_path.exists():
        return None, None
    data = json.loads(preds_path.read_text())
    preds   = data.get("test_preds")
    actuals = data.get("test_actual")
    if preds is None or actuals is None:
        return None, None
    return np.array(preds, dtype=float), np.array(actuals, dtype=float)
