"""orchestration.py — per-model and walk-forward artefact orchestrators.

Extracted from Cell 8 and Cell 10 of notebooks/1_classical_baselines.ipynb.
All notebook globals are replaced with explicit keyword arguments so these
functions can be called from any context (Colab, local, CI).
"""

import json, pickle, shutil, traceback, time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .plotting import (
    CB, _save, _flatten_history,
    _plot_loss_curve, _plot_pred_vs_actual, _plot_residuals,
    _plot_rolling_error, _plot_directional_confusion, _plot_equity_curve,
    _plot_sp500_per_model, _plot_sp500_predictions, _plot_physics_breakdown,
    _plot_physics_ratio, _plot_lambda_schedule,
)


def _preds_from_checkpoint(model_name, fold_idx, base_dir, full_df, wf_cfg):
    """Reload pickled checkpoint and re-predict on fold test set.

    Fallback for fold JSONs written before the to_dict() fix (i.e. those
    that lack test_preds / test_actual keys). Classical model checkpoints
    are tiny pickles; re-prediction is O(n_test) ≈ instant.

    Args:
        model_name: Registry key (e.g. "garch").
        fold_idx: Zero-based fold index.
        base_dir: Root walk-forward artefact directory.
        full_df: Full feature DataFrame passed to run_walk_forward.
        wf_cfg: The same TrainingConfig used for the walk-forward run
                (supplies walk_forward.test_years and .val_months).
    Returns:
        (test_preds, test_actual) as np.ndarrays, or (None, None) on failure.
    """
    ckpt_path = (
        base_dir / model_name / f"fold_{fold_idx}" / "seed_42"
        / f"{model_name}.pt"
    )
    if not ckpt_path.exists():
        return None, None
    try:
        with open(ckpt_path, "rb") as fh:
            model = pickle.load(fh)
        from src.data.splitter import WalkForwardSplitter
        splitter = WalkForwardSplitter(
            full_df,
            test_years=wf_cfg.walk_forward.test_years,
            val_months=wf_cfg.walk_forward.val_months,
        )
        for idx, train_df, _, test_df in splitter:
            if idx == fold_idx:
                break
        train_ret  = train_df["log_return"].dropna().to_numpy(dtype=float)
        test_ret   = test_df["log_return"].dropna().to_numpy(dtype=float)
        last_train = float(train_ret[-1]) if len(train_ret) else 0.0
        preds = model.predict(test_ret, last_train)
        return preds, test_ret
    except Exception:
        return None, None


def emit_model_artefacts(
    model_name: str,
    result,
    *,
    per_model_dir: Path,
    drive_per_model_dir: Optional[Path] = None,
    patience: int = 20,
) -> Path:
    """Write the full per-model artefact set the moment training finishes.

    Runs immediately after run_experiment() returns. Every plot follows the
    matplotlib conventions declared in the plan: OO API, constrained_layout,
    300-dpi PNG + vector PDF, colorblind palette, despined axes, rasterized
    scatter/hexbin, and plt.close() after save to keep the Colab kernel alive
    across the 21-model loop.

    Residual diagnostics (Jarque–Bera + Ljung–Box on squared residuals at lag 20)
    are computed here and persisted under ``metrics.json -> residual_diagnostics``
    so that the summary cell can surface them without re-loading the raw tensors.

    Args:
        model_name: Registry key (e.g. "random_walk", "garch").
        result: ModelResult with fields best_val_loss, best_epoch, history
                (list[dict]), checkpoint_path (Path|None), test_preds/test_actual
                (np.ndarray|None), equity_curve (np.ndarray|None),
                test_metrics (dict), wall_clock_sec (float).
        per_model_dir: Root directory for per-model artefacts
                       (e.g. Path("results/per_model")).
        drive_per_model_dir: If not None and its parent exists, mirror the
                             artefact directory into
                             drive_per_model_dir / model_name.
        patience: Early-stopping patience window used to shade the loss curve
                  (default 20).

    Returns:
        Path to per_model_dir / model_name / (artefacts already flushed).
    """
    out_dir = per_model_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    residual_diagnostics: dict = {
        "jarque_bera_stat": None, "jarque_bera_p": None,
        "ljung_box_sq_stat": None, "ljung_box_sq_p": None,
        "ljung_box_lags": 20,
    }

    if getattr(result, "checkpoint_path", None):
        shutil.copy2(result.checkpoint_path, out_dir / "checkpoint.pt")

    hist_df = pd.DataFrame(result.history) if result.history else pd.DataFrame()
    if not hist_df.empty:
        hist_df.to_csv(out_dir / "history.csv", index=False)

    is_pinn = "pinn" in model_name

    if not hist_df.empty:
        _plot_loss_curve(hist_df, result.best_epoch, patience, model_name, out_dir)

    if getattr(result, "test_preds", None) is not None:
        pred = np.asarray(result.test_preds).ravel()
        actual = np.asarray(result.test_actual).ravel()
        resid = actual - pred
        r2 = float(result.test_metrics.get("r_squared", float("nan")))
        _plot_pred_vs_actual(pred, actual, r2, model_name, out_dir)
        _plot_residuals(resid, model_name, out_dir)
        _plot_rolling_error(pred, actual, model_name, out_dir)
        _plot_directional_confusion(pred, actual, model_name, out_dir)

        try:
            jb_stat, jb_p = stats.jarque_bera(resid)
            residual_diagnostics["jarque_bera_stat"] = float(jb_stat)
            residual_diagnostics["jarque_bera_p"] = float(jb_p)
        except Exception:
            pass
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lags = min(20, max(1, len(resid) // 5))
            lb = acorr_ljungbox(resid ** 2, lags=[lags], return_df=True)
            residual_diagnostics["ljung_box_sq_stat"] = float(lb["lb_stat"].iloc[0])
            residual_diagnostics["ljung_box_sq_p"] = float(lb["lb_pvalue"].iloc[0])
            residual_diagnostics["ljung_box_lags"] = int(lags)
        except Exception:
            pass

    if getattr(result, "equity_curve", None) is not None:
        _plot_equity_curve(result, model_name, out_dir)
        _plot_sp500_per_model(result, model_name, out_dir)

    if is_pinn and not hist_df.empty:
        _flatten_history(hist_df)
        _plot_physics_breakdown(hist_df, model_name, out_dir)
        _plot_physics_ratio(hist_df, model_name, out_dir)
        _plot_lambda_schedule(hist_df, model_name, out_dir)

    (out_dir / "metrics.json").write_text(json.dumps({
        "model": model_name,
        "best_val_loss": result.best_val_loss,
        "best_epoch": result.best_epoch,
        "wall_clock_sec": getattr(result, "wall_clock_sec", None),
        "test_metrics": result.test_metrics,
        "residual_diagnostics": residual_diagnostics,
    }, indent=2, default=str))

    if drive_per_model_dir is not None and drive_per_model_dir.parent.exists():
        shutil.copytree(out_dir, drive_per_model_dir / model_name, dirs_exist_ok=True)

    tm = result.test_metrics
    print(f"  ✓ {model_name}: RMSE={tm.get('rmse', float('nan')):.5f}  "
          f"DA={tm.get('directional_accuracy', float('nan')):.3f}  "
          f"Sharpe={tm.get('sharpe', float('nan')):+.3f}  "
          f"→ {out_dir}")
    return out_dir


def train_one(
    model_name: str,
    *,
    config,
    df,
    checkpoint_dir: Path,
    ticker: str,
    per_model_dir: Path,
    drive_per_model_dir: Optional[Path] = None,
    run_experiment_fn,
) -> object:
    """Train one model end-to-end, emit artefacts, return the result.

    Wraps run_experiment_fn in try/except so a single model failure does not
    abort the loop — traceback is printed and None is returned.

    Args:
        model_name: Registry key routed through run_experiment_fn.
        config: TrainingConfig instance (epochs, patience, seeds, …).
                ``config.seeds[0]`` is used as the single training seed.
        df: Full feature DataFrame (output of compute_features).
        checkpoint_dir: Root checkpoint directory; model checkpoints are
                        written to checkpoint_dir / model_name.
        ticker: Ticker symbol string (e.g. "^GSPC").
        per_model_dir: Root directory for per-model artefacts.
        drive_per_model_dir: Optional Drive mirror root (passed through to
                             emit_model_artefacts).
        run_experiment_fn: Callable with the same signature as
                           src.training.runner.run_experiment.

    Returns:
        ModelResult on success, or None on failure.
    """
    print(f"\n{'='*60}\nTraining: {model_name.upper()} on {ticker}\n{'='*60}")
    t0 = time.time()
    try:
        result = run_experiment_fn(
            config=config,
            model_name=model_name,
            dataframe=df,
            checkpoint_dir=checkpoint_dir / model_name,
            seed=config.seeds[0],
            ticker=ticker,
        )
        result.wall_clock_sec = time.time() - t0
        emit_model_artefacts(
            model_name,
            result,
            per_model_dir=per_model_dir,
            drive_per_model_dir=drive_per_model_dir,
            patience=getattr(config, "patience", 20) or 20,
        )
        return result
    except Exception:
        print(f"  ✗ {model_name} FAILED — continuing with next model:")
        traceback.print_exc()
        return None


def emit_wf_artefacts(
    model_name,
    wf_result,
    base_dir,
    *,
    per_model_dir: Path,
    drive_per_model_dir: Optional[Path] = None,
    raw=None,
    full_df=None,
    wf_cfg=None,
):
    """Emit all per-model artefacts after walk-forward completes.

    Saves to per_model_dir / model_name / (Group A + Group B).

    GROUP A — Walk-forward views
        wf_equity_curves : per-fold equity curves overlaid + grand-mean.
        wf_fold_metrics  : per-fold bar chart with μ annotation on dashed line.

    GROUP B — Full OOS single-view (2018-2023 concatenated)
        Reads test_preds / test_actual from fold JSONs. If missing (old run),
        falls back to reloading the checkpoint and re-predicting (requires
        full_df and wf_cfg to be supplied).
        pred_vs_actual, residuals, rolling_error, directional_confusion,
        equity_curve, sp500_vs_predicted (only if raw is not None).

    Args:
        model_name: Registry key (e.g. "garch").
        wf_result:  WalkForwardResult from run_walk_forward().
        base_dir:   Root walk-forward artefact directory (WF_BASE_DIR).
        per_model_dir: Root directory for per-model artefacts (required).
        drive_per_model_dir: Optional Drive mirror root; if not None and its
                             parent exists, the artefact dir is mirrored.
        raw: Raw OHLCV DataFrame from DataFetcher (needed for
             sp500_vs_predicted). Skipped if None.
        full_df: Full feature DataFrame (needed for checkpoint fallback).
        wf_cfg: TrainingConfig with walk_forward settings (same fallback).
    """
    out_dir = per_model_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load per-fold data ────────────────────────────────────────────────────
    fold_equities, fold_bh_curves = [], []
    all_preds,     all_actuals    = [], []

    for fold_idx in range(wf_result.n_folds):
        result_path = (
            base_dir / model_name / f"fold_{fold_idx}" / "seed_42"
            / f"{model_name}_result.json"
        )
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text())
        if data.get("equity_curve"):
            fold_equities.append(np.asarray(data["equity_curve"], dtype=float))
        if data.get("buy_hold_curve"):
            fold_bh_curves.append(np.asarray(data["buy_hold_curve"], dtype=float))

        preds = data.get("test_preds")
        acts  = data.get("test_actual")
        if preds and acts:
            all_preds.extend(preds)
            all_actuals.extend(acts)
        elif full_df is not None and wf_cfg is not None:
            fp, fa = _preds_from_checkpoint(model_name, fold_idx, base_dir,
                                             full_df, wf_cfg)
            if fp is not None:
                all_preds.extend(fp.tolist())
                all_actuals.extend(fa.tolist())

    test_years = wf_result.test_years[: len(wf_result.fold_metrics)]

    # ════════════════════════════════════════════════════════════════════════
    # GROUP A — Walk-forward views
    # ════════════════════════════════════════════════════════════════════════

    # Figure 1: overlaid fold equity curves (enlarged)
    if fold_equities:
        fig, ax = plt.subplots(figsize=(11.0, 5.5), constrained_layout=True)
        for i, (eq, yr) in enumerate(zip(fold_equities, test_years)):
            ax.plot(eq, color=CB[i % len(CB)], lw=1.2, alpha=0.80, label=str(yr))
        for i, bh in enumerate(fold_bh_curves):
            ax.plot(bh, color=CB[i % len(CB)], lw=0.7, alpha=0.22, ls="--")
        max_len = max(len(e) for e in fold_equities)
        padded  = np.array([np.pad(e, (0, max_len - len(e)), mode="edge")
                            for e in fold_equities])
        ax.plot(padded.mean(axis=0), color="black", lw=2.0, ls="--",
                label="Mean", zorder=5)
        ax.axhline(1.0, color="grey", lw=0.6, ls=":", alpha=0.6)
        ax.set(xlabel="Test step", ylabel="Equity (×)",
               title=f"{model_name} — Walk-Forward Equity Curves (per fold)")
        ax.legend(loc="upper left", ncol=4, fontsize=9)
        sns.despine(ax=ax)
        _save(fig, out_dir, "wf_equity_curves")

    # Figure 2: per-fold metric bars (enlarged, cleaner)
    _metrics = ["rmse", "directional_accuracy", "sharpe", "sortino"]
    _labels  = ["RMSE", "Directional Accuracy", "Sharpe", "Sortino"]
    n = len(test_years)

    fig, axes = plt.subplots(1, len(_metrics), figsize=(15.0, 5.5), constrained_layout=True)
    fig.suptitle(f"{model_name} — Per-Fold Metrics (2018–2023)",
                 fontsize=14, fontweight="bold")

    for ax, metric, label in zip(axes, _metrics, _labels):
        vals   = [fm.get(metric, float("nan")) for fm in wf_result.fold_metrics]
        colors = [CB[j % len(CB)] for j in range(n)]
        ax.bar(range(n), vals, color=colors, edgecolor="white", lw=0.6, width=0.65)
        mean_val = wf_result.mean_metrics.get(metric, float("nan"))
        ax.axhline(mean_val, color="black", lw=1.4, ls="--", alpha=0.85, zorder=3)
        ax.text(n - 0.52, mean_val, f"μ={mean_val:.3f}",
                va="bottom", ha="right", fontsize=9,
                color="black", fontweight="bold")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(y) for y in test_years], rotation=45, ha="right")
        ax.tick_params(axis="both", labelsize=9)
        sns.despine(ax=ax)

    _save(fig, out_dir, "wf_fold_metrics")

    # ════════════════════════════════════════════════════════════════════════
    # GROUP B — Full OOS single-view plots (2018–2023 concatenated)
    # ════════════════════════════════════════════════════════════════════════
    if all_preds and all_actuals:
        pred   = np.array(all_preds,   dtype=float)
        actual = np.array(all_actuals, dtype=float)
        resid  = actual - pred
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2_oos = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        _plot_pred_vs_actual(pred, actual, r2_oos, model_name, out_dir)
        _plot_residuals(resid, model_name, out_dir)
        _plot_rolling_error(pred, actual, model_name, out_dir)
        _plot_directional_confusion(pred, actual, model_name, out_dir)

        strategy_returns = np.sign(pred) * actual
        oos_result = SimpleNamespace(
            equity_curve   = np.cumprod(1.0 + strategy_returns),
            buy_hold_curve = np.cumprod(1.0 + actual),
            test_metrics   = wf_result.mean_metrics,
        )
        _plot_equity_curve(oos_result, model_name, out_dir)
    else:
        print(f"  ⚠ No test_preds found in fold JSONs and no fallback available"
              f" — Group B plots skipped for {model_name}.")

    # S&P 500 actual vs predicted price path — only if raw is supplied
    if raw is not None:
        _plot_sp500_predictions(model_name, wf_result, base_dir, out_dir, raw=raw)

    # ── Drive mirror ─────────────────────────────────────────────────────────
    if drive_per_model_dir is not None and drive_per_model_dir.parent.exists():
        shutil.copytree(out_dir, drive_per_model_dir / model_name, dirs_exist_ok=True)

    # ── Stdout mean ± std table ───────────────────────────────────────────────
    print(f"\n  Walk-Forward averages ({wf_result.n_folds} folds):")
    for key in ["rmse", "mae", "directional_accuracy", "sharpe",
                "sortino", "max_drawdown", "calmar", "r_squared"]:
        mean = wf_result.mean_metrics.get(key, float("nan"))
        std  = wf_result.std_metrics.get(key, float("nan"))
        print(f"    {key:<28} {mean:+.5f}  ±  {std:.5f}")
