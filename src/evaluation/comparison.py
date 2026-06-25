"""Side-by-side comparison of trained models.

Turns a ``{model_name -> TrainingResult}`` dict — the natural output
of running many ``run_experiment`` calls — into a ranked
:class:`pandas.DataFrame` and a saved 2×2 matplotlib summary figure.
Used by the Phase 8 dissertation notebooks to build the main results
table and figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.training.result import TrainingResult, WalkForwardResult

_METRIC_KEYS = (
    "rmse", "mae", "r_squared", "directional_accuracy",
    "sharpe", "sortino", "max_drawdown", "calmar",
)


def compare_models(
    results: dict[str, TrainingResult],
    sort_by: str = "sharpe",
    ascending: bool = False,
) -> pd.DataFrame:
    """Collapse a dict of :class:`TrainingResult`s into one ranked table.

    One row per model. Missing metrics are filled with ``NaN`` (they
    never trigger a ``KeyError``), so callers can mix PINN and
    baseline results produced by different code paths. Literature
    quality bands are attached as ``<metric>_band`` columns using
    :func:`src.evaluation.benchmarks.classify_metric`.

    Args:
        results: Mapping ``{model_name: TrainingResult}``. Each
            ``TrainingResult`` must have ``test_metrics`` populated —
            normally done automatically by ``runner.run_experiment``
            after the best checkpoint is loaded. Must be non-empty.
        sort_by: Column name to sort on. If the column is absent the
            frame is returned in insertion order. Default
            ``"sharpe"``.
        ascending: Sort direction. Default ``False`` so the best
            Sharpe appears first.

    Returns:
        :class:`pandas.DataFrame` with columns (in order):
        ``model``, ``rmse``, ``mae``, ``r_squared``,
        ``directional_accuracy``, ``sharpe``, ``sortino``,
        ``max_drawdown``, ``calmar``, ``best_val_loss``,
        ``best_epoch``, then the six ``<metric>_band`` label columns
        for ``sharpe``, ``sortino``, ``calmar``,
        ``directional_accuracy``, ``r_squared``, ``max_drawdown``.
        **Note:** ``rmse`` / ``mae`` bands are sigma-relative and
        therefore omitted — the caller should add them with the
        per-ticker ``σ_r`` when available.

    Raises:
        ValueError: If ``results`` is empty.
    """
    if not results:
        raise ValueError("compare_models requires at least one result")

    rows = []
    for name, r in results.items():
        row: dict[str, object] = {"model": name}
        for key in _METRIC_KEYS:
            row[key] = float(r.test_metrics.get(key, float("nan")))
        row["best_val_loss"] = r.best_val_loss
        row["best_epoch"] = r.best_epoch
        rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    from src.evaluation.benchmarks import classify_metric
    for key in ("sharpe", "sortino", "calmar", "directional_accuracy",
                "r_squared", "max_drawdown"):
        df[f"{key}_band"] = df[key].apply(lambda v, k=key: classify_metric(k, v))
    return df


def plot_comparison(
    results: dict[str, TrainingResult],
    save_path: str | Path,
) -> None:
    """Render a 2×2 model-comparison figure and save as PNG.

    Panels (row-major):
      1. ``(0, 0)`` — Sharpe ratio bar chart (higher is better).
      2. ``(0, 1)`` — RMSE bar chart (lower is better).
      3. ``(1, 0)`` — Directional accuracy bar chart, y-axis in
         ``[0, 1]`` with 0.5 as the coin-flip reference level.
      4. ``(1, 1)`` — Overlaid validation-loss curves, one line per
         model keyed to :attr:`TrainingResult.history`.

    The figure is written to disk at ``dpi=120`` and closed
    immediately to free memory — safe to call inside a loop.

    Args:
        results: Same mapping as :func:`compare_models` —
            ``{model_name: TrainingResult}`` with populated
            ``test_metrics`` and non-empty ``history``.
        save_path: Destination PNG path. Parent directories are
            created with ``mkdir(parents=True, exist_ok=True)`` if
            missing.

    Returns:
        ``None``. Side-effect: writes a PNG to ``save_path``.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = compare_models(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(df["model"], df["sharpe"])
    axes[0, 0].set_title("Sharpe Ratio (higher is better)")
    axes[0, 0].tick_params(axis="x", rotation=45)

    axes[0, 1].bar(df["model"], df["rmse"])
    axes[0, 1].set_title("RMSE (lower is better)")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].bar(df["model"], df["directional_accuracy"])
    axes[1, 0].set_title("Directional Accuracy")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].tick_params(axis="x", rotation=45)

    for name, r in results.items():
        epochs = [m.epoch for m in r.history]
        val = [m.val_loss for m in r.history]
        axes[1, 1].plot(epochs, val, label=name)
    axes[1, 1].set_title("Validation Loss Curves")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("val_loss")
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def compare_walk_forward(
    results: dict[str, WalkForwardResult],
    sort_by: str = "sharpe_mean",
    ascending: bool = False,
) -> pd.DataFrame:
    """Collapse walk-forward results into a ranked mean ± std table.

    One row per model. Each metric appears as two columns:
    ``{metric}_mean`` and ``{metric}_std``, matching the dissertation
    results table format.

    Args:
        results: Mapping ``{model_name: WalkForwardResult}`` produced by
            :func:`~src.training.walk_forward.run_walk_forward` or
            :func:`~src.training.walk_forward.aggregate_walk_forward`.
            Must be non-empty.
        sort_by: Column to sort on. Default ``"sharpe_mean"`` (descending).
            If the column is absent, the frame is returned in insertion order.
        ascending: Sort direction. Default ``False`` (best Sharpe first).

    Returns:
        :class:`pandas.DataFrame` with columns: ``model``, ``n_folds``,
        and for each metric in ``_METRIC_KEYS``: ``{metric}_mean`` and
        ``{metric}_std``. One row per model, sorted by ``sort_by``.

    Raises:
        ValueError: If ``results`` is empty.

    Examples:
        # In a notebook cell:
        from src.evaluation.comparison import compare_walk_forward

        df = compare_walk_forward(all_wf_results)
        display(df.round(4))
    """
    if not results:
        raise ValueError("compare_walk_forward requires at least one result")

    rows = []
    for name, r in results.items():
        row: dict[str, object] = {"model": name, "n_folds": r.n_folds}
        for key in _METRIC_KEYS:
            row[f"{key}_mean"] = float(r.mean_metrics.get(key, float("nan")))
            row[f"{key}_std"] = float(r.std_metrics.get(key, float("nan")))
        rows.append(row)

    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    return df
