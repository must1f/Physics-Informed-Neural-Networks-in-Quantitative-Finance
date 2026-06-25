"""Low-level plot builders for per-model artefact emission.

All public and private functions follow the matplotlib conventions from the
Per-Model Output Contract:

* Object-oriented API only — every figure via ``plt.subplots(..., constrained_layout=True)``.
* 300-dpi PNG + vector PDF via :func:`_save`; ``plt.close(fig)`` after every
  save to prevent Colab OOM across the 19-model loop.
* Seaborn ``"colorblind"`` palette (stored in :data:`CB`); despined axes; no
  legend frame.
* ``rasterized=True`` on scatter/hexbin elements.

Call :func:`setup_theme` once per process before using any plot function.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# Module-level palette list — populated by setup_theme()
CB: list = []


def setup_theme() -> list:
    """Apply publication-quality matplotlib/seaborn defaults and populate the module-level CB palette.

    Calls sns.set_theme with the exact rc params defined in the Per-Model Artefact
    Helper (Cell 4 of 1_classical_baselines.ipynb). Must be called once per process
    before any plot function is used.

    Returns:
        list: The seaborn "colorblind" palette (also stored in module-level CB).
              CB[0]=train, [1]=val, [2]=pred, [3]=strategy, [4]=buy-hold.
    """
    sns.set_theme(
        context="notebook", style="ticks", palette="colorblind",
        rc={
            "figure.dpi": 120, "savefig.dpi": 300,
            "savefig.bbox": "tight", "savefig.facecolor": "white",
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.titlesize": 13, "axes.labelsize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 10, "ytick.labelsize": 10,
            "legend.frameon": False, "legend.fontsize": 10,
            "font.family": "serif", "mathtext.fontset": "cm",
            "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
            "lines.linewidth": 1.4,
        },
    )
    palette = sns.color_palette("colorblind")  # CB[0] train, [1] val, [2] pred, [3] strategy, [4] buy-hold
    CB.clear()
    CB.extend(palette)
    return CB


def _save(fig, out_dir: Path, name: str) -> None:
    """Write a figure as 300-dpi PNG + vector PDF, show inline, then close.

    The ``plt.close(fig)`` call after ``plt.show()`` is the Colab OOM guard —
    without it, figures accumulate in memory across the 19-model training loop.

    Args:
        fig: Matplotlib ``Figure`` object to save.
        out_dir: Destination directory (must already exist).
        name: Filename stem — written as ``{name}.png`` and ``{name}.pdf``.

    Returns:
        None.
    """
    fig.savefig(out_dir / f"{name}.png")
    fig.savefig(out_dir / f"{name}.pdf")
    plt.show(); plt.close(fig)

def _annotate(ax, text, loc=(0.02, 0.98)):
    """Overlay a monospace 9 pt rounded textbox on an axis.

    Used to surface key statistics (R², Sharpe, JB p-value) directly on
    the relevant subplot without cluttering the legend.

    Args:
        ax: Matplotlib ``Axes`` instance to annotate.
        text: Multi-line string displayed verbatim in a white rounded box.
        loc: ``(x, y)`` position in axes coordinates (default top-left).

    Returns:
        None.
    """
    ax.text(*loc, text, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.90, lw=0.4))

# --- Plot builders --------------------------------------------------------
def _plot_loss_curve(hist_df, best_epoch, patience, model_name, out_dir):
    """Two-panel train/val loss curve — linear scale on top, log-y underneath.

    Marks the best epoch with a crimson vertical rule and fills the early-stopping
    patience window so the reader can see why training stopped when it did.

    Args:
        hist_df: ``pd.DataFrame`` with columns ``train_loss`` and ``val_loss``,
            one row per epoch.
        best_epoch: Integer epoch number of the best checkpoint (1-indexed).
            If falsy, the vertical rule and patience shading are omitted.
        patience: Early-stopping patience in epochs — used to shade the window
            from ``best_epoch`` to ``best_epoch + patience``.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``loss_curve.{png,pdf}``.

    Returns:
        None.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0),
                             sharex=True, constrained_layout=True)
    epochs = hist_df.index.to_numpy() + 1
    for ax, yscale in zip(axes, ("linear", "log")):
        ax.plot(epochs, hist_df.get("train_loss"), color=CB[0], ls="-", lw=1.5, label="Train")
        ax.plot(epochs, hist_df.get("val_loss"), color=CB[1], ls="--", lw=1.5, label="Val")
        if best_epoch:
            ax.axvline(best_epoch, color="crimson", ls=":", lw=1.2, alpha=0.8,
                       label=f"Best epoch {best_epoch}")
            ax.axvspan(best_epoch, best_epoch + patience, color="crimson",
                       alpha=0.08, label="Early-stop patience")
        ax.set(yscale=yscale, ylabel=f"Loss ({yscale})")
        sns.despine(ax=ax)
    axes[-1].set_xlabel("Epoch")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{model_name} — Loss Curve")
    _save(fig, out_dir, "loss_curve")

def _plot_pred_vs_actual(pred, actual, r2, model_name, out_dir):
    """Left: hexbin density scatter + 45-degree reference line + R² textbox.
    Right: actual (solid black) vs predicted (dashed, coloured) time-series.

    The figure is intentionally wide so the time-series panel (width_ratio 2.5)
    has enough horizontal resolution to expose regime-level prediction failures.

    Args:
        pred: 1-D ``np.ndarray`` of model predictions on the log-return scale,
            shape ``(T,)``.
        actual: 1-D ``np.ndarray`` of ground-truth log-returns, shape ``(T,)``,
            aligned with ``pred``.
        r2: Coefficient of determination (float, dimensionless) annotated on
            the hexbin panel.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``pred_vs_actual.{png,pdf}``.

    Returns:
        None.
    """
    fig = plt.figure(figsize=(15.0, 6.0), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 2.5])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    lim = [float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))]
    hb = ax0.hexbin(actual, pred, gridsize=45, cmap="viridis",
                    mincnt=1, rasterized=True, linewidths=0.2)
    ax0.plot(lim, lim, color="black", ls="--", lw=1.0, alpha=0.7)
    ax0.set(xlim=lim, ylim=lim, xlabel="Actual", ylabel="Predicted")
    _annotate(ax0, f"R² = {r2:.3f}\nn = {len(actual):,}")
    fig.colorbar(hb, ax=ax0, label="count", shrink=0.85)

    steps = np.arange(len(actual))
    ax1.plot(steps, actual, color="black", lw=0.8, label="Actual", alpha=0.85)
    ax1.plot(steps, pred, color=CB[2], lw=1.2, ls="--", alpha=0.9, label="Predicted")
    ax1.set(xlabel="Test step", ylabel="Value")
    ax1.legend(loc="upper right")

    for ax in (ax0, ax1):
        sns.despine(ax=ax)
    fig.suptitle(f"{model_name} — Predictions (test set)", fontsize=14, fontweight="bold")
    _save(fig, out_dir, "pred_vs_actual")

def _plot_residuals(resid, model_name, out_dir):
    """Three-panel residual diagnostic: residuals over time, histogram + Normal fit, QQ plot.

    The Jarque–Bera p-value is annotated on the histogram so the reader can
    assess residual normality at a glance.

    Args:
        resid: 1-D ``np.ndarray`` of residuals ``actual - pred`` on the
            log-return scale, shape ``(T,)``.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``residuals.{png,pdf}``.

    Returns:
        None.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0), constrained_layout=True)
    sigma = resid.std()
    axes[0].plot(resid, color=CB[3], lw=0.7)
    axes[0].axhspan(-2*sigma, 2*sigma, color=CB[3], alpha=0.12, label="±2σ")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set(xlabel="Test step", ylabel="Residual", title="Residuals over time")
    axes[0].legend(loc="upper right")

    _, bins, _ = axes[1].hist(resid, bins=55, color=CB[0],
                               edgecolor="black", lw=0.3, density=True, alpha=0.8)
    grid = np.linspace(bins[0], bins[-1], 200)
    axes[1].plot(grid, stats.norm.pdf(grid, resid.mean(), sigma),
                 color="crimson", lw=1.5, label="Normal fit")
    _, jb_p = stats.jarque_bera(resid)
    _annotate(axes[1], f"μ = {resid.mean():.2e}\nσ = {sigma:.2e}\nJB p = {jb_p:.2e}")
    axes[1].set(xlabel="Residual", ylabel="Density", title="Histogram + Normal")
    axes[1].legend(loc="upper right")

    stats.probplot(resid, dist="norm", plot=axes[2])
    axes[2].set_title("QQ plot")
    for ax in axes: sns.despine(ax=ax)
    fig.suptitle(f"{model_name} — Residual Diagnostics", fontsize=14, fontweight="bold")
    _save(fig, out_dir, "residuals")

def _plot_rolling_error(pred, actual, model_name, out_dir, window=60):
    """Rolling RMSE and rolling directional accuracy across the test window.

    Exposes regime-dependent failure: a flat model can have a decent aggregate
    RMSE but collapse to coin-flip DA during a high-volatility period.

    Args:
        pred: 1-D ``np.ndarray`` of model predictions on the log-return scale,
            shape ``(T,)``.
        actual: 1-D ``np.ndarray`` of ground-truth log-returns, shape ``(T,)``.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``rolling_error.{png,pdf}``.
        window: Rolling window length in periods (default 60 trading days).

    Returns:
        None.
    """
    rolling_rmse = pd.Series((actual - pred) ** 2).rolling(window).mean().pow(0.5)
    rolling_da = pd.Series((np.sign(pred) == np.sign(actual)).astype(float)).rolling(window).mean()
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.0),
                             sharex=True, constrained_layout=True)
    axes[0].plot(rolling_rmse, color=CB[0], lw=1.4)
    axes[0].set(ylabel=f"Rolling RMSE ({window})",
                title=f"{model_name} — Rolling test error")
    axes[1].plot(rolling_da, color=CB[1], lw=1.4)
    axes[1].axhline(0.5, color="crimson", ls="--", lw=1.0, alpha=0.7, label="Chance")
    axes[1].set(xlabel="Test step", ylabel=f"Rolling DA ({window})")
    axes[1].legend(loc="lower right")
    for ax in axes: sns.despine(ax=ax)
    _save(fig, out_dir, "rolling_error")

def _plot_directional_confusion(pred, actual, model_name, out_dir):
    """Row-normalised confusion heatmap of sign(pred) vs sign(actual).

    Rows are actual direction (+ / −); columns are predicted direction (+, 0, −).
    The ``pred 0 (flat)`` column exists so that zero-forecast models
    (e.g. ``random_walk`` where ``sign(0)=0``) are not mislabelled as
    "predicts negative". For models that never emit ``sign(pred)=0`` the
    middle column stays empty.

    Diagonal cells of the +/− block show the hit rate; off-diagonals show the
    two failure modes (false up / false down).

    Args:
        pred: 1-D ``np.ndarray`` of model predictions (any scale — only the
            sign is used), shape ``(T,)``.
        actual: 1-D ``np.ndarray`` of ground-truth log-returns, shape ``(T,)``.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``directional_confusion.{png,pdf}``.

    Returns:
        None.
    """
    sp, sa = np.sign(pred), np.sign(actual)
    cm = np.array([
        [((sa == +1) & (sp == +1)).sum(),
         ((sa == +1) & (sp ==  0)).sum(),
         ((sa == +1) & (sp == -1)).sum()],
        [((sa != +1) & (sp == +1)).sum(),
         ((sa != +1) & (sp ==  0)).sum(),
         ((sa != +1) & (sp == -1)).sum()],
    ], dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i,j])}\n({norm[i,j]:.2f})",
                    ha="center", va="center", fontsize=10,
                    color="white" if norm[i,j] < 0.5 else "black")
    ax.set(xticks=[0, 1, 2], yticks=[0, 1],
           xticklabels=["pred +", "pred 0 (flat)", "pred −"],
           yticklabels=["actual +", "actual −"],
           title=f"{model_name} — Directional Confusion")
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalised")
    _save(fig, out_dir, "directional_confusion")

def _plot_sp500_per_model(result, model_name, out_dir) -> None:
    """Reconstructed S&P 500 price path vs model predictions, anchored at $10,000.

    What it does: converts ``test_actual`` and ``test_preds`` (both on the
    log-return scale) into cumulative price paths anchored at a nominal
    $10,000 starting value, then plots a two-panel figure — actual vs
    predicted price with over/under-prediction fills (top), and sign-based
    strategy equity vs buy-and-hold equity at $10k (bottom).  Saved as
    ``sp500_predictions.{png,pdf}``.

    Inputs:
        result: object with fields:

            * ``test_preds``     — ``(T,)`` ``np.ndarray``, model's log-return
              predictions on the held-out test set (raw scale, not normalised).
            * ``test_actual``    — ``(T,)`` ``np.ndarray``, ground-truth
              log-returns aligned with ``test_preds``.
            * ``equity_curve``   — ``(T,)`` ``np.ndarray``, cumulative product
              of ``(1 + sign(pred) * actual)`` from ``evaluate_on_test``; the
              first element is ``1 + strategy_returns[0]``, **not** ``1.0``.
            * ``buy_hold_curve`` — ``(T,)`` ``np.ndarray``, cumulative product
              of ``(1 + actual)``.
            * ``test_metrics``   — ``dict[str, float]`` with at minimum
              ``sharpe``, ``sortino``, ``max_drawdown``, ``calmar``,
              ``directional_accuracy``, ``rmse``, ``information_ratio``.

        model_name: registry key used in the figure title and filename.
        out_dir: ``Path`` — destination directory for PNG and PDF outputs.

    Returns:
        None.  Returns early without raising if ``test_preds`` / ``test_actual``
        are absent, ``None``, or empty.

    Price reconstruction:
        A synthetic index ``P_t = 10000 × exp(Σ_{i=0}^{t-1} r_i)`` is built
        by prepending ``0`` to the cumulative sum of log-returns so the series
        starts exactly at $10,000 regardless of the first return's sign.
        The strategy equity series is formed analogously from ``equity_curve``:
        ``[10000, 10000 × eq[0], 10000 × eq[1], …]``.
    """
    _p  = getattr(result, "test_preds",      None)
    _a  = getattr(result, "test_actual",     None)
    _eq = getattr(result, "equity_curve",    None)
    _bh = getattr(result, "buy_hold_curve",  None)
    pred_arr   = np.asarray(_p  if _p  is not None else []).ravel()
    actual_arr = np.asarray(_a  if _a  is not None else []).ravel()
    eq         = np.asarray(_eq if _eq is not None else []).ravel()
    bh         = np.asarray(_bh if _bh is not None else []).ravel()

    if len(pred_arr) == 0 or len(actual_arr) == 0:
        return

    INITIAL = 10_000.0

    # ── Price path reconstruction ─────────────────────────────────────────────
    # Prepend 0 so the series starts at INITIAL before any return is applied.
    actual_price = INITIAL * np.exp(np.r_[0.0, np.cumsum(actual_arr)])
    pred_price   = INITIAL * np.exp(np.r_[0.0, np.cumsum(pred_arr)])
    steps_price  = np.arange(len(actual_price))

    # ── Equity in dollars ─────────────────────────────────────────────────────
    strat_dollar = np.r_[INITIAL, INITIAL * eq] if len(eq) > 0 else None
    bh_dollar    = np.r_[INITIAL, INITIAL * bh] if len(bh) > 0 else None
    steps_eq     = np.arange(len(strat_dollar)) if strat_dollar is not None else None

    strat_pnl = float(strat_dollar[-1] - INITIAL) if strat_dollar is not None else float("nan")
    bh_pnl    = float(bh_dollar[-1]    - INITIAL) if bh_dollar    is not None else float("nan")

    tm = result.test_metrics
    n_rows = 2 if strat_dollar is not None else 1
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(13.0, 7.2 if n_rows == 2 else 4.8),
        sharex=False, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1.6]} if n_rows == 2 else None,
    )
    ax_price = axes[0] if n_rows == 2 else axes

    # ── Top panel: actual vs predicted price path ─────────────────────────────
    ax_price.plot(steps_price, actual_price, color="black", lw=1.6, alpha=0.88,
                  label="S&P 500 (actual, reconstructed)", zorder=5)
    ax_price.plot(steps_price, pred_price, color=CB[2], lw=1.2, alpha=0.90,
                  label="Model predicted", zorder=4)
    ax_price.fill_between(steps_price, actual_price, pred_price,
                          where=(pred_price >= actual_price),
                          alpha=0.16, color="seagreen", linewidth=0,
                          label="Over-predicts")
    ax_price.fill_between(steps_price, actual_price, pred_price,
                          where=(pred_price <  actual_price),
                          alpha=0.16, color="crimson", linewidth=0,
                          label="Under-predicts")

    stats_text = (
        f"Sharpe  {tm.get('sharpe', float('nan')):+.2f}\n"
        f"DA      {tm.get('directional_accuracy', float('nan')):.3f}\n"
        f"RMSE    {tm.get('rmse', float('nan')):.5f}\n"
        f"IR      {tm.get('information_ratio', float('nan')):+.2f}"
    )
    ax_price.text(0.02, 0.97, stats_text, transform=ax_price.transAxes,
                  va="top", ha="left", family="monospace", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.90, lw=0.4))

    pnl_text = f"Strategy P&L ${strat_pnl:+,.0f}  vs  B&H ${bh_pnl:+,.0f}  (on $10k)"
    ax_price.text(0.98, 0.97, pnl_text, transform=ax_price.transAxes,
                  va="top", ha="right", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.90, lw=0.4))

    ax_price.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_price.set(xlabel="Test step" if n_rows == 1 else "",
                 ylabel="Price (USD, $10k base)",
                 title=f"{model_name.upper()} — Reconstructed S&P 500: Actual vs Predicted")
    ax_price.legend(loc="lower left", fontsize=8, ncol=2)
    sns.despine(ax=ax_price)

    # ── Bottom panel: strategy equity vs B&H in dollars ──────────────────────
    if n_rows == 2:
        ax_eq = axes[1]
        ax_eq.plot(steps_eq, strat_dollar, color=CB[3], lw=1.4, label="Strategy")
        if bh_dollar is not None:
            ax_eq.plot(steps_eq, bh_dollar, color=CB[4], lw=1.2, ls="--", alpha=0.85,
                       label="Buy & Hold")
        ax_eq.axhline(INITIAL, color="black", lw=0.8, ls=":", alpha=0.6, label="Cash ($10k)")
        ax_eq.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_eq.set(xlabel="Test step", ylabel="Equity (USD)")
        ax_eq.legend(loc="upper left", fontsize=8)
        sns.despine(ax=ax_eq)

    _save(fig, out_dir, "sp500_predictions")


def _plot_equity_curve(result, model_name, out_dir):
    """Two-panel equity curve: strategy vs buy-and-hold (top), underwater drawdown (bottom).

    The stats box (Sharpe / Sortino / MDD / Calmar) sits in the upper-right
    and the legend in the upper-left to avoid collision.

    Args:
        result: Object with attributes:

            * ``equity_curve``   — ``(T,)`` ``np.ndarray``, cumulative product
              of ``(1 + sign(pred) * actual)`` starting from
              ``1 + strategy_returns[0]``.
            * ``buy_hold_curve`` — ``(T,)`` ``np.ndarray`` or ``None``.
            * ``test_metrics``   — ``dict[str, float]`` with keys
              ``sharpe``, ``sortino``, ``max_drawdown``, ``calmar``.

        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``equity_curve.{png,pdf}``.

    Returns:
        None.
    """
    eq = np.asarray(result.equity_curve)
    bh = np.asarray(result.buy_hold_curve) if getattr(result, "buy_hold_curve", None) is not None else None
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0),
                             sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(eq, color=CB[3], lw=1.4, label="Strategy")
    if bh is not None:
        axes[0].plot(bh, color=CB[4], lw=1.2, ls="--", alpha=0.85, label="Buy & Hold")
    axes[0].axhline(eq[0], color="black", lw=0.8, ls=":", alpha=0.6, label="Cash")
    axes[0].set(ylabel="Equity (×)", title=f"{model_name} — Sign-Based Strategy")
    tm = result.test_metrics
    stats_text = (
        f"Sharpe  {tm.get('sharpe', float('nan')):+.2f}\n"
        f"Sortino {tm.get('sortino', float('nan')):+.2f}\n"
        f"MDD     {tm.get('max_drawdown', float('nan')):+.2%}\n"
        f"Calmar  {tm.get('calmar', float('nan')):+.2f}"
    )
    axes[0].text(0.98, 0.98, stats_text, transform=axes[0].transAxes,
                 va="top", ha="right", family="monospace", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.90, lw=0.4))
    axes[0].legend(loc="upper left")
    axes[1].fill_between(np.arange(len(dd)), dd, 0, color="crimson", alpha=0.45)
    axes[1].set(xlabel="Test step", ylabel="Drawdown")
    for ax in axes: sns.despine(ax=ax)
    _save(fig, out_dir, "equity_curve")

def _plot_sp500_predictions(model_name, wf_result, base_dir, out_dir, *, raw):
    """S&P 500 actual closing price vs model's predicted price path (OOS, all folds).

    Loads per-fold predictions from disk, reconstructs the price series as
    P_t = P_start * exp(cumsum(log_returns)) anchored to the real S&P 500
    close one trading day before each fold's test window begins.
    Green fill = model over-predicts price; crimson fill = under-predicts.

    Classical baselines use a single seed (seed_42) and fold_{idx} directories
    (0-indexed), matching the walk-forward splitter convention in this notebook.

    Args:
        model_name: Registry key (e.g. "garch") — used to locate fold JSONs.
        wf_result:  WalkForwardResult for this model (provides test_years, n_folds).
        base_dir:   Root walk-forward artefact directory (WF_BASE_DIR).
        out_dir:    Destination Path for PNG + PDF.
        raw:        OHLCV DataFrame from DataFetcher (keyword-only). Must have a
                    Close column (or MultiIndex with Close at level 0).
    """
    # Resolve S&P 500 close prices from the provided `raw` DataFrame
    _raw = raw
    try:
        if isinstance(_raw.columns, pd.MultiIndex):
            close = _raw["Close"].iloc[:, 0].squeeze().sort_index()
        elif "Close" in _raw.columns:
            close = _raw["Close"].squeeze().sort_index()
        else:
            close = _raw.iloc[:, 0].sort_index()
    except Exception as e:
        print(f"  ⚠  Could not resolve Close prices ({e}) — sp500_vs_predicted skipped")
        return

    # Load per-fold predictions and reconstruct price series
    # Classical baselines: fold_{idx}/seed_42/{model}_result.json
    folds = []
    for fold_idx in range(wf_result.n_folds):
        result_path = (
            base_dir / model_name / f"fold_{fold_idx}" / "seed_42"
            / f"{model_name}_result.json"
        )
        if not result_path.exists():
            continue
        data = json.loads(result_path.read_text())
        p = data.get("test_preds")
        a = data.get("test_actual")
        if not p or not a:
            continue

        fold_pred = np.asarray(p, dtype=float)
        fold_act  = np.asarray(a, dtype=float)
        n_preds   = len(fold_pred)

        # Map fold_idx → test year for date alignment
        fold_year = wf_result.test_years[fold_idx] if fold_idx < len(wf_result.test_years) else None
        if fold_year is None:
            continue

        # Align to the last n_preds trading days of the test year
        year_dates = close.index[close.index.year == fold_year]
        if len(year_dates) < n_preds:
            year_dates = close.index[close.index.year >= fold_year][:n_preds]
        test_dates = year_dates[-n_preds:]

        # Anchor to actual close one day before the test window
        anchor_idx = close.index.get_loc(test_dates[0])
        P_start    = float(close.iloc[max(0, anchor_idx - 1)])

        folds.append((
            test_dates,
            P_start * np.exp(np.cumsum(fold_act)),   # actual price reconstruction
            P_start * np.exp(np.cumsum(fold_pred)),  # predicted price path
            fold_pred,                                # raw log-return predictions
            fold_act,                                 # raw log-return actuals
        ))

    if not folds:
        print(f"  ⚠  No fold predictions found — sp500_vs_predicted skipped for {model_name}")
        return

    dates  = np.concatenate([f[0].values for f in folds])
    actual = np.concatenate([f[1]        for f in folds])
    pred   = np.concatenate([f[2]        for f in folds])
    fold_boundaries = [f[0][0] for f in folds[1:]]   # interior boundaries only

    # Strategy P&L on $10,000 initial capital (sign-based: long if pred>0, short if pred<0)
    INITIAL_CAPITAL = 10_000.0
    all_ret_pred = np.concatenate([f[3] for f in folds])
    all_ret_act  = np.concatenate([f[4] for f in folds])
    strat_equity = float(np.prod(1.0 + np.sign(all_ret_pred) * all_ret_act))
    bh_equity    = float(np.prod(1.0 + all_ret_act))
    strat_pnl    = INITIAL_CAPITAL * (strat_equity - 1.0)
    bh_pnl       = INITIAL_CAPITAL * (bh_equity    - 1.0)

    fig, ax = plt.subplots(figsize=(13.0, 4.8), constrained_layout=True)

    ax.plot(dates, actual, color="black", lw=1.6, alpha=0.88,
            label="S&P 500 (actual)", zorder=5)
    ax.plot(dates, pred, color=CB[2], lw=1.2, alpha=0.90,
            label="Predicted", zorder=4)
    ax.fill_between(dates, actual, pred,
                    where=(pred >= actual), alpha=0.16,
                    color="seagreen", linewidth=0, label="Over-predicts")
    ax.fill_between(dates, actual, pred,
                    where=(pred <  actual), alpha=0.16,
                    color="crimson",  linewidth=0, label="Under-predicts")

    for fb in fold_boundaries:
        ax.axvline(fb, color="slategray", lw=0.8, ls=":", alpha=0.65)
    for f in folds:
        yr = pd.Timestamp(f[0][0]).year
        ax.text(f[0][0], actual.max(), f" {yr}",
                fontsize=7.5, color="slategray", va="top")

    m, s = wf_result.mean_metrics, wf_result.std_metrics
    _annotate(ax,
        f"Sharpe {m.get('sharpe', float('nan')):+.3f}±{s.get('sharpe', 0):.3f}   "
        f"DA {m.get('directional_accuracy', float('nan')):.3f}   "
        f"RMSE {m.get('rmse', float('nan')):.4f}\n"
        f"Strategy P&L ${strat_pnl:+,.0f}  vs  B&H ${bh_pnl:+,.0f}  (on $10k)",
        loc=(0.01, 0.97))

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set(xlabel="Date", ylabel="S&P 500 (USD)",
           title=f"{model_name.upper()} — S&P 500 Actual vs Predicted Price (OOS)")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    sns.despine(ax=ax)
    _save(fig, out_dir, "sp500_vs_predicted")

def _flatten_history(hist_df: pd.DataFrame) -> None:
    """Expand ``EpochMetrics`` breakdown dicts into flat columns and add physics diagnostics.

    What it does: reads the ``train_breakdown`` and ``val_breakdown`` object
    columns (each cell is a ``dict[str, float]`` from ``EpochMetrics``),
    expands every key into a dedicated column, then computes ``physics_ratio``
    and ``lambda_scale`` so the physics-ratio and physics-breakdown plots have
    the columns they expect.  Operates on the DataFrame produced by
    ``pd.DataFrame(list[EpochMetrics])``; no copy is made.

    Inputs:
        hist_df: ``pd.DataFrame`` built from ``list[EpochMetrics]``, shape
            ``(n_epochs, n_fields)``.  Required columns:

            * ``train_breakdown`` — object dtype; each cell a ``dict[str, float]``
              with keys ``"data"`` (unweighted MSE), ``"total"`` (composite loss),
              and one entry per active physics constraint (e.g. ``"gbm"``,
              ``"ou"``).
            * ``val_breakdown``   — same structure as ``train_breakdown``.
            * ``train_loss``      — ``float64`` total composite training loss.
            * ``physics_scale``   — ``float64`` in ``[0, 1]``, the warmup
              multiplier from ``PhysicsScheduler``; ``1.0`` when no warmup.

            Classical / baseline runs may have empty dicts in the breakdown
            columns — that case is handled gracefully (``physics_ratio = 0``).

    Returns:
        None.  The following columns are added to ``hist_df`` in-place:

        * ``train_<key>`` — ``float64`` per-epoch training loss for each key
          in ``train_breakdown`` (e.g. ``train_data``, ``train_gbm``,
          ``train_total``).
        * ``val_<key>``   — same, from ``val_breakdown``.
        * ``{constraint}_loss`` — alias of ``train_{constraint}`` for every
          physics key (not ``"data"``, ``"total"``, or keys ending in
          ``"_unweighted"``).  The ``*_loss`` suffix is the pattern
          ``_plot_physics_breakdown`` scans for.  Values are **λ-weighted**
          contributions so ``physics_ratio`` correctly reflects the actual
          share of total loss, even when λ << 1.
        * ``physics_ratio`` — ``float64`` in ``[0, 1]``.  Per-epoch fraction
          of ``train_loss`` attributable to physics constraints:
          ``sum(λ-weighted {constraint}_loss columns) / (train_loss + 1e-12)``.
          Zero when no physics constraints are active.
        * ``lambda_scale`` — alias of ``physics_scale``; consumed by
          ``_plot_physics_ratio`` to render the λ-warmup overlay.
    """
    # ── Expand train_breakdown ────────────────────────────────────────────────
    if "train_breakdown" in hist_df.columns:
        bd_series = hist_df["train_breakdown"]
        all_keys: set[str] = set()
        for bd in bd_series:
            if isinstance(bd, dict):
                all_keys.update(bd.keys())
        for k in all_keys:
            hist_df[f"train_{k}"] = bd_series.apply(
                lambda d, _k=k: float(d.get(_k, 0.0)) if isinstance(d, dict) else 0.0
            )
            if k not in ("data", "total") and not k.endswith("_unweighted"):
                # _plot_physics_breakdown scans for columns ending in '_loss'
                # Exclude raw-residual diagnostic keys so physics_ratio only
                # sums the λ-weighted contributions.
                hist_df[f"{k}_loss"] = hist_df[f"train_{k}"]

    # ── Expand val_breakdown ──────────────────────────────────────────────────
    if "val_breakdown" in hist_df.columns:
        vd_series = hist_df["val_breakdown"]
        all_keys = set()
        for bd in vd_series:
            if isinstance(bd, dict):
                all_keys.update(bd.keys())
        for k in all_keys:
            hist_df[f"val_{k}"] = vd_series.apply(
                lambda d, _k=k: float(d.get(_k, 0.0)) if isinstance(d, dict) else 0.0
            )

    # ── physics_ratio ─────────────────────────────────────────────────────────
    if "train_loss" in hist_df.columns:
        phys_loss_cols = [
            c for c in hist_df.columns
            if c.endswith("_loss") and c not in {"train_loss", "val_loss", "total_loss"}
        ]
        if phys_loss_cols:
            phys_sum = hist_df[phys_loss_cols].sum(axis=1)
            hist_df["physics_ratio"] = phys_sum / (hist_df["train_loss"] + 1e-12)
        else:
            hist_df["physics_ratio"] = 0.0

    # ── lambda_scale alias ────────────────────────────────────────────────────
    if "physics_scale" in hist_df.columns:
        hist_df["lambda_scale"] = hist_df["physics_scale"]


def _plot_physics_breakdown(hist_df, model_name, out_dir):
    """Per-constraint physics-loss lines on log-y scale (PINN models only).

    No-op for classical / baseline models whose ``hist_df`` has no columns
    ending in ``_loss`` or starting with ``physics_``.

    Args:
        hist_df: ``pd.DataFrame`` produced by :func:`_flatten_history`, with
            one ``{constraint}_loss`` column per active physics constraint.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``physics_breakdown.{png,pdf}``.

    Returns:
        None.
    """
    phys_cols = [c for c in hist_df.columns
                 if (c.endswith("_loss") and c not in {"train_loss", "val_loss", "total_loss"})
                 or c.startswith("physics_")]
    if not phys_cols: return
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    for i, c in enumerate(phys_cols):
        ax.plot(hist_df.index + 1, hist_df[c], color=CB[i % len(CB)], lw=1.2, label=c)
    ax.set(yscale="log", xlabel="Epoch", ylabel="Loss (log)",
           title=f"{model_name} — Physics Loss Breakdown")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    sns.despine(ax=ax)
    _save(fig, out_dir, "physics_breakdown")

def _plot_physics_ratio(hist_df, model_name, out_dir):
    """Physics contribution ratio with optional λ warmup overlay (PINN models only).

    Primary y-axis shows ``physics_ratio`` (fraction of total training loss
    from physics constraints, in [0, 1]). If ``lambda_scale`` is present a
    secondary y-axis overlays the warmup multiplier.

    Args:
        hist_df: ``pd.DataFrame`` from :func:`_flatten_history` with a
            ``physics_ratio`` column and optionally ``lambda_scale``.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``physics_ratio.{png,pdf}``.

    Returns:
        None.
    """
    if "physics_ratio" not in hist_df.columns: return
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    epochs = hist_df.index + 1
    ax.fill_between(epochs, 0, hist_df["physics_ratio"], color=CB[2], alpha=0.35,
                    label="λ·physics / total loss")
    ax.plot(epochs, hist_df["physics_ratio"], color=CB[2], lw=1.2)
    ax.set(xlabel="Epoch", ylabel="Ratio", ylim=(0, 1),
           title=f"{model_name} — Physics Contribution Ratio")
    if "lambda_scale" in hist_df.columns:
        ax2 = ax.twinx()
        ax2.plot(epochs, hist_df["lambda_scale"], color=CB[3], ls="--", lw=1.2, label="λ warmup")
        ax2.set_ylabel("λ scale"); ax2.spines["top"].set_visible(False)
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    else:
        ax.legend(loc="upper left")
    sns.despine(ax=ax, right=False)
    _save(fig, out_dir, "physics_ratio")

def _plot_lambda_schedule(hist_df, model_name, out_dir):
    """Per-constraint λ(t) step plot showing how each physics weight evolves (PINN models only).

    No-op when no ``lambda_*`` columns (other than ``lambda_scale``) are present.

    Args:
        hist_df: ``pd.DataFrame`` from :func:`_flatten_history` with columns
            of the form ``lambda_{constraint}`` for each physics term.
        model_name: Registry key used in the figure title.
        out_dir: Destination ``Path`` for ``lambda_schedule.{png,pdf}``.

    Returns:
        None.
    """
    lam_cols = [c for c in hist_df.columns
                if c.startswith("lambda_") and c != "lambda_scale"]
    if not lam_cols: return
    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    for i, c in enumerate(lam_cols):
        ax.step(hist_df.index + 1, hist_df[c], color=CB[i % len(CB)],
                where="post", lw=1.2, label=c.replace("lambda_", "λ "))
    ax.set(xlabel="Epoch", ylabel="λ (weight)",
           title=f"{model_name} — Per-Constraint λ Schedule")
    ax.legend(fontsize=9); sns.despine(ax=ax)
    _save(fig, out_dir, "lambda_schedule")
