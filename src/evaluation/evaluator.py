"""Run a trained model over a test dataset and compute all metrics.

Handles both baseline models (forward(x)) and PINN variants
(forward(x, metadata)) by inspecting :class:`BasePINN` membership.
Runs under ``torch.no_grad()`` and moves tensors to the supplied
device.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import collate_fn
from src.evaluation.metrics import compute_all_metrics
from src.models.base_pinn import BasePINN


def evaluate_on_test(
    model: torch.nn.Module,
    test_ds,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infer over a held-out test set and return metrics plus raw arrays.

    Bridges the training layer and the evaluation layer: iterates the
    test ``DataLoader`` under ``torch.no_grad()``, concatenates the
    per-batch predictions and targets, then delegates to
    :func:`compute_all_metrics`. Handles both baseline models
    (``forward(x)``) and PINN variants (``forward(x, metadata)``) by
    checking :class:`BasePINN` membership; PINN outputs may be a
    tuple ``(prediction, residuals_dict)`` — only the prediction is
    used.

    The ``returns`` passed to the financial metrics is the **sign-based
    strategy return** ``sign(pred) * actual`` — i.e. go long when the
    model predicts a positive next-step log-return, short when it
    predicts negative, flat on an exact zero. This makes Sharpe /
    Sortino / MDD / Calmar describe the model's P&L, not the market's.
    Passing ``actual`` directly (buy-and-hold) would collapse every
    model's financial metrics to the same numbers — the bug this
    function's pre-2026-04-18 version shipped with. Since log-returns
    are small daily, ``sign(r_log) == sign(r_simple)``, so using the
    log-return target as the per-period P&L is fine for Sharpe-style
    ratios; for wider moves, pre-exponentiate before calling
    :func:`compute_all_metrics` directly.

    Args:
        model: Trained :class:`torch.nn.Module`. Moved to ``device``
            and switched to ``eval()`` mode inside the function —
            caller does not need to do either beforehand.
        test_ds: A :class:`~src.data.dataset.TimeSeriesDataset`
            yielding ``(x, y, metadata)`` tuples. ``x`` shape
            ``(seq_len, F)``, ``y`` shape ``(1,)``, ``metadata`` dict
            with keys that PINNs may consume (e.g. ``"prices"``,
            ``"returns"``, ``"dt"``).
        device: Torch device for inference — typically the
            :attr:`Trainer.device` the model was trained on.
        batch_size: DataLoader batch size. Default 64. Does not
            affect results — purely memory/speed tradeoff.

    Returns:
        5-tuple ``(metrics, pred_arr, actual_arr, equity_curve, buy_hold_curve)``
        where ``metrics`` is a flat ``dict[str, float]`` with the nine keys
        produced by :func:`compute_all_metrics`; ``pred_arr`` and
        ``actual_arr`` are 1-D ``np.ndarray`` of shape ``(T,)`` on the
        log-return scale; ``equity_curve`` and ``buy_hold_curve`` are
        cumulative-product equity curves of shape ``(T,)`` starting at 1.0.
    """
    model = model.to(device).eval()
    loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=torch.cuda.is_available(),
        persistent_workers=False,  # single-pass at end of training; no reuse.
    )
    is_pinn = isinstance(model, BasePINN)

    preds, actuals = [], []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            if is_pinn:
                out = model(x, meta)
                if isinstance(out, tuple):
                    out = out[0]
            else:
                out = model(x)
            preds.append(out.detach().cpu().numpy().reshape(-1))
            actuals.append(y.detach().cpu().numpy().reshape(-1))

    pred_arr = np.concatenate(preds)
    actual_arr = np.concatenate(actuals)
    strategy_returns = np.sign(pred_arr) * actual_arr
    equity_curve = np.cumprod(1.0 + strategy_returns)
    buy_hold_curve = np.cumprod(1.0 + actual_arr)
    metrics = compute_all_metrics(pred_arr, actual_arr, strategy_returns)
    return metrics, pred_arr, actual_arr, equity_curve, buy_hold_curve
