"""Sign-based long/short backtester with transaction costs.

Strategy rule: at step t, go long (+1) if prediction[t] > 0 else short
(-1). Transaction cost is charged on the absolute change in position
between consecutive steps (turnover), scaled by ``transaction_cost``.
The first step incurs a cost proportional to ``|position[0]|`` to
account for opening the initial position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.constants import TRANSACTION_COST
from src.evaluation.metrics import (
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


@dataclass
class BacktestResult:
    """Immutable container bundling all artefacts from one backtest.

    All arrays are 1-D ``np.ndarray`` of length ``N`` aligned by
    time-step (same length as the ``predictions`` passed to
    :meth:`SignBasedBacktester.run`). Scalars are annualised assuming
    daily data (252 periods/year).

    Attributes:
        positions: Integer array in ``{+1, -1}`` of shape ``(N,)`` —
            ``+1`` long when ``pred > 0`` else ``-1`` short.
        strategy_returns: Per-step simple return **net of transaction
            costs**, shape ``(N,)``. Computed as
            ``position_t * actual_return_t - transaction_cost *
            |Δposition_t|``.
        equity_curve: Cumulative growth-of-£1,
            ``cumprod(1 + strategy_returns)``, shape ``(N,)``. First
            element is ``1 + strategy_returns[0]``.
        sharpe: Annualised Sharpe ratio of ``strategy_returns``
            (scalar, dimensionless).
        sortino: Annualised Sortino ratio of ``strategy_returns``
            (scalar, dimensionless).
        max_drawdown: Peak-to-trough drawdown of ``equity_curve`` as
            a positive fraction in ``[0, 1]``.
        cumulative_return: Total return over the backtest window,
            ``equity_curve[-1] - 1`` (scalar, simple return).
    """
    positions: np.ndarray
    strategy_returns: np.ndarray
    equity_curve: np.ndarray
    sharpe: float
    sortino: float
    max_drawdown: float
    cumulative_return: float


class SignBasedBacktester:
    """Vectorised long/short backtester keyed on the sign of the prediction.

    Strategy rule: at step ``t``, ``position_t = +1`` if
    ``prediction_t > 0`` else ``-1`` (i.e. zeros/negatives are short).
    This is the simplest strategy that exposes directional skill
    without introducing hyperparameters — appropriate for a
    dissertation comparison where the point is model quality, not
    execution edge.

    Transaction costs are charged on **turnover** — the absolute change
    in position between consecutive steps. Cost at step ``t`` is
    ``transaction_cost * |position_t - position_{t-1}|`` with the
    convention ``position_{-1} = 0``, so the first bar always pays
    ``transaction_cost`` for opening the initial position.

    Args:
        transaction_cost: Per-unit-turnover cost as a decimal (e.g.
            ``0.001`` = 10 basis points). Default
            :data:`~src.constants.TRANSACTION_COST`. Must be
            non-negative.

    Raises:
        ValueError: If ``transaction_cost < 0``.
    """

    def __init__(self, transaction_cost: float = TRANSACTION_COST):
        if transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative")
        self.transaction_cost = float(transaction_cost)

    def run(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
    ) -> BacktestResult:
        """Run the sign-based strategy and compute summary metrics.

        Only the *sign* of ``predictions`` is used, so their scale is
        irrelevant — callers may pass raw model outputs, log-returns,
        or simple returns interchangeably.

        Args:
            predictions: 1-D array of model predictions, shape
                ``(N,)``. Coerced to float. Positions are
                ``+1`` where ``predictions > 0``, else ``-1``.
            actual_returns: 1-D array of realised **simple returns**
                at the matching time steps, shape ``(N,)``. Values
                must satisfy ``> -1`` so the equity curve stays
                positive.

        Returns:
            :class:`BacktestResult` with per-step ``positions``,
            net-of-cost ``strategy_returns``, ``equity_curve``,
            and the annualised ``sharpe`` / ``sortino`` /
            ``max_drawdown`` / ``cumulative_return`` scalars.

        Raises:
            ValueError: If the two inputs differ in shape or are not
                1-D.
        """
        predictions = np.asarray(predictions, dtype=float)
        actual_returns = np.asarray(actual_returns, dtype=float)
        if predictions.shape != actual_returns.shape:
            raise ValueError(
                f"Shape mismatch: pred {predictions.shape} vs "
                f"actual {actual_returns.shape}"
            )
        if predictions.ndim != 1:
            raise ValueError("Inputs must be 1-D")

        positions = np.where(predictions > 0, 1, -1).astype(int)

        prev = np.concatenate([[0], positions[:-1]])
        turnover = np.abs(positions - prev).astype(float)

        gross = positions * actual_returns
        costs = self.transaction_cost * turnover
        strategy_returns = gross - costs
        equity_curve = np.cumprod(1.0 + strategy_returns)

        return BacktestResult(
            positions=positions,
            strategy_returns=strategy_returns,
            equity_curve=equity_curve,
            sharpe=sharpe_ratio(strategy_returns),
            sortino=sortino_ratio(strategy_returns),
            max_drawdown=max_drawdown(equity_curve),
            cumulative_return=float(equity_curve[-1] - 1.0),
        )
