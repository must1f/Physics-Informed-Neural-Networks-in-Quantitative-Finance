"""Global constants shared across the FINN package.

All three constants are used in :mod:`src.evaluation.metrics` and
:mod:`src.evaluation.backtester`.  Changing a value here propagates to every
metric computation without touching individual functions.

Constants:
    TRANSACTION_COST (float): Per-unit-turnover transaction cost as a decimal
        fraction (0.001 = 10 basis points).  Applied by
        :class:`~src.evaluation.backtester.SignBasedBacktester` on each
        position change.
    RISK_FREE_RATE (float): Annual risk-free rate as a decimal (0.02 = 2%).
        Divided by ``TRADING_DAYS_PER_YEAR`` when computing per-period excess
        returns in :func:`~src.evaluation.metrics.sharpe_ratio` and
        :func:`~src.evaluation.metrics.sortino_ratio`.
    TRADING_DAYS_PER_YEAR (int): Calendar convention for annualisation (252).
        Used in all annualised ratio computations.
"""

TRANSACTION_COST = 0.001       # 10 basis points
RISK_FREE_RATE = 0.02          # 2% annual
TRADING_DAYS_PER_YEAR = 252
