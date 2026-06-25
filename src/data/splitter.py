"""Expanding-window walk-forward data splitter for time series CV."""
from __future__ import annotations

from typing import Iterator

import pandas as pd


class WalkForwardSplitter:
    """Generate expanding-window (train, val, test) fold slices for walk-forward CV.

    For each test year T, produces three non-overlapping DataFrame slices:

    - **train**: all rows with index before the val window start.
    - **val**: last ``val_months`` calendar months of year (T-1), i.e.
      ``[Nov 1, Dec 31]`` for ``val_months=2``. Exact boundaries are
      determined by the DatetimeIndex, so market holidays are handled
      correctly.
    - **test**: full calendar year T, sliced as ``df.loc[str(T)]``.

    The train window grows by one year per fold (expanding window). Val
    and test windows are fixed-size per fold.

    Args:
        df: Feature DataFrame with a ``DatetimeIndex``. Must span at
            least from before the first test year minus one year (for val)
            to the end of the last test year.
        test_years: Ordered list of calendar years to use as test windows.
            E.g. ``[2018, 2019, 2020, 2021, 2022, 2023]`` for 6 folds.
        val_months: Number of calendar months immediately before each test
            year to reserve for validation. Default 2 (~42 business days).

    Yields:
        Tuples of ``(fold_idx, train_df, val_df, test_df)`` — zero-indexed.
        All three DataFrames share the same columns as the input ``df``.

    Raises:
        ValueError: If any test year has no data in ``df``.

    Examples:
        # In a notebook cell:
        from src.data.splitter import WalkForwardSplitter

        for fold_idx, train_df, val_df, test_df in WalkForwardSplitter(df, test_years=[2018, 2019]):
            print(f"Fold {fold_idx}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        test_years: list[int],
        val_months: int = 2,
    ) -> None:
        """Store the feature DataFrame and fold configuration.

        Args:
            df: Feature DataFrame with a ``DatetimeIndex``. Must span at
                least from before the first test year minus one year (for val)
                to the end of the last test year.
            test_years: Ordered list of calendar years to use as test windows
                (e.g. ``[2018, 2019, 2020, 2021, 2022, 2023]``).
            val_months: Number of calendar months immediately before each
                test year to reserve for validation. Default 2 (~42 business
                days).
        """
        self._df = df
        self._test_years = test_years
        self._val_months = val_months

    def __iter__(self) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Iterate over walk-forward folds, yielding one tuple per test year.

        Yields:
            Tuple of ``(fold_idx, train_df, val_df, test_df)`` where:

            * ``fold_idx``: zero-based integer fold index.
            * ``train_df``: all rows strictly before the validation window
              start; shares the same columns as the input ``df``.
            * ``val_df``: last ``val_months`` calendar months of year
              ``(test_year - 1)``; typically Nov–Dec for ``val_months=2``.
            * ``test_df``: all rows in calendar year ``test_year`` (sliced
              with ``df.loc[str(test_year)]``).

        Raises:
            ValueError: If any test year has no data in ``df``.
        """
        for fold_idx, test_year in enumerate(self._test_years):
            val_end = pd.Timestamp(f"{test_year - 1}-12-31")
            val_start = val_end - pd.DateOffset(months=self._val_months) + pd.DateOffset(days=1)

            train_df = self._df.loc[: val_start - pd.DateOffset(days=1)]
            val_df = self._df.loc[val_start:val_end]
            test_df = self._df.loc[str(test_year)]

            if test_df.empty:
                raise ValueError(
                    f"No data for test year {test_year}. "
                    f"DataFrame spans {self._df.index.min()} – {self._df.index.max()}."
                )

            yield fold_idx, train_df, val_df, test_df
