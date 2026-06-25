import pandas as pd
import numpy as np
import pytest
from src.data.splitter import WalkForwardSplitter


def _make_df(start="2010-01-01", end="2023-12-31") -> pd.DataFrame:
    """Minimal daily DataFrame with DatetimeIndex."""
    idx = pd.bdate_range(start, end)  # business days only
    return pd.DataFrame({"log_return": np.random.randn(len(idx)) * 0.01}, index=idx)


def test_splitter_yields_six_folds():
    df = _make_df()
    folds = list(WalkForwardSplitter(df, test_years=[2018, 2019, 2020, 2021, 2022, 2023]))
    assert len(folds) == 6


def test_splitter_fold_indices_are_zero_based():
    df = _make_df()
    folds = list(WalkForwardSplitter(df, test_years=[2018, 2019]))
    assert folds[0][0] == 0
    assert folds[1][0] == 1


def test_splitter_test_covers_full_year():
    df = _make_df()
    _, train_df, val_df, test_df = list(WalkForwardSplitter(df, test_years=[2018]))[0]
    assert test_df.index.year.unique().tolist() == [2018]


def test_splitter_no_overlap_between_splits():
    df = _make_df()
    _, train_df, val_df, test_df = list(WalkForwardSplitter(df, test_years=[2018]))[0]
    assert len(set(train_df.index) & set(val_df.index)) == 0
    assert len(set(val_df.index) & set(test_df.index)) == 0
    assert len(set(train_df.index) & set(test_df.index)) == 0


def test_splitter_expanding_window_grows():
    df = _make_df()
    folds = list(WalkForwardSplitter(df, test_years=[2018, 2019, 2020]))
    train_lens = [len(f[1]) for f in folds]
    assert train_lens[0] < train_lens[1] < train_lens[2]


def test_splitter_val_window_two_months():
    df = _make_df()
    _, train_df, val_df, test_df = list(WalkForwardSplitter(df, test_years=[2018], val_months=2))[0]
    # Val should be Nov–Dec 2017 (business days only, ~42 days)
    assert val_df.index.min() >= pd.Timestamp("2017-11-01")
    assert val_df.index.max() <= pd.Timestamp("2017-12-31")
    assert len(val_df) > 30  # sanity: at least 30 business days
