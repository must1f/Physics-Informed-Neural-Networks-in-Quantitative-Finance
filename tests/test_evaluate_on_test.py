"""Tests the bridge from trained model + test dataset → metrics dict."""
import numpy as np
import pandas as pd
import torch

from src.data.dataset import TimeSeriesDataset
from src.evaluation.evaluator import evaluate_on_test
from src.models.registry import build_model


def _toy_df(n=400):
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Close": prices,
        "log_return": np.concatenate([[0.0], np.diff(np.log(prices))]),
        **{f"f{i}": rng.standard_normal(n) for i in range(14)},
    }, index=idx)


def test_evaluate_on_test_returns_metric_dict():
    df = _toy_df()
    _, _, test_ds = TimeSeriesDataset.from_dataframe(
        df, seq_len=20, target_col="log_return",
    )
    input_dim = test_ds.features.shape[1]
    model = build_model("lstm", input_dim=input_dim)
    metrics, _, _, _, _ = evaluate_on_test(model, test_ds, device=torch.device("cpu"))
    for key in ("rmse", "mae", "r_squared", "sharpe", "max_drawdown"):
        assert key in metrics
        assert isinstance(metrics[key], float)
