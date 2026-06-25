"""Tests for walk_forward orchestration and aggregation."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.training.config import TrainingConfig, WalkForwardConfig
from src.training.result import TrainingResult, WalkForwardResult
from src.training.walk_forward import aggregate_walk_forward, run_walk_forward


def _make_df(start="2010-01-01", end="2023-12-31") -> pd.DataFrame:
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"log_return": np.random.randn(len(idx)) * 0.01}, index=idx)


def _fake_result(model_name="gbm_pinn", rmse=0.01, sharpe=1.0) -> TrainingResult:
    r = TrainingResult(model_name=model_name)
    r.test_metrics = {"rmse": rmse, "sharpe": sharpe, "directional_accuracy": 0.55}
    r.test_preds = np.array([0.001, -0.002])
    r.test_actual = np.array([0.001, -0.001])
    return r


def _base_config(wf_years=None) -> TrainingConfig:
    return TrainingConfig(
        epochs=2,
        seeds=[42, 123],
        walk_forward=WalkForwardConfig(
            test_years=wf_years or [2018, 2019],
            val_months=2,
        ),
    )


def test_run_walk_forward_calls_run_experiment_per_fold_and_seed(tmp_path):
    config = _base_config(wf_years=[2018, 2019])
    df = _make_df()

    with patch("src.training.walk_forward.run_experiment", return_value=_fake_result()) as mock_exp:
        result = run_walk_forward(config, "gbm_pinn", df, base_dir=str(tmp_path))

    # 2 folds × 2 seeds = 4 calls
    assert mock_exp.call_count == 4


def test_run_walk_forward_skips_completed_fold_seed(tmp_path):
    config = _base_config(wf_years=[2018])
    df = _make_df()

    # Pre-write sentinel so fold 0 / seed 42 appears complete
    sentinel = tmp_path / "gbm_pinn" / "fold_0" / "seed_42" / "gbm_pinn_result.json"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text(json.dumps(_fake_result().to_dict()))

    with patch("src.training.walk_forward.run_experiment", return_value=_fake_result()) as mock_exp:
        run_walk_forward(config, "gbm_pinn", df, base_dir=str(tmp_path))

    # Only 1 call (seed 123); seed 42 was skipped
    assert mock_exp.call_count == 1


def test_aggregate_walk_forward_computes_mean_and_std(tmp_path):
    model_name = "gbm_pinn"
    seeds = [42, 123]
    test_years = [2018, 2019]

    for fold_idx in range(2):
        for seed in seeds:
            d = tmp_path / model_name / f"fold_{fold_idx}" / f"seed_{seed}"
            d.mkdir(parents=True)
            result = _fake_result(rmse=0.01 * (fold_idx + 1), sharpe=1.0 + fold_idx * 0.2)
            (d / f"{model_name}_result.json").write_text(json.dumps(result.to_dict()))

    wf = aggregate_walk_forward(str(tmp_path), model_name)

    assert isinstance(wf, WalkForwardResult)
    assert wf.n_folds == 2
    assert wf.model_name == model_name
    assert "rmse" in wf.mean_metrics
    assert "sharpe" in wf.std_metrics
    assert wf.mean_metrics["rmse"] == pytest.approx(0.015, abs=1e-6)
