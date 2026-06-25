"""F4 regression test: OU/Langevin mu_init must vary across seeds.

The pre-fix behaviour computed mu_init from the full training split,
which is seed-independent (the split is deterministic; no shuffling).
This caused ``global_pinn`` to produce identical RMSE/DA/Sharpe across
all three seeds in ``notebooks/4_core_pinns_extended.ipynb`` — the
physics attractor was the same for every seed.

After F4, ``mu_init`` is a bootstrap mean over the training returns
seeded by the experiment seed, so distinct seeds produce distinct
initial drift priors while the bootstrap remains a consistent
estimator of the true empirical drift.
"""
import numpy as np

from src.training.runner import _bootstrap_mean


def test_bootstrap_mean_is_reproducible_for_same_seed():
    ret = np.random.RandomState(0).normal(0.0005, 0.01, size=2000)
    a = _bootstrap_mean(ret, seed=42)
    b = _bootstrap_mean(ret, seed=42)
    assert a == b


def test_bootstrap_mean_varies_across_seeds():
    ret = np.random.RandomState(0).normal(0.0005, 0.01, size=2000)
    means = {_bootstrap_mean(ret, seed=s) for s in (42, 123, 7)}
    assert len(means) == 3, f"Expected 3 distinct bootstrap means, got {means}"


def test_bootstrap_mean_close_to_true_mean():
    # Sanity: bootstrap is a consistent estimator of the sample mean.
    ret = np.random.RandomState(0).normal(0.0005, 0.01, size=20_000)
    boot = _bootstrap_mean(ret, seed=42)
    true = float(np.mean(ret))
    # Standard error of bootstrap mean ≈ σ / √n ≈ 7e-5 here; 5σ window.
    assert abs(boot - true) < 5e-4, (boot, true)


def test_bootstrap_mean_handles_empty():
    assert _bootstrap_mean(np.array([]), seed=42) == 0.0
