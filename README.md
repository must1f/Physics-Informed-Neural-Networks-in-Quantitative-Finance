# FINN v2: Physics-Informed Neural Networks for Financial Forecasting

A dissertation codebase asking one fairly stubborn question: do physics constraints from quantitative finance actually help LSTM-based models predict next-day S&P 500 returns, or are they expensive regularisers dressed up as theory?

The short answer, across 6 walk-forward folds and 21 model variants, is mostly no. That is documented here too.

---

## What this is

FINN trains and evaluates neural networks that embed financial stochastic differential equations directly into their loss functions. The physics constraints (GBM, Ornstein-Uhlenbeck, Black-Scholes, Hawkes process) are not hard constraints; they are penalty terms that push predicted log-returns toward what the SDE would imply. The question is whether that signal is informative or just noise.

Six research questions guide the experiments:

- **RQ1**: Do PINNs outperform plain LSTM/GRU baselines?
- **RQ2**: Which physics constraint (GBM vs OU vs BS) performs best?
- **RQ3**: Does the GBM+OU combination produce additive gains?
- **RQ4**: Can Hawkes process constraints capture volatility clustering?
- **RQ5**: Do the novel stacked and residual PINN architectures improve on single-encoder PINNs?
- **RQ6**: Does any model beat buy-and-hold on risk-adjusted returns?

---

## Repository layout

```
submission/
├── src/                        # Python research package
│   ├── models/                 # 21 model variants
│   │   ├── base_pinn.py        # Abstract base class (BasePINN)
│   │   ├── pinn.py             # Core PINN (single encoder)
│   │   ├── stacked_pinn.py     # Dual-encoder stacked PINN (RQ5)
│   │   ├── residual_pinn.py    # Residual-correction PINN (RQ5)
│   │   ├── hawkes_decoupled_pinn.py  # Two-head Hawkes model (RQ4 v2)
│   │   ├── baselines.py        # LSTM, GRU, BiLSTM, Transformer
│   │   ├── classical.py        # GARCH, random walk, historical mean
│   │   └── registry.py         # Central model factory (build_model)
│   ├── losses/
│   │   ├── physics.py          # GBM, OU, BS, Langevin, Hawkes constraints
│   │   ├── composite.py        # Weighted sum of data + physics losses
│   │   └── data_losses.py      # MSE and MAE data-fit terms
│   ├── data/
│   │   ├── fetcher.py          # yfinance downloader with caching
│   │   ├── features.py         # 18-feature engineering pipeline
│   │   ├── pipeline.py         # End-to-end data pipeline
│   │   ├── dataset.py          # PyTorch Dataset (sequence windowing)
│   │   └── splitter.py         # Walk-forward fold splitter
│   ├── training/
│   │   ├── trainer.py          # Training loop with early stopping
│   │   ├── runner.py           # Single experiment orchestrator
│   │   ├── walk_forward.py     # Multi-fold, multi-seed orchestration
│   │   ├── scheduler.py        # Physics λ warmup (cosine/linear/step)
│   │   ├── config.py           # TrainingConfig dataclass
│   │   └── result.py           # TrainingResult / WalkForwardResult
│   ├── evaluation/
│   │   ├── metrics.py          # RMSE, MAE, R², DA, Sharpe, Sortino, etc.
│   │   ├── evaluator.py        # Post-training evaluation runner
│   │   ├── backtester.py       # Long/flat strategy backtester
│   │   ├── forecast_tests.py   # Diebold-Mariano test
│   │   ├── comparison.py       # Cross-model comparison utilities
│   │   └── volatility.py       # GARCH volatility evaluation
│   ├── artefacts/
│   │   ├── orchestration.py    # Result serialisation/loading
│   │   └── plotting.py         # Equity curve and metric plots
│   └── utils/
│       ├── logger.py           # Loguru-based logging setup
│       └── reproducibility.py  # Seed setting (torch, numpy, random)
├── notebooks/                  # Colab experiment notebooks (in order)
│   ├── 0_eda.ipynb
│   ├── 0b_feature_screening.ipynb
│   ├── 1_classical_baselines.ipynb
│   ├── 2_neural_baselines.ipynb
│   ├── 3_bilstm_analysis.ipynb
│   ├── 4_core_pinns.ipynb
│   ├── 4_core_pinns_walk_forward.ipynb
│   ├── 5_gbm_ou_pinn.ipynb
│   ├── 6_BS_PINN_Global_PINN.ipynb
│   ├── 6_BS_Global_WF.ipynb
│   ├── 7_Hawkes_PINN.ipynb
│   ├── 7_Hawkes_WF.ipynb
│   ├── 8_stacked_residual_pinns.ipynb
│   ├── 8_stacked_residual_WF.ipynb
│   ├── 9_hawkes_v2.ipynb
│   └── 10_final_dm_test.ipynb
├── tests/                      # pytest suite (36 files, 589 tests)
├── configs/
│   └── dissertation.yaml       # Single source of truth for all runs
└── pyproject.toml
```

---

## Models

There are 21 model variants across four groups.

**Classical baselines** (no learning): random walk, historical mean, persistence, GARCH.

**Neural baselines**: LSTM, GRU, BiLSTM, attention-LSTM, Transformer. These have no physics terms. They are the comparison point for everything else.

**Core PINNs** (single LSTM encoder, physics penalty added to loss):

| Name | Physics constraint(s) |
|---|---|
| `baseline_pinn` | none (ablation) |
| `gbm_pinn` | Geometric Brownian Motion |
| `ou_pinn` | Ornstein-Uhlenbeck |
| `bs_pinn` | Black-Scholes |
| `gbm_ou_pinn` | GBM + OU |
| `global_pinn` | GBM + OU + BS + Langevin |
| `hawkes_pinn` | Hawkes intensity |
| `hawkes_ou_pinn` | Hawkes + OU |

**Novel architectures** (RQ5 — do novel dual-encoder and residual-correction designs outperform single-encoder PINNs?):

| Name | Architecture |
|---|---|
| `stacked_pinn` | Two LSTM encoders in series — second encoder sees first encoder's hidden state |
| `residual_pinn` | Base encoder predicts return; second head corrects its residual error |
| `hawkes_v2_pinn` | Decoupled mean/variance heads, Gaussian QMLE NLL loss (RQ4 follow-up) |
| `hawkes_v2_multiscale_pinn` | Two-component Hawkes kernel (short + long memory) |

---

## Data and features

Data: S&P 500 (`^GSPC`) daily closes with VIX and 10-year Treasury yield as macro features, 2010-01-01 to 2024-12-31, downloaded via yfinance.

The feature set went through a VIF and mutual information screen. Three candidates were dropped: `bollinger_upper` (VIF=270), `simple_return` (r=1.0 with `log_return`, VIF=3634) and `rolling_skewness_20` (MI=0.000). What remains:

```
log_return, rolling_volatility_5, rolling_volatility_20, atr_14, vol_premium,
momentum_5, momentum_20, rsi_14, macd, macd_signal, bollinger_lower,
close_normalized, volume_normalized, vix_level, vix_change, overnight_gap,
tnx_level, tnx_change
```

That is 18 features. Sequence length is 60 trading days (~3 months).

---

## Walk-forward protocol

Test years: 2018, 2019, 2020, 2021, 2022, 2023 (6 folds). Each fold has a 4-month validation buffer between training and test to prevent look-ahead. Three random seeds per fold (42, 123, 456) give 18 trials per model. The final Diebold-Mariano test is run in `10_final_dm_test.ipynb` across all folds and seeds.

Crash-resume is built in: if a fold×seed result JSON already exists on disk, it is skipped. Long Colab runs can be interrupted and restarted without rerunning completed folds.

---

## Installation

```bash
pip install -e ".[dev]"
```

For Colab:

```bash
pip install -e ".[colab]"
```

Requires Python 3.10+.

---

## Running the tests

```bash
pytest
```

The suite covers physics residual correctness, metric functions, trainer smoke tests, walk-forward splits and backtester logic. Tests run against real PyTorch modules, not mocks.

---

## Configuration

Everything lives in `configs/dissertation.yaml`. The key sections:

```yaml
data:
  tickers: ["^GSPC"]
  start_date: "2010-01-01"
  end_date: "2024-12-31"
  sequence_length: 60

physics:
  lambdas:
    gbm: 0.1
    ou: 0.1
    bs: 0.01        # lower than others — BS residual is noisier on daily data
    hawkes: 0.1
  warmup_epochs: 20
  warmup_strategy: "cosine"

walk_forward:
  test_years: [2018, 2019, 2020, 2021, 2022, 2023]
  val_months: 4

seeds: [42, 123, 456]
```

λ values were set by held-out validation. The BS constraint gets a lower weight because on daily data the Black-Scholes residual is noisier than GBM or OU. Raising it to 0.1 causes training collapse in some seeds.

---

## Architecture details

**Physics constraints** are implemented as `nn.Module` subclasses following a strategy pattern. Each one implements a single `residual(predictions, metadata)` method returning a scalar. Adding or removing a constraint does not touch the model or training loop. You pass a list of constraint instances to `PINNModel`.

**OUConstraint and LangevinConstraint** are initialised with `mu_init`, the bootstrap mean log-return over the training fold. Without this, both constraints start with a long-run mean of zero, which causes all seeds to collapse to identical predictions regardless of data.

**HawkesConstraint** is initialised with `mu0_init = mean(r²)` over training, so the baseline intensity starts on the right scale (roughly 1e-4 for daily returns) rather than the legacy softplus(0) ≈ 0.69.

**StackedPINN** runs two LSTM encoders in sequence, where the second encoder takes the first encoder's hidden state as additional input. **ResidualPINN** trains a base encoder to predict the return, then adds a second head that predicts the residual error of the base, combining both at inference. Both share the same GBM+OU constraints as `gbm_ou_pinn` so the RQ5 comparison isolates architectural differences.

---

## Reproducibility

Each experiment seed calls `set_seed(seed)` from `src/utils/reproducibility.py` before data loading and model construction. `torch.backends.cudnn.deterministic = True` is set. Results across seeds are averaged in the walk-forward aggregation.

---

## Notebooks and what each one does

| Notebook | Content |
|---|---|
| `0_eda.ipynb` | Price series, return distributions, autocorrelation, regime detection |
| `0b_feature_screening.ipynb` | VIF analysis, mutual information, feature drop decisions |
| `1_classical_baselines.ipynb` | Random walk, historical mean, persistence, GARCH |
| `2_neural_baselines.ipynb` | LSTM, GRU, BiLSTM, Transformer (no physics) |
| `3_bilstm_analysis.ipynb` | BiLSTM ablation and fold-level analysis |
| `4_core_pinns.ipynb` | Core PINN single-fold training (RQ1/RQ2) |
| `4_core_pinns_walk_forward.ipynb` | Walk-forward for RQ1/RQ2 |
| `5_gbm_ou_pinn.ipynb` | GBM+OU combined constraint (RQ3) |
| `6_BS_PINN_Global_PINN.ipynb` | Black-Scholes and global PINN (RQ3 continued) |
| `6_BS_Global_WF.ipynb` | Walk-forward for BS and global PINN |
| `7_Hawkes_PINN.ipynb` | Hawkes intensity constraint (RQ4) |
| `7_Hawkes_WF.ipynb` | Walk-forward for Hawkes models |
| `8_stacked_residual_pinns.ipynb` | Novel architectures (RQ5) |
| `8_stacked_residual_WF.ipynb` | Walk-forward for RQ5 models |
| `9_hawkes_v2.ipynb` | Revised Hawkes: NLL loss, stationarity reparametrisation, decoupled variance head |
| `10_final_dm_test.ipynb` | Diebold-Mariano pairwise tests across all models (RQ6) |


---

## Dependencies

Core: `torch>=2.0`, `pandas>=2.0`, `numpy>=1.24`, `scipy>=1.10`, `scikit-learn`, `statsmodels`, `yfinance`, `pandas-ta`, `loguru`, `pyyaml`, `tqdm`.

Dev extras: `pytest`, `ruff`, `black`, `mypy`.

---

## Notes on results

The null results are real. Across the walk-forward folds, directional accuracy sits around 50% for every PINN variant. Sharpe ratios are generally negative or negligible after a 10 bps transaction cost. The Diebold-Mariano tests do not reject equal predictive accuracy between any PINN and its neural baseline counterpart.

There are a few reasonable explanations. The physics constraints here encode equilibrium dynamics (mean reversion, log-normal diffusion, martingale pricing) that assume something approximating stationarity. The 2018–2023 test window had COVID-19, a 2022 rate shock and a prolonged low-volatility bull run before that. If the regime switches faster than the model can adapt, the physics term is pulling predictions toward the wrong equilibrium. Whether a longer training window, adaptive constraint weighting or a different SDE family would change that is left open.

---

## Licence

MIT — see [LICENSE](LICENSE).
