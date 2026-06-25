"""
Physics-informed constraints for financial time series.

Strategy pattern: one ABC (PhysicsConstraint) and five concrete
implementations encoding real financial PDEs / SDEs as differentiable
residual losses.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DAILY_DT: float = 1.0 / 252.0
RISK_FREE_RATE: float = 0.02
EPS: float = 1e-8


# ═══════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════

class PhysicsConstraint(nn.Module, ABC):
    """Base class for all physics-informed loss terms.

    Each subclass encodes a financial PDE/SDE.  The ``residual`` method
    returns a scalar measuring how far the network's predictions deviate
    from the governing equation (lower is better).
    """

    name: str = "base"

    @abstractmethod
    def residual(self, predictions: Tensor, metadata: dict) -> Tensor:
        """Compute the PDE/SDE residual.

        Args:
            predictions: Model output, typically ``[batch, 1]``.
            metadata: Dictionary carrying auxiliary tensors (prices,
                returns, volatilities, etc.) needed by the specific
                constraint.

        Returns:
            Scalar residual loss (mean-squared residual).
        """


# ═══════════════════════════════════════════════════════════════════════════
# 1. Geometric Brownian Motion
# ═══════════════════════════════════════════════════════════════════════════

class GBMConstraint(PhysicsConstraint):
    """Geometric Brownian Motion: ``dS = mu*S*dt + sigma*S*dW``.

    Penalises the predicted next-step log-return for deviating from the
    Itô-corrected drift estimated from the historical price window::

        E[ ln(S_{t+1}/S_t) ] = (mu - sigma^2 / 2) * dt

    Estimation strategy
    -------------------
    * ``mu`` is estimated from *simple* returns ``S_{t+1}/S_t - 1`` so the
      Itô correction is applied exactly once (see audit note below).
    * ``sigma`` is estimated as the sample standard deviation of log
      returns, annualised by ``1/sqrt(dt)``.
    * Both are per-sample statistics — no learnable parameters.

    Inputs
    ------
    predictions : Tensor ``[batch, 1]``
        Model output. **Must represent the next-step log-return**
        ``ln(S_{t+1}/S_t)`` (i.e. the raw model head output for a log-return
        target). If your model predicts simple returns or prices, convert
        before passing in.
    metadata : dict with
        ``prices`` : Tensor ``[batch, seq_len]`` or ``[batch, seq_len, 1]``
            **Raw (un-normalised) close prices** on the original price
            scale. Log returns are derived internally.

    Returns
    -------
    Tensor (scalar)
        Mean-squared residual between the predicted log-return and the
        Itô-corrected GBM drift ``(mu_hat - 0.5*sigma_hat^2) * dt``.
        Always non-negative.
    """

    name = "gbm"

    def __init__(self, dt: float = DAILY_DT) -> None:
        """Initialise GBMConstraint.

        Args:
            dt: Time-step in years. Default ``1/252`` (daily). No learnable
                parameters; all statistics are estimated from the batch.
        """
        super().__init__()
        self.dt = dt

    def residual(self, predictions: Tensor, metadata: dict) -> Tensor:
        prices = metadata["prices"]
        if prices.dim() == 3:
            prices = prices.squeeze(-1)

        # Simple returns → unbiased estimator of mu (no Itô correction baked in).
        simple_returns = prices[:, 1:] / (prices[:, :-1] + EPS) - 1.0
        mu_hat = simple_returns.mean(dim=1, keepdim=True) / self.dt

        # Log-return std is the standard volatility estimator.
        log_returns = torch.log(prices[:, 1:] / (prices[:, :-1] + EPS) + EPS)
        sigma_hat = log_returns.std(dim=1, keepdim=True) / math.sqrt(self.dt)

        pred = predictions.view(-1, 1)

        # Itô correction applied exactly once: E[d ln S] = (mu - sigma^2/2) dt.
        drift = mu_hat - 0.5 * sigma_hat ** 2
        residual = pred - drift * self.dt
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Ornstein-Uhlenbeck
# ═══════════════════════════════════════════════════════════════════════════

class OUConstraint(PhysicsConstraint):
    """Ornstein-Uhlenbeck mean reversion: ``dX = theta*(mu - X)*dt + sigma*dW``.

    Given the last observed return ``X_t``, the Euler step gives::

        E[X_{t+1} | X_t] = X_t + theta * (mu - X_t) * dt

    Learnable parameters
    --------------------
    theta   : mean-reversion speed (``softplus``, always > 0). Init: 1.0.
    sigma   : volatility (``softplus``, ≥ 0). API completeness; not used in
              the drift residual.

    Fixed buffers (non-learnable)
    -----------------------------
    mu      : long-run mean (``register_buffer``, fixed at ``mu_init`` for
              the lifetime of the module). Returns are signed so no softplus.
              Fixed rather than learnable to prevent gradient-driven drift
              during early training epochs before the physics warmup engages —
              the seed=123 catastrophic flip observed in notebook 4 was caused
              by mu drifting to a large negative value in epoch 1, pulling the
              LSTM into a permanent short-bias basin (2026-04-20 fix). The
              bootstrap-seeded ``mu_init`` already encodes the empirical drift;
              further gradient updates add instability without improving prior
              fidelity.

    Parameters
    ----------
    dt : float
        Time-step in years. Default ``1/252`` (daily).
    mu_init : float
        Initial value for the **fixed buffer** ``mu`` (non-learnable). Expected to be
        a **seed-varying bootstrap mean** over the training-split
        log-returns produced by
        :func:`src.training.runner._bootstrap_mean`. The bootstrap is a
        consistent estimator of the empirical drift (~+5 × 10⁻⁴/day for
        S&P 500) while injecting per-seed variation in the initial
        attractor — without it, multi-seed runs collapse to identical
        predictions because the full-sample mean is seed-independent
        (2026-04-19 audit of ``notebooks/4_core_pinns_extended.ipynb``).
        Default ``0.0`` preserves backward-compatibility for zero-drift
        assets and unit tests that construct the constraint directly.

    Inputs (``residual``)
    ---------------------
    predictions : Tensor ``[batch, 1]``
        Model output for the next-step return on the **same scale** as the
        ``returns`` metadata tensor (typically log-returns).
    metadata : dict
        ``returns`` : Tensor ``[batch, seq_len]`` or ``[batch, seq_len, 1]``
            Historical return sequence (log-returns, not prices). The last
            column is treated as ``X_t``. requires_grad not needed.

    Returns
    -------
    Tensor (scalar, non-negative)
        Mean-squared residual between ``predictions`` and the OU expected
        next-step return ``X_t + theta*(mu - X_t)*dt``.
    """

    name = "ou"

    def __init__(self, dt: float = DAILY_DT, mu_init: float = 0.0) -> None:
        super().__init__()
        self.dt = dt
        self._theta_raw = nn.Parameter(torch.tensor(1.0))
        # mu is a fixed buffer — NOT a learnable parameter. Seeded from the
        # bootstrap training mean via mu_init so the prior is physically
        # correct, but frozen to prevent gradient-driven drift before warmup.
        self.register_buffer("mu", torch.tensor(float(mu_init)))
        self._sigma_raw = nn.Parameter(torch.tensor(0.0))

    @property
    def theta(self) -> Tensor:
        """Mean-reversion speed; softplus-constrained scalar > 0."""
        return F.softplus(self._theta_raw)

    @property
    def sigma(self) -> Tensor:
        """Volatility scale; softplus-constrained scalar >= 0. Not used in drift residual."""
        return F.softplus(self._sigma_raw)

    def residual(self, predictions: Tensor, metadata: dict) -> Tensor:
        """MSE between predicted return and OU Euler-Maruyama step; see class docstring for full spec."""
        returns = metadata["returns"]
        if returns.dim() == 3:
            returns = returns.squeeze(-1)

        last_return = returns[:, -1:]  # X_t, shape [batch, 1]
        pred = predictions.view(-1, 1)

        # Euler-Maruyama expected value: X_t + theta * (mu - X_t) * dt.
        # mu is a fixed buffer (non-learnable), seeded at mu_init.
        expected = last_return + self.theta * (self.mu - last_return) * self.dt

        residual = pred - expected
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Black-Scholes no-arbitrage PDE
# ═══════════════════════════════════════════════════════════════════════════

class BlackScholesConstraint(PhysicsConstraint):
    """Black-Scholes physical-measure (P) drift constraint::

        E[ln(S_{t+1}/S_t)] = (μ̂ − σ²/2) · dt

    Penalises the model's return prediction for deviating from the
    Black-Scholes drift under the **physical measure P**, where μ̂ is the
    empirical drift estimated per-batch from the rolling price window.

    Measure correction rationale
    ----------------------------
    The previous formulation used the risk-neutral (Q-measure) drift
    ``(r − σ²/2)·dt`` with ``r = 0.02``.  For S&P 500 daily data this
    gives ~8×10⁻⁶/day, roughly 50× smaller than the empirical equity
    drift (~4×10⁻⁴/day).  The Q-measure prior therefore actively fought
    the data signal.  Replacing ``r`` with the per-batch rolling mean
    ``μ̂`` aligns the constraint with the data-generating process under P,
    while preserving the Itô correction (−σ²/2·dt) and the full
    Black-Scholes GBM derivation.

    Learnable parameters
    --------------------
    ``_sigma_log_scale`` : additive log correction to realised volatility —
        ``σ_eff = σ_data · exp(log_scale)``.  Initialised to 0 (identity).

    Parameters
    ----------
    risk_free_rate : float
        Annualised risk-free rate ``r``. Retained as a fallback for
        degenerate windows with fewer than 2 price steps; not used in
        normal operation.
    dt : float
        Time-step in years. Default ``1/252`` (daily).

    Inputs (``residual``)
    ---------------------
    predictions : Tensor ``[batch, 1]``
        **Normalised** model output (z-scored next-step log return).
    metadata : dict with
        ``prices`` : ``[batch, seq_len]`` or ``[batch, seq_len+1]``
            Raw (un-normalised) close prices. Injected by
            :meth:`PINNModel._build_physics_metadata`; recovered from the
            normalised feature column when not supplied by the dataset.
        ``volatilities`` : ``[batch, 1]`` or ``[batch]``
            Annualised realised volatility σ on the raw scale.
            Provided by :meth:`PINNModel._build_physics_metadata` from
            the model's ``vol_head``.
        ``target_mean`` : ``[batch, 1]`` or scalar
            Training-set log-return mean — used to de-normalise predictions.
        ``target_std`` : ``[batch, 1]`` or scalar
            Training-set log-return std — used to de-normalise predictions.

    Returns
    -------
    Tensor (scalar, non-negative)
        Mean-squared deviation of the de-normalised prediction from the
        physical-measure drift ``(μ̂ − σ_eff²/2) · dt``.
    """

    name = "bs"

    def __init__(
        self,
        risk_free_rate: float = RISK_FREE_RATE,
        dt: float = DAILY_DT,
    ) -> None:
        """Initialise BlackScholesConstraint.

        Args:
            risk_free_rate: Annualised risk-free rate ``r`` (used only as fallback
                when the rolling price window contains fewer than 2 steps).
            dt: Time-step in years. Default ``1/252`` (daily).

        Learnable parameters:
            _sigma_log_scale: Additive log correction to realised volatility
                (unconstrained scalar; initialised to 0 so ``σ_eff = σ_data``
                at construction).
        """
        super().__init__()
        self.r = risk_free_rate
        self.dt = dt
        self._sigma_log_scale = nn.Parameter(torch.tensor(0.0))

    def residual(self, predictions: Tensor, metadata: dict) -> Tensor:
        """MSE between de-normalised predicted return and physical-measure BS drift; see class docstring."""
        volatilities: Tensor = metadata["volatilities"]
        target_mean: Tensor = metadata["target_mean"]
        target_std: Tensor = metadata["target_std"]
        prices: Tensor = metadata["prices"]  # [B, seq_len] or [B, seq_len+1]

        if volatilities.dim() == 1:
            volatilities = volatilities.unsqueeze(-1)
        if prices.dim() == 3:
            prices = prices.squeeze(-1)

        sigma = volatilities * torch.exp(self._sigma_log_scale)

        # De-normalise: predictions are z-scored; drift is on raw return scale.
        pred_raw = predictions.view(-1, 1) * target_std + target_mean

        # Physical-measure (P) drift: E[ln S_{t+1}/S_t] = (mu_hat - sigma²/2)*dt.
        # mu_hat estimated per-batch from the rolling price window — replaces the
        # fixed risk-free rate r used in the Q-measure formulation. With r=0.02
        # and daily S&P 500 vol ~19% the Q-drift is ~8e-6/day vs the empirical
        # P-drift ~4e-4/day; the Q prior actively fought the data signal.
        simple_returns = prices[:, 1:] / (prices[:, :-1] + EPS) - 1.0
        if simple_returns.shape[1] > 0:
            mu_hat = simple_returns.mean(dim=1, keepdim=True) / self.dt
        else:
            # Degenerate window (<2 steps) — fall back to fixed risk-free rate.
            mu_hat = torch.full_like(volatilities, self.r)

        expected_drift = (mu_hat - 0.5 * sigma ** 2) * self.dt

        residual = pred_raw - expected_drift
        return (residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Langevin dynamics
# ═══════════════════════════════════════════════════════════════════════════

class LangevinConstraint(PhysicsConstraint):
    """Overdamped Langevin dynamics::

        dX = -gamma * grad_U(X) * dt + sqrt(2 * gamma * T) * dW

    Potential: ``U(X) = 0.5 * (X - mu_data)^2`` so
    ``grad_U(X) = X - mu_data`` — restoring force toward the training-set
    empirical mean, not toward zero. When ``mu_init=0.0`` (default) this
    reduces to the original ``U(X) = 0.5 * X^2``.

    The residual has two additive parts:

    * **Drift residual** — ``pred - (X_t - gamma*(X_t - mu_data)*dt)``,
      penalised in L2.
    * **Diffusion residual** — ``|pred - X_t| - sqrt(2*gamma*T*dt)``,
      soft-match of the increment magnitude to the Brownian scale, L2.

    Learnable parameters
    --------------------
    gamma       : friction coefficient (softplus, > 0).
    temperature : noise intensity (softplus, ≥ 0).

    Fixed buffer
    ------------
    mu_data : scalar Tensor, not a gradient parameter.
        Equilibrium point of the potential (training-set mean log-return).
        Registered as a buffer so it travels with ``.to(device)`` and
        ``state_dict`` without entering the optimiser.

    Parameters
    ----------
    mu_init : float
        Training-split log-return prior — expected to be a
        **seed-varying bootstrap mean** produced by
        :func:`src.training.runner._bootstrap_mean`. Sets the potential
        equilibrium point ``mu_data``. Bootstrap seeding is required to
        avoid the multi-seed collapse observed in
        ``notebooks/4_core_pinns_extended.ipynb`` where the deterministic
        full-sample mean produced identical attractors (and therefore
        identical predictions) for every seed. Default ``0.0`` preserves
        original zero-mean behaviour for unit tests.
    dt : float
        Time-step in years. Default ``1/252`` (daily).

    Inputs (``residual``)
    ---------------------
    predictions : Tensor ``[batch, 1]``
        Model output for the next-step return (log-return scale).
    metadata : dict
        ``returns`` : Tensor ``[batch, seq_len]`` or ``[batch, seq_len, 1]``
            Historical log-returns. The last column is ``X_t``.

    Returns
    -------
    Tensor (scalar, non-negative)
        Mean of drift² + diffusion² residuals across the batch.
    """

    name = "langevin"

    def __init__(self, dt: float = DAILY_DT, mu_init: float = 0.0) -> None:
        super().__init__()
        self.dt = dt
        self._gamma_raw = nn.Parameter(torch.tensor(0.5))
        self._temperature_raw = nn.Parameter(torch.tensor(-1.0))  # softplus → ~0.31
        # Fixed equilibrium — not a learnable param so it does not compete
        # with gamma/temperature in the optimiser.
        self.register_buffer("mu_data", torch.tensor(float(mu_init)))

    @property
    def gamma(self) -> Tensor:
        """Friction coefficient; softplus-constrained scalar > 0."""
        return F.softplus(self._gamma_raw)

    @property
    def temperature(self) -> Tensor:
        """Noise intensity T; softplus-constrained scalar >= 0."""
        return F.softplus(self._temperature_raw)

    def residual(self, predictions: Tensor, metadata: dict) -> Tensor:
        """MSE of Langevin drift + diffusion residuals; see class docstring for full spec."""
        returns = metadata["returns"]
        if returns.dim() == 3:
            returns = returns.squeeze(-1)

        last_return = returns[:, -1:]  # X_t, shape [batch, 1]
        pred = predictions.view(-1, 1)

        gamma = self.gamma
        T = self.temperature

        # U(X) = 0.5*(X - mu_data)^2  =>  grad_U(X) = X - mu_data.
        # Restores toward the training-set empirical mean, not zero.
        grad_U = last_return - self.mu_data

        # Drift residual: pred - (last_return - gamma * grad_U * dt)
        expected_drift = last_return - gamma * grad_U * self.dt
        drift_residual = pred - expected_drift

        # Diffusion residual: |pred - last_return| vs expected noise magnitude
        expected_diffusion = torch.sqrt(2.0 * gamma * T * self.dt + EPS)
        diffusion_residual = (pred - last_return).abs() - expected_diffusion

        return (drift_residual ** 2 + diffusion_residual ** 2).mean()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Hawkes self-exciting process
# ═══════════════════════════════════════════════════════════════════════════

class HawkesConstraint(PhysicsConstraint):
    """Hawkes self-exciting process for volatility clustering.

    Conditional intensity at the forecast step ``T+1``::

        lambda(T+1) = mu0 + sum_{s=1..T} |r_s| * alpha * exp(-beta * (T+1 - s))

    Each historical absolute return acts as an event mark that excites
    future intensity, with exponentially decaying memory. The timestep
    differences are measured in integer bar units (``dt`` is absorbed into
    ``beta``).

    Learnable parameters
    --------------------
    mu0   : baseline intensity    (softplus, > 0). Initialised via
            ``mu0_init``: when provided (typically training-set ``mean(r²)``),
            ``_mu0_raw`` is set to the inverse-softplus of that value so
            ``softplus(_mu0_raw) == mu0_init`` at construction. This seeds
            the initial intensity ``λ ≈ mu0_init ≈ pred² ∼ σ²``, avoiding
            the legacy default ``softplus(0) = ln 2 ≈ 0.69`` which dominates
            daily ``pred² ∼ 1e-4`` by ~3 orders of magnitude. Default
            ``mu0_init=None`` preserves legacy behaviour for backward
            compatibility with pre-existing tests and checkpoints.
    alpha : excitation magnitude  (softplus, ≥ 0).
    beta  : decay rate            (softplus, > 0).
    _ema_scale (float | None)     — running EMA of raw loss magnitude used
        for normalisation; not an nn.Parameter and absent from
        state_dict(). Resets to None if the constraint is reconstructed
        from a checkpoint — the first post-load batch re-initialises it,
        producing a warm-up epoch before the scale stabilises.
    _EMA_ALPHA (float = 0.1)      — class-level EMA decay rate (fraction
        of new value blended in each call).

    Parameters
    ----------
    mu0_init : float | None
        Empirical baseline-intensity seed on the return-variance scale.
        Compute as ``float(np.mean(train_returns ** 2))`` so initial
        ``λ`` matches ``pred²`` at construction. ``None`` (default)
        keeps the legacy ``_mu0_raw = 0`` init, giving ``mu0 ≈ 0.69``.

    Inputs
    ------
    predictions : Tensor ``[batch, 1]``
        Model output for the **next-step return** (signed). The constraint
        compares ``pred**2`` against the intensity, so the sign is ignored
        but the magnitude matters.
    metadata : dict with
        ``returns`` : Tensor ``[batch, seq_len]`` or ``[batch, seq_len, 1]``
            Historical return sequence. Only the absolute values are used
            as event marks.

    Returns
    -------
    Tensor (scalar)
        EMA-normalised mean of ``(pred**2 - lambda)**2`` across the batch.
        Divided by a running EMA of the raw loss magnitude (alpha=0.1) so
        the term stays O(1) and λ_hawkes acts as a true fractional weight
        relative to the data MSE. Always non-negative.
    """

    name = "hawkes"
    _EMA_ALPHA: float = 0.1

    def __init__(self, mu0_init: float | None = None) -> None:
        super().__init__()
        # Initialised per novel_architectures.md §1.4.
        # When mu0_init is provided, seed _mu0_raw so softplus(_mu0_raw) == mu0_init
        # (within float precision). Uses math.expm1 for a numerically stable
        # inverse-softplus when mu0_init is small-positive (typical: ~1e-4).
        if mu0_init is None:
            _mu0_raw_init = 0.0                               # softplus ≈ 0.69 (legacy)
        else:
            # inverse softplus: _raw = log(exp(y) - 1) = log(expm1(y))
            _mu0_raw_init = math.log(math.expm1(float(mu0_init)))
        self._mu0_raw = nn.Parameter(torch.tensor(_mu0_raw_init))
        self._alpha_raw = nn.Parameter(torch.tensor(-0.5))    # softplus ≈ 0.47
        self._beta_raw = nn.Parameter(torch.tensor(1.5))      # softplus ≈ 1.74
        # EMA scale for loss normalisation — keeps the Hawkes residual
        # at O(1) relative to the data MSE regardless of absolute magnitude.
        # Initialised to None; set on first residual() call.
        self._ema_scale: float | None = None

    @property
    def mu0(self) -> Tensor:
        """Baseline intensity (scalar, > 0); softplus-constrained, seeded at mu0_init."""
        return F.softplus(self._mu0_raw)

    @property
    def alpha(self) -> Tensor:
        """Excitation magnitude (scalar, >= 0); softplus-constrained."""
        return F.softplus(self._alpha_raw)

    @property
    def beta(self) -> Tensor:
        """Decay rate (scalar, > 0); softplus-constrained."""
        return F.softplus(self._beta_raw)

    def residual(self, predictions: Tensor, metadata: dict) -> Tensor:
        """EMA-normalised MSE of (pred² − λ_hawkes); see class docstring for full spec."""
        returns = metadata["returns"]
        if returns.dim() == 3:
            returns = returns.squeeze(-1)

        B, T = returns.shape
        pred = predictions.view(-1, 1)

        # Causal exponential kernel matrix W[s, t] = exp(-beta * (t - s)) for t > s
        # We only need the intensity at timestep T+1
        beta = self.beta
        alpha = self.alpha
        mu0 = self.mu0

        # Time differences from each historical step s to the forecast step T+1
        # (T+1 - s) for s in [1..T]  →  [T, T-1, ..., 1]
        time_diffs = torch.arange(T, 0, -1, device=returns.device, dtype=returns.dtype)

        # Causal kernel weights: alpha * exp(-beta * dt)  →  [T]
        kernel = alpha * torch.exp(-beta * time_diffs)

        # Absolute returns as event marks
        abs_returns = returns.abs()  # [B, T]

        # Intensity at T+1: mu0 + sum_s |r_s| * kernel_s
        intensity = mu0 + (abs_returns * kernel.unsqueeze(0)).sum(dim=1, keepdim=True)  # [B, 1]

        # EMA-normalised Hawkes loss: keeps the term O(1) relative to data MSE.
        residual = pred ** 2 - intensity
        raw_loss = (residual ** 2).mean()

        # Detached EMA update — normalisation must not affect gradients
        # through the scale itself, only through raw_loss.
        raw_val = raw_loss.detach().item()
        if self._ema_scale is None:
            self._ema_scale = raw_val
        else:
            self._ema_scale = (
                self._EMA_ALPHA * raw_val
                + (1.0 - self._EMA_ALPHA) * self._ema_scale
            )
        scale = max(self._ema_scale, 1e-12)
        return raw_loss / scale


# ═══════════════════════════════════════════════════════════════════════════
# 5b. Hawkes v2 — decoupled NLL with stationarity-by-construction kernel
# ═══════════════════════════════════════════════════════════════════════════

class HawkesConstraintV2(PhysicsConstraint):
    """Hawkes v2: Gaussian QMLE NLL + structural match, stationarity by design.

    Fixes the three structural problems of :class:`HawkesConstraint`:

    1. **No sign-flip.** The legacy residual ``|pred² − λ|²`` is symmetric
       in ``pred``'s sign, producing a double-well optimum at ``±√λ``.
       v2 consumes the signed ``target`` via a Gaussian quasi-log-likelihood,
       ``0.5·((target − mean_pred)²/σ² + log σ²)`` — asymmetric, convex in
       ``σ²``, and couples directly to the observed return so the mean head
       has a unique sign-correct optimum.
    2. **Stationarity-by-construction.** Branching ratio ``b`` is the only
       trainable self-excitation scalar, bounded in ``(0, 0.95)`` via
       ``sigmoid`` so ``α·Σexp(−β·Δt) < 1`` is *structurally* guaranteed.
       The excitation coefficient ``α`` is derived, not learnt, from
       ``α = b / kernel_sum(β)``.
    3. **Memory floor on β.** ``β = 0.05 + softplus(raw_β)`` prevents the
       kernel-decay-to-zero degeneracy that made the original run unstable
       on longer windows.

    Residual decomposition
    ----------------------
    Two additive terms (summed unweighted inside the constraint, so the
    outer λ in ``CompositeLoss`` scales them as a pair)::

        nll    = 0.5 · ((target − mean_pred)² / σ² + log σ²)
        struct = (log σ² − log λ_hawkes)²

        residual = mean_over_batch(nll + struct_weight · struct)

    * ``nll`` is the Gaussian QMLE log-likelihood (Bollerslev–Wooldridge
      1992): couples observed next-step returns to the variance head, no
      sign degeneracy.
    * ``struct`` is the physics-informed regulariser that drives the
      variance head toward the Hawkes intensity computed from past
      squared returns with learnable ``(μ₀, b, β)``.  Without ``struct``
      the constraint reduces to GARCH-style QMLE with no self-exciting
      structure; ``struct`` is what makes this a *Hawkes*-informed PINN.

    Learnable parameters
    --------------------
    _mu0_raw      : ``[num_components]``, softplus → ``μ₀_k > 0``. Summed
                    across components to form the batch-wide baseline
                    intensity ``μ₀ = Σ_k softplus(_mu0_raw[k])``.
    _branching_raw: ``[num_components]``, ``b_k = 0.95·sigmoid(raw)`` per
                    component. Hard cap at 0.95 keeps the kernel
                    sub-critical even if several components are active.
                    Initialised from ``N(0, 0.1)`` so each seed starts at a
                    distinct branching ratio (zero-init collapsed all seeds
                    to 0.475 when gradients were weak).
    _beta_raw     : ``[num_components]``, ``β_k = 0.05 + softplus(raw)``
                    per component. Floor keeps kernel decay finite.

    Derived (not learnt)
    --------------------
    α_k = b_k / kernel_sum_k(β_k, T), where
    kernel_sum_k = Σ_{s=1..T} exp(−β_k · (T + 1 − s)).

    Parameters
    ----------
    mu0_init : float | None
        Empirical baseline-intensity seed on the return-variance scale,
        typically ``float(np.mean(train_returns ** 2))``. Divided evenly
        across ``num_components`` and inverse-softplussed into
        ``_mu0_raw``. ``None`` (default) keeps the legacy zero init
        (``softplus(0) ≈ 0.69``) — rarely what you want on daily data.
    num_components : int
        Number of exponential kernel components ``K``. ``K=1`` reproduces
        the single-scale Hawkes; ``K=2`` enables the short/long
        heterogeneous-Hawkes formulation of Hardiman–Bouchaud (2013).
        Each component has independent ``(μ₀_k, b_k, β_k)``; the total
        intensity sums across components.
    struct_weight : float
        Weight on the structural match term ``(log σ² − log λ_hawkes)²``
        inside the residual. ``1.0`` (default) balances NLL vs
        structure; set to ``0.0`` to ablate the Hawkes structure and
        recover a pure GARCH-QMLE constraint (useful as a v2 ablation
        baseline).

    Inputs (``residual``)
    ---------------------
    predictions : Tensor ``[batch, 1]``
        Mean head output ``mean_pred`` on the **same scale as target**
        (normalised z-scored next-step log-return). Used in the NLL
        residual ``(target − mean_pred)²``.
    metadata : dict
        Required keys — consumed from the dict produced by
        :class:`HawkesDecoupledPINN._build_physics_metadata` plus the
        ``target`` injection performed by :meth:`CompositeLoss.forward`::

          ``returns``  : Tensor ``[batch, seq_len]`` or ``[batch, seq_len, 1]``
              Historical log-returns. Squared to form the event marks
              ``|r_s|²`` for the Hawkes kernel.
          ``log_var``  : Tensor ``[batch, 1]``
              Log-variance head output ``log σ̂²_{t+1}``. Exponentiated
              and clamped to form ``σ² = max(exp(log_var), EPS)``.
          ``target``   : Tensor ``[batch, 1]``
              Observed next-step target ``r_{t+1}`` (same scale as
              ``predictions``). Injected by ``CompositeLoss`` from its
              ``target`` argument so the constraint does not need the
              trainer-level batch contract.

    Returns
    -------
    Tensor (scalar, non-negative)
        Mean over batch of ``nll + struct_weight · struct``. Always
        finite (σ² and intensity are clamped at ``EPS`` to prevent
        ``log 0``). Units are dimensionless (log-density residual).
    """

    name = "hawkes_v2"

    def __init__(
        self,
        mu0_init: float | None = None,
        num_components: int = 1,
        struct_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_components = int(num_components)
        self.struct_weight = float(struct_weight)

        # μ₀ per component, inverse-softplus seeded so Σ softplus(_mu0_raw) ≈ mu0_init.
        if mu0_init is None:
            per_comp_raw = 0.0
        else:
            per_comp = float(mu0_init) / max(self.num_components, 1)
            per_comp_raw = math.log(math.expm1(max(per_comp, 1e-8)))
        self._mu0_raw = nn.Parameter(
            torch.full((self.num_components,), float(per_comp_raw))
        )

        # Branching ratio raw logits — small seed-varying noise ensures distinct
        # post-training values across seeds (zero init collapses all seeds to
        # sigmoid(0)·0.95 = 0.475 when gradients are weak).
        self._branching_raw = nn.Parameter(torch.randn(self.num_components) * 0.1)

        # β raw — linearly spread across components so multi-scale variants
        # start with distinct decay rates; softplus(0.5..2.0) → β ∈ ~[1, 2.1].
        if self.num_components == 1:
            beta_init = torch.tensor([1.0])
        else:
            beta_init = torch.linspace(0.5, 2.0, self.num_components)
        self._beta_raw = nn.Parameter(beta_init)

    # ── Transformed parameter views ────────────────────────────────────
    @property
    def mu0(self) -> Tensor:
        """Baseline intensity μ₀ summed across components (scalar, > 0)."""
        return F.softplus(self._mu0_raw).sum()

    @property
    def branching(self) -> Tensor:
        """Per-component branching ratios b_k ∈ (0, 0.95), ``[num_components]``."""
        return 0.95 * torch.sigmoid(self._branching_raw)

    @property
    def beta(self) -> Tensor:
        """Per-component decay rates β_k > 0.05, ``[num_components]``."""
        return 0.05 + F.softplus(self._beta_raw)

    def kernel_params_snapshot(self, T: int = 60) -> dict:
        """Post-activation kernel parameters for logging / collapse diagnostics.

        Dumps the interpretable parameters of the fitted Hawkes kernel at a
        given sequence length. Unlike ``named_parameters()`` (which exposes
        the raw pre-activation buffers ``_mu0_raw``, ``_beta_raw``,
        ``_branching_raw``), this helper evaluates the softplus / sigmoid
        transforms and additionally reports the derived excitation
        coefficients ``α_k = b_k / Σ_s exp(−β_k · (T − s))`` — which are
        NOT parameters and are therefore absent from the standard state
        dict. Primarily used to verify multi-scale kernels have
        distinguishable decay rates (no mode collapse β₁ ≈ β₂) and
        non-trivial branching after training (RQ4 audit).

        Parameters
        ----------
        T : int, default 60
            Sequence length used to derive α. α is T-dependent because
            the normalising sum ``Σ_s exp(−β·(T−s))`` depends on how
            much history the kernel integrates. Pass the training
            sequence length for interpretable values.

        Returns
        -------
        dict
            JSON-serialisable with keys:
            ``K`` (int), ``T_used`` (int), ``mu0`` (float),
            ``beta`` (list[float], length K), ``branching`` (list[float],
            length K), ``alpha`` (list[float], length K). Values are
            detached from autograd; safe to call at any time.
        """
        device = self._beta_raw.device
        dtype = self._beta_raw.dtype
        with torch.no_grad():
            _, alpha = self._kernel_and_alpha(int(T), device=device, dtype=dtype)
            return {
                "K": int(self.num_components),
                "T_used": int(T),
                "mu0": float(self.mu0.item()),
                "beta": self.beta.detach().cpu().tolist(),
                "branching": self.branching.detach().cpu().tolist(),
                "alpha": alpha.detach().cpu().tolist(),
            }

    def _kernel_and_alpha(
        self, T: int, device: torch.device, dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Compute kernel weights and derived α per component.

        Args:
            T: sequence length (number of historical steps).
            device, dtype: tensor creation options.

        Returns:
            Tuple ``(kernel, alpha)``:
              * ``kernel``: ``[K, T]`` with ``kernel[k, s] = exp(−β_k · (T − s))``
                for ``s = 0..T-1`` (so the most recent step, index T-1, has the
                smallest decay factor).
              * ``alpha``: ``[K]`` derived excitation coefficients
                ``α_k = b_k / Σ_s kernel[k, s]``. Stationarity invariant:
                ``α_k · Σ_s kernel[k, s] = b_k < 0.95``.
        """
        beta = self.beta.view(-1, 1)                                   # [K, 1]
        # Time differences T-s for s in [0..T-1] → [T, T-1, ..., 1]
        time_diffs = torch.arange(T, 0, -1, device=device, dtype=dtype)  # [T]
        kernel = torch.exp(-beta * time_diffs)                         # [K, T]
        kernel_sum = kernel.sum(dim=1).clamp_min(EPS)                  # [K]
        alpha = self.branching / kernel_sum                            # [K]
        return kernel, alpha

    def residual(self, predictions: Tensor, metadata: dict) -> Tensor:
        """Compute NLL + structural-match residual; see class docstring.

        Scale handling
        --------------
        ``predictions`` and ``metadata["target"]`` are on the z-scored
        target scale (dataset convention), whereas ``metadata["returns"]``
        is on the raw log-return scale. The Hawkes intensity derived from
        ``returns²`` is therefore on the raw variance scale ``σ²_raw``.
        For NLL and the structural match to be dimensionally consistent
        we de-normalise the predictions into raw-return units using
        ``metadata["target_mean"]`` / ``target_std"]``. The variance head
        ``log_var`` is treated as raw log-variance throughout (the head
        is unconstrained, so training shifts it to the correct scale
        under gradient pressure from both NLL and the struct term).
        """
        returns = metadata["returns"]
        if returns.dim() == 3:
            returns = returns.squeeze(-1)
        target_z = metadata["target"].view(-1, 1)      # [B, 1] (z-scored)
        log_var = metadata["log_var"].view(-1, 1)      # [B, 1]
        mean_pred_z = predictions.view(-1, 1)          # [B, 1] (z-scored)

        # De-normalise to raw log-return scale so NLL matches the scale
        # of Hawkes intensity (computed from raw returns²).
        tgt_mean = metadata.get("target_mean")
        tgt_std = metadata.get("target_std")
        if tgt_std is not None:
            std = tgt_std.view(-1, 1) if torch.is_tensor(tgt_std) else torch.tensor(float(tgt_std))
            std = std.to(target_z.device, target_z.dtype)
        else:
            std = torch.tensor(1.0, device=target_z.device, dtype=target_z.dtype)
        if tgt_mean is not None:
            mean = tgt_mean.view(-1, 1) if torch.is_tensor(tgt_mean) else torch.tensor(float(tgt_mean))
            mean = mean.to(target_z.device, target_z.dtype)
        else:
            mean = torch.tensor(0.0, device=target_z.device, dtype=target_z.dtype)

        target = target_z * std + mean                  # [B, 1] raw
        mean_pred = mean_pred_z * std + mean            # [B, 1] raw

        B, T = returns.shape
        kernel, alpha = self._kernel_and_alpha(
            T, device=returns.device, dtype=returns.dtype,
        )                                              # [K, T], [K]

        # λ_hawkes[B, 1] = μ₀ + Σ_k α_k · Σ_s kernel[k, s] · r_s²
        past_sq = (returns ** 2).unsqueeze(1)           # [B, 1, T]
        weighted = (kernel.unsqueeze(0) * past_sq       # [B, K, T]
                    * alpha.view(1, -1, 1)).sum(dim=-1) # [B, K]
        intensity = self.mu0 + weighted.sum(dim=-1, keepdim=True)  # [B, 1]
        intensity = intensity.clamp_min(EPS)

        variance = torch.exp(log_var).clamp_min(EPS)   # [B, 1]

        # (1) Gaussian QMLE NLL on raw scale — sign enters only via
        # (target − mean_pred)², which is centred on mean_pred, so the
        # mean head has a unique sign-correct optimum (no double-well).
        innov = target - mean_pred                      # [B, 1] raw
        nll = 0.5 * (innov.pow(2) / variance + torch.log(variance))

        # (2) Structural match — regularises the variance head toward the
        # Hawkes intensity computed from past squared returns with the
        # stationarity-constrained kernel. This is what makes it a
        # "Hawkes-informed" PINN rather than plain GARCH-QMLE.
        struct = (log_var - torch.log(intensity)).pow(2)

        return (nll + self.struct_weight * struct).mean()
