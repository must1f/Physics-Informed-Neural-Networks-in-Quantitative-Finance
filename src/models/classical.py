"""Classical econometric baseline models for financial return prediction.

These models are NOT nn.Module subclasses — they bypass the neural training
loop and are dispatched by _run_classical_experiment in runner.py.

Model summary:
    random_walk     pred_t = 0                   log-return scale (EMH lower bound)
    historical_mean pred_t = mean(train_returns)  log-return scale
    persistence     pred_t = actual_{t-1}         log-return scale (carry-forward)
    garch           pred_t = mu from GARCH(1,1)   log-return scale (MLE constant mean)
"""
from __future__ import annotations

import numpy as np

_CLASSICAL_NAMES: frozenset[str] = frozenset({
    "random_walk", "historical_mean", "persistence", "garch",
})


def is_classical(model_name: str) -> bool:
    """Return True if *model_name* maps to a classical (non-neural) baseline.

    Args:
        model_name: any string (typically a registry key).

    Returns:
        ``True`` iff *model_name* is one of
        ``{"random_walk", "historical_mean", "persistence", "garch"}``.
    """
    return model_name in _CLASSICAL_NAMES


class RandomWalkModel:
    """EMH lower-bound baseline: predict zero log-return for every step.

    Under the random-walk hypothesis, E[r_t | F_{t-1}] = 0, so the best
    forecast is that tomorrow's price equals today's (zero expected change).

    Learnable parameters: none.
    """

    def fit(self, train_returns: np.ndarray) -> None:
        """No-op: random walk has no learnable parameters.

        Args:
            train_returns: 1-D ndarray of training log-returns, shape ``(N_train,)``.
                Not used — exists only to satisfy the common fit/predict interface.
        """

    def predict(self, test_returns: np.ndarray, last_train_return: float) -> np.ndarray:
        """Predict zero log-return for every test step.

        Args:
            test_returns: 1-D ndarray of realised test log-returns,
                shape ``(N_test,)``. Used only to determine output length.
            last_train_return: last realised training return (unused).

        Returns:
            ``np.zeros(N_test)`` on the log-return scale, shape ``(N_test,)``.
        """
        return np.zeros(len(test_returns))


class HistoricalMeanModel:
    """Unconditional-mean benchmark: predict the training-set mean return.

    Learnable parameters:
        ``_mean`` (float): OLS estimate of E[r], unconstrained, on the
        log-return scale.
    """

    _mean: float = 0.0

    def fit(self, train_returns: np.ndarray) -> None:
        """Compute and store the unconditional mean of *train_returns*.

        Args:
            train_returns: 1-D ndarray of training log-returns,
                shape ``(N_train,)``, log-return scale.
        """
        self._mean = float(train_returns.mean())

    def predict(self, test_returns: np.ndarray, last_train_return: float) -> np.ndarray:
        """Return the constant unconditional mean for all test steps.

        Args:
            test_returns: 1-D ndarray of realised test log-returns,
                shape ``(N_test,)``. Used only to determine output length.
            last_train_return: last realised training return (unused).

        Returns:
            ``np.full(N_test, _mean)`` on the log-return scale,
            shape ``(N_test,)``.
        """
        return np.full(len(test_returns), self._mean)


class PersistenceModel:
    """Autocorrelation benchmark: pred_t = actual_{t-1} (daily carry-forward).

    The first test prediction uses the last training return; thereafter each
    prediction is the previous period's realised return (ex-post oracle on the
    test window — acceptable since persistence is a baseline, not a strategy).

    Learnable parameters:
        ``_last_train`` (float): last value of ``train_returns``, stored by
        :meth:`fit`; seeds the carry-forward at test step 0.
    """

    _last_train: float = 0.0

    def fit(self, train_returns: np.ndarray) -> None:
        """Store the final training return as the carry-forward seed.

        Args:
            train_returns: 1-D ndarray of training log-returns,
                shape ``(N_train,)``, log-return scale.
        """
        self._last_train = float(train_returns[-1])

    def predict(self, test_returns: np.ndarray, last_train_return: float) -> np.ndarray:
        """Carry-forward: pred_t = actual_{t-1}; pred_0 = last training return.

        Args:
            test_returns: 1-D ndarray of realised test log-returns,
                shape ``(N_test,)``, log-return scale.
            last_train_return: last realised training return (unused —
                :meth:`fit` already stored it in ``_last_train``).

        Returns:
            1-D ndarray ``[_last_train, r_0, r_1, …, r_{N-2}]`` on the
            log-return scale, shape ``(N_test,)``.
        """
        return np.concatenate([[self._last_train], test_returns[:-1]])


class GARCHModel:
    """GARCH(1,1) conditional-volatility benchmark via the ``arch`` package.

    Fits a constant-mean GARCH(1,1) on training log-returns using MLE.
    The constant conditional mean ``mu`` provides next-step return predictions.
    Falls back to the historical mean when ``arch`` is unavailable or MLE fails.

    Learnable parameters (MLE via ``arch``):
        ``_mu`` (float): constant conditional-mean term, log-return scale.
        ``_omega``, ``_alpha``, ``_beta`` (float | None): GARCH variance
            parameters on the native log-return scale (``_omega`` is rescaled
            from the arch ``×100`` convention by dividing by ``100²``).
        ``_loglik`` (float | None): log-likelihood at the MLE optimum.
        ``_persistence`` (float | None): ``alpha + beta``; stationarity
            requires this to be < 1. All variance parameters are ``None``
            when ``_fit_ok`` is ``False``.

    Prediction-recursion state (populated by :meth:`fit`):
        ``_train_var`` (float): unconditional training variance — fallback
            when ``arch`` is unavailable.
        ``_last_sigma2`` (float | None): conditional variance at the last
            training observation; seeds σ²₀ of :meth:`predict_volatility`.
        ``_last_eps`` (float | None): last training residual r_T − μ;
            seeds ε₀ of :meth:`predict_volatility`.

    Public reporting API:
        :meth:`params` returns a JSON-serialisable dict with every fitted
        parameter so runners can persist it as a sidecar next to metrics.
    """

    _mean: float = 0.0
    _mu: float = 0.0
    _omega: float | None = None
    _alpha: float | None = None
    _beta: float | None = None
    _loglik: float | None = None
    _persistence: float | None = None
    _fit_ok: bool = False
    _train_var: float = 0.0
    _last_sigma2: float | None = None
    _last_eps: float | None = None

    def fit(self, train_returns: np.ndarray) -> None:
        """Fit constant-mean GARCH(1,1) on *train_returns*; fall back to mean on failure.

        On success, stores all MLE parameters (mu, omega, alpha, beta,
        loglik, persistence) and the recursion-seed state (_last_sigma2,
        _last_eps). omega is rescaled from the ``arch`` convention
        (returns ×100) back to native log-return scale by dividing by 100².

        Args:
            train_returns: 1-D ndarray of training log-returns,
                shape ``(N_train,)``, log-return scale. Multiplied by 100
                internally before passing to ``arch_model`` (arch convention).
        """
        self._mean = float(train_returns.mean())
        self._mu = self._mean
        self._train_var = float(np.var(train_returns, ddof=1)) if len(train_returns) > 1 else 0.0
        try:
            from arch import arch_model  # type: ignore[import]
            am = arch_model(
                train_returns * 100,
                mean="Constant", vol="Garch", p=1, q=1,
            )
            res = am.fit(disp="off", show_warning=False)
            p = res.params
            self._mu = float(p.get("mu", self._mean * 100)) / 100.0
            # omega lives on the (returns * 100)² scale; bring it back to
            # the native log-return variance scale.
            omega_scaled = p.get("omega", None)
            self._omega = float(omega_scaled) / (100.0 ** 2) if omega_scaled is not None else None
            self._alpha = float(p.get("alpha[1]")) if "alpha[1]" in p else None
            self._beta = float(p.get("beta[1]")) if "beta[1]" in p else None
            self._loglik = float(getattr(res, "loglikelihood", float("nan")))
            if self._alpha is not None and self._beta is not None:
                self._persistence = self._alpha + self._beta
            self._fit_ok = True
            # Seeds for predict_volatility: last training conditional variance
            # and last residual on the log-return scale.
            cond_vol_scaled = np.asarray(res.conditional_volatility)
            # arch returns conditional volatility on the (returns * 100) scale;
            # bring back to native log-return scale by dividing by 100.
            cond_sigma = cond_vol_scaled / 100.0
            self._last_sigma2 = float(cond_sigma[-1] ** 2)
            self._last_eps = float(train_returns[-1] - self._mu)
            self._train_var = float(np.var(train_returns, ddof=1))
        except Exception:
            self._fit_ok = False

    def params(self) -> dict[str, float | bool | None]:
        """Return fitted GARCH(1,1) parameters as a JSON-serialisable dict.

        Intended for dissertation-appendix reporting and as the seed
        state for :meth:`predict_volatility`. Keys:
            fit_ok: whether MLE converged (False means the fallback
                historical-mean path was used; variance params are None).
            mu, omega, alpha, beta: MLE parameter estimates on the
                native log-return scale.
            persistence: alpha + beta (< 1 required for stationarity).
            loglik: log-likelihood at the MLE.
            last_sigma2: conditional variance at the last training
                observation — seeds σ²_0 of predict_volatility (None
                when fit_ok is False).
            last_eps: last training residual r_T − μ — seeds ε_0 of
                predict_volatility (None when fit_ok is False).
            train_var: unconditional training variance — used as the
                flat fallback when fit_ok is False.

        Returns:
            dict[str, float|bool|None] with exactly the ten keys above.
        """
        return {
            "fit_ok": self._fit_ok,
            "mu": self._mu,
            "omega": self._omega,
            "alpha": self._alpha,
            "beta": self._beta,
            "persistence": self._persistence,
            "loglik": self._loglik,
            "last_sigma2": self._last_sigma2,
            "last_eps": self._last_eps,
            "train_var": self._train_var,
        }

    def predict_volatility(
        self,
        test_returns: np.ndarray,
        last_train_return: float,
    ) -> np.ndarray:
        """One-step-ahead GARCH(1,1) conditional σ_t on the test window.

        Applies the walk-forward recursion
        ``σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}``
        where ``ε_{t-1} = r_{t-1} - μ`` uses the realised test return at
        t-1 (expanding-information one-step-ahead forecasting, not
        multi-step projection). The recursion is seeded from
        ``_last_sigma2`` and the provided ``last_train_return``, so no
        look-ahead bias is introduced.

        Falls back to the unconditional ``sqrt(_train_var)`` flat
        forecast when ``_fit_ok is False`` (arch missing or MLE failed),
        matching the behaviour of :meth:`predict` which falls back to
        the training mean in the same situation.

        Args:
            test_returns: 1-D ndarray of realised test log-returns,
                shape (N_test,). Used only as the r_{t-1} input to the
                variance recursion — NOT as the forecast target.
            last_train_return: Scalar, last realised training return.
                Seeds ε_0 = last_train_return - μ.

        Returns:
            1-D ndarray of σ_t forecasts on the log-return scale, shape
            (N_test,), strictly positive. When ``_fit_ok is False``
            returns a flat array at ``sqrt(_train_var)``.

        Learnable parameters used:
            μ, ω, α, β — stored on the object by :meth:`fit`.
        """
        test_returns = np.asarray(test_returns, dtype=float)
        n = len(test_returns)
        if not self._fit_ok or self._omega is None:
            fallback_sigma = float(np.sqrt(self._train_var)) if self._train_var > 0 else 1e-6
            return np.full(n, fallback_sigma)
        omega = float(self._omega)
        alpha = float(self._alpha)
        beta = float(self._beta)
        mu = float(self._mu)
        sigma2 = np.empty(n, dtype=float)
        eps_prev = float(last_train_return - mu)
        sigma2_prev = float(self._last_sigma2) if self._last_sigma2 is not None else self._train_var
        for t in range(n):
            sigma2[t] = omega + alpha * eps_prev ** 2 + beta * sigma2_prev
            eps_prev = float(test_returns[t] - mu)
            sigma2_prev = sigma2[t]
        return np.sqrt(sigma2)

    def predict(self, test_returns: np.ndarray, last_train_return: float) -> np.ndarray:
        """Constant conditional-mean forecast from GARCH MLE.

        For a constant-mean GARCH(1,1), E[r_t | F_{t-1}] = μ for all t.
        Falls back to the training-set historical mean when ``arch`` is
        unavailable or MLE failed (``_fit_ok is False``).

        Args:
            test_returns: 1-D ndarray of realised test log-returns,
                shape ``(N_test,)``. Used only to determine output length.
            last_train_return: last realised training return (unused for
                the constant-mean model).

        Returns:
            ``np.full(N_test, mu)`` on the log-return scale,
            shape ``(N_test,)``.
        """
        mu = self._mu if self._fit_ok else self._mean
        return np.full(len(test_returns), mu)
