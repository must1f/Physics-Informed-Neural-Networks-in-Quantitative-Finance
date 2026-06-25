"""Formal significance tests for forecast comparison and directional skill.

Two pure-function tests used to attach formal p-values to the classical
and neural baseline rankings:

* :func:`diebold_mariano` — Diebold & Mariano (1995) test of equal
  predictive accuracy between two forecast series.
* :func:`pesaran_timmermann` — Pesaran & Timmermann (1992) test of
  directional-accuracy independence from chance.

Both functions operate on NumPy arrays of equal length, return a
standard ``(statistic, p_value)`` pair, and make no assumptions beyond
what the canonical papers require.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    loss: str = "se",
    h: int = 1,
    hln_correction: bool = True,
) -> tuple[float, float]:
    """Diebold–Mariano test of equal predictive accuracy.

    Tests H0: ``E[L(e_a_t)] == E[L(e_b_t)]`` against the two-sided
    alternative, using the standard DM statistic
    ``DM = d_bar / sqrt(V_hat / T)`` where ``d_t = L(e_a_t) - L(e_b_t)``,
    ``V_hat`` is the Newey–West HAC estimator of the long-run variance of
    ``d_t`` with bandwidth ``h - 1``, and ``T`` is the series length. For
    short samples applies the Harvey–Leybourne–Newbold (1997) finite-sample
    correction: scale by ``sqrt((T + 1 - 2h + h(h-1)/T) / T)`` and refer
    to a Student-t(T-1) distribution instead of the normal.

    Sign convention: a **negative** DM statistic means model A (first arg)
    has lower loss, i.e. forecasts better. Interpret the p-value against
    your chosen alpha; a significant result lets you reject equal accuracy.

    Args:
        errors_a: 1-D array of forecast errors from model A, shape (T,),
            in the same units as ``errors_b`` (typically raw residuals
            ``actual - pred`` on the log-return scale).
        errors_b: 1-D array of forecast errors from model B, shape (T,).
            Must have the same length as ``errors_a``.
        loss: Loss transform applied to each error before differencing.
            ``"se"`` squared error (default — matches RMSE comparisons),
            ``"ae"`` absolute error (matches MAE comparisons).
        h: Forecast horizon in periods. 1 for one-step-ahead (the classical
            baselines are all h=1). For h > 1 the long-run variance uses
            lags 0..h-1 per Diebold & Mariano (1995).
        hln_correction: If True (default) apply the HLN small-sample
            correction and use a Student-t reference distribution;
            otherwise use the asymptotic normal.

    Returns:
        ``(dm_stat, p_value)`` tuple of Python floats. ``p_value`` is
        two-sided. Returns ``(0.0, 1.0)`` if the loss-differential series
        is identically zero (two models produce identical forecasts).

    Raises:
        ValueError: If inputs are not 1-D, lengths differ, ``h < 1``, or
            ``loss`` is not in ``{"se", "ae"}``.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.ndim != 1 or errors_b.ndim != 1:
        raise ValueError("errors_a and errors_b must both be 1-D arrays")
    if errors_a.shape != errors_b.shape:
        raise ValueError(
            f"Shape mismatch: errors_a {errors_a.shape} vs errors_b {errors_b.shape}"
        )
    if h < 1:
        raise ValueError(f"Forecast horizon h must be >= 1, got {h}")

    if loss == "se":
        losses = (errors_a ** 2, errors_b ** 2)
    elif loss == "ae":
        losses = (np.abs(errors_a), np.abs(errors_b))
    else:
        raise ValueError(f"loss must be 'se' or 'ae', got {loss!r}")

    d = losses[0] - losses[1]
    T = d.size
    d_bar = float(d.mean())

    # Identical forecasts → no statistic; report a neutral p = 1.
    if np.allclose(d, 0.0):
        return 0.0, 1.0

    # Long-run variance via Newey–West HAC with bandwidth h - 1.
    gamma0 = float(np.var(d, ddof=0))
    var_lr = gamma0
    for k in range(1, h):
        gamma_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        var_lr += 2.0 * gamma_k

    if var_lr <= 0.0:
        return 0.0, 1.0  # degenerate — treat as inconclusive

    dm_stat = d_bar / np.sqrt(var_lr / T)

    if hln_correction:
        factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
        dm_stat *= factor
        p_value = 2.0 * float(stats.t.sf(abs(dm_stat), df=T - 1))
    else:
        p_value = 2.0 * float(stats.norm.sf(abs(dm_stat)))

    return float(dm_stat), p_value


def dm_test(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    actual: np.ndarray,
    *,
    h: int = 1,
) -> tuple[float, float]:
    """Convenience wrapper: Diebold–Mariano test taking predictions, not errors.

    Computes residuals ``e = actual - pred`` for each model then calls
    :func:`diebold_mariano` with ``loss="se"`` and HLN correction. At h=1 the
    statistic is algebraically identical to the plain-normal asymptotic form
    ``d_bar / sqrt(var(d, ddof=1) / n)``; the only difference is a more
    conservative t(T-1) tail (more rigorous for finite samples).

    Sign convention: positive DM statistic means model A has **larger** MSE
    (model B forecasts better). Returns ``(nan, nan)`` when degenerate.

    Args:
        pred_a: 1-D array of model-A point forecasts, shape (T,), log-return scale.
        pred_b: 1-D array of model-B point forecasts, shape (T,), same scale.
        actual: 1-D array of realised log-returns aligned with both forecasts, shape (T,).
        h: Forecast horizon in periods (default 1 for one-step-ahead).

    Returns:
        ``(dm_stat, p_value)`` tuple of floats. Returns ``(nan, nan)`` when the
        test is degenerate (n < 2, identical forecasts, zero-variance differential).
    """
    pred_a_ = np.asarray(pred_a, dtype=float)
    pred_b_ = np.asarray(pred_b, dtype=float)
    actual_ = np.asarray(actual, dtype=float)
    if pred_a_.shape != actual_.shape or pred_b_.shape != actual_.shape:
        return float("nan"), float("nan")
    ea = actual_ - pred_a_
    eb = actual_ - pred_b_
    if len(ea) < 2:
        return float("nan"), float("nan")
    stat, p = diebold_mariano(ea, eb, loss="se", h=h)
    # diebold_mariano returns (0.0, 1.0) for degenerate identical-loss series.
    if stat == 0.0 and p == 1.0 and np.allclose(ea ** 2, eb ** 2):
        return float("nan"), float("nan")
    return stat, p


def sign_disagreement(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Fraction of timesteps where sign(pred_a) ≠ sign(pred_b).

    Args:
        pred_a: 1-D array of model-A forecasts, shape (T,), log-return scale.
        pred_b: 1-D array of model-B forecasts, shape (T,), same scale.

    Returns:
        Float in [0, 1]. 0 means the two models agree on direction everywhere.
    """
    sa = np.sign(np.asarray(pred_a, dtype=float).ravel())
    sb = np.sign(np.asarray(pred_b, dtype=float).ravel())
    if sa.shape != sb.shape:
        return float("nan")
    return float((sa != sb).mean())


def pesaran_timmermann(
    pred: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float]:
    """Pesaran–Timmermann (1992) test of directional predictive accuracy.

    Tests H0: ``sign(pred)`` and ``sign(actual)`` are **independent**
    (i.e. the model has no directional skill) against the one-sided
    alternative of positive dependence. The statistic

    .. math::
        S_n = \\frac{P^* - \\hat P}{\\sqrt{\\widehat{\\text{Var}}(P^*) -
                                           \\widehat{\\text{Var}}(\\hat P)}}

    is asymptotically standard normal under H0, where ``P*`` is the
    observed hit rate and ``P̂`` is the hit rate implied by independence
    of the marginal sign distributions.

    A **positive** statistic with small p-value means the model's
    directional forecasts are significantly better than independence —
    the canonical "does the model predict direction?" test in the
    financial-forecasting literature.

    Ties: ``np.sign(0)`` is mapped to +1 before the test, because the
    original paper assumes a binary up/down classification. Pass
    strictly-nonzero returns if the zero class matters; the classical
    ``random_walk`` baseline has sign(pred)==0 for every step and will
    therefore show P̂ == P(actual>0) under this convention.

    Args:
        pred: 1-D array of predicted values (signs are what matter),
            shape (N,). Typically log-return predictions; any scale is
            fine because only the sign is used.
        actual: 1-D array of realised values aligned with ``pred``,
            shape (N,). Same scale convention as ``pred``.

    Returns:
        ``(pt_stat, p_value)`` tuple of Python floats. ``p_value`` is
        one-sided (upper tail) since the economic interpretation is
        "does the model beat independence?". Returns ``(0.0, 0.5)`` if
        either predicted or actual signs are degenerate (constant).

    Raises:
        ValueError: If inputs are not 1-D or shapes differ.
    """
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if pred.ndim != 1 or actual.ndim != 1:
        raise ValueError("pred and actual must both be 1-D arrays")
    if pred.shape != actual.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs actual {actual.shape}"
        )

    # Binary up/down indicators — ties → +1 by PT convention.
    x = (np.sign(pred) >= 0).astype(float)   # 1 if pred >= 0
    y = (np.sign(actual) >= 0).astype(float) # 1 if actual >= 0
    n = x.size
    if n == 0:
        return 0.0, 0.5

    px = x.mean()
    py = y.mean()
    # Degenerate marginals (all up or all down on one side) → test
    # undefined; return neutral p.
    if min(px, py, 1 - px, 1 - py) == 0.0:
        return 0.0, 0.5

    p_star = float((x == y).mean())
    p_hat = px * py + (1 - px) * (1 - py)

    var_p_star = p_star * (1 - p_star) / n
    var_p_hat = (
        (2 * py - 1) ** 2 * px * (1 - px) / n
        + (2 * px - 1) ** 2 * py * (1 - py) / n
        + 4 * px * py * (1 - px) * (1 - py) / (n ** 2)
    )

    denom_var = var_p_star - var_p_hat
    if denom_var <= 0.0:
        return 0.0, 0.5

    pt_stat = (p_star - p_hat) / np.sqrt(denom_var)
    p_value = float(stats.norm.sf(pt_stat))  # one-sided upper tail

    return float(pt_stat), p_value
