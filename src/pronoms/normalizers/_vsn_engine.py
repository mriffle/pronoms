"""
Native Python implementation of the VSN (Variance Stabilizing Normalization)
algorithm.

This module contains no public API. It is consumed by ``VSNNormalizer``.
The numerical contract is to match the R Bioconductor ``vsn`` package
(Huber et al., 2002) for the ``calib="affine"`` / unstratified / non-reference
case, which is the only mode currently exposed through the pronoms
``VSNNormalizer`` interface.

Algorithm (mirrors R sources ``vsn2.c``/``vsn2.R`` in tmp/vsn):

1. Model: ``h_ij = arsinh(exp(b_j) * y_ij + a_j)`` where ``y_ij`` is the
   raw intensity for feature *i* and sample *j*. Parameters ``a_j`` (offset)
   and ``b_j = exp(beta_j)`` (positive scaling factor) are per-sample.
2. Negative profile log-likelihood (sigma^2 and per-feature mu profiled out
   by the envelope theorem) is optimized with L-BFGS-B.
3. Robust LTS iteration: refit on the rows whose post-transform residual
   variance is below the ``lts_quantile`` quantile, computed within five
   intensity slices. Up to ``cvg_niter`` iterations.
4. Final transform: ``h_ij_log2 = arsinh(...)/ln(2) - hoffset``, where
   ``hoffset = log2(2 * exp(mean(beta_j)))``.

Internal arrays use the R orientation (n_features rows by n_samples cols).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class VSNFit:
    """Fitted VSN parameters (single, unstratified affine fit).

    Attributes
    ----------
    a : np.ndarray
        Per-sample offset, shape ``(n_samples,)``.
    b_log : np.ndarray
        Per-sample log-scaling-factor, shape ``(n_samples,)``. The actual
        scaling factor is ``b = exp(b_log)``.
    mu : np.ndarray
        Profiled per-feature mean of the transformed data on the natural
        ``arsinh`` scale, shape ``(n_features,)``. Entries that were trimmed
        out by the LTS step in iterations after the first are ``NaN`` --
        matching R's behavior.
    sigsq : float
        Profiled error variance on the natural ``arsinh`` scale.
    hoffset : float
        Per-stratum offset applied during transform to put output on a
        log2-like scale. For unstratified fits this is a single scalar.
    coefficients : np.ndarray
        Three-dimensional array shaped ``(1, n_samples, 2)`` matching
        R-VSN's ``vsn@coefficients`` convention; ``[0, j, 0]`` is ``a_j``
        and ``[0, j, 1]`` is ``b_log_j`` (i.e. ``log(b_j)``).
    converged : bool
        Whether the inner L-BFGS-B converged on the final LTS iteration.
    n_lts_iter : int
        Number of LTS iterations actually executed.
    """

    a: np.ndarray
    b_log: np.ndarray
    mu: np.ndarray
    sigsq: float
    hoffset: float
    coefficients: np.ndarray
    converged: bool
    n_lts_iter: int


# ---------------------------------------------------------------------------
# Negative profile log-likelihood and gradient (vectorized).
# ---------------------------------------------------------------------------


def _objective(theta: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """Profile NLL and gradient for unstratified affine VSN.

    Parameters
    ----------
    theta : np.ndarray, shape (2 * n_samples,)
        Concatenation of ``[a_0, ..., a_{n-1}, b_log_0, ..., b_log_{n-1}]``.
    Y : np.ndarray, shape (n_features, n_samples)
        Raw input data in R orientation.

    Returns
    -------
    nll : float
        Negative profile log-likelihood, matching the R-VSN convention.
    grad : np.ndarray, shape (2 * n_samples,)
        Gradient of ``nll`` w.r.t. ``theta``.
    """
    n_features, n_samples = Y.shape
    a = theta[:n_samples]
    b_log = theta[n_samples:]
    b = np.exp(b_log)

    Y_lin = b[np.newaxis, :] * Y + a[np.newaxis, :]
    h = np.arcsinh(Y_lin)
    mu = h.mean(axis=1)
    r = h - mu[:, np.newaxis]
    ssq = float(np.sum(r * r))
    nt = n_features * n_samples
    sigsq = ssq / nt

    jac1 = float(np.sum(np.log1p(Y_lin * Y_lin)))
    jac2 = float(n_features * b_log.sum())
    scale = 0.5 * nt * np.log(2.0 * np.pi * sigsq)
    nll = scale + 0.5 * nt + 0.5 * jac1 - jac2

    rfac = 1.0 / sigsq
    A = 1.0 / np.sqrt(1.0 + Y_lin * Y_lin)
    z = r * rfac + A * Y_lin
    gr_a = np.sum(z * A, axis=0)
    gr_b = b * np.sum(z * A * Y, axis=0) - n_features
    grad = np.concatenate([gr_a, gr_b])
    return float(nll), grad


# ---------------------------------------------------------------------------
# LTS subset selection.
# ---------------------------------------------------------------------------


def _lts_select(
    hy: np.ndarray,
    lts_quantile: float,
    n_slices: int = 5,
) -> np.ndarray:
    """Select rows whose residual variance is in the lower ``lts_quantile``.

    Mirrors the slicing logic in R's ``vsnLTS`` (vsn2.R:148-163): rank
    features by ``hmean``, cut into ``n_slices`` equal-width slices over
    the rank range, compute the ``lts_quantile`` quantile of per-row
    residual variance within each slice, and keep rows below the threshold
    OR in the lowest-intensity slice (slice 1).

    Parameters
    ----------
    hy : np.ndarray, shape (n_features, n_samples)
        Transformed values ``arsinh(...)`` on the natural scale.
    lts_quantile : float
    n_slices : int

    Returns
    -------
    np.ndarray of bool, shape (n_features,)
    """
    n_features = hy.shape[0]
    hmean = hy.mean(axis=1)
    rvar = np.sum((hy - hmean[:, np.newaxis]) ** 2, axis=1)

    # Rank with average-of-ties (R's default for rank()).
    order = np.argsort(hmean, kind="mergesort")
    ranks = np.empty(n_features, dtype=np.float64)
    ranks[order] = np.arange(1, n_features + 1, dtype=np.float64)
    # Replace ties with their average rank to match R's default.
    sorted_h = hmean[order]
    i = 0
    while i < n_features:
        j = i + 1
        while j < n_features and sorted_h[j] == sorted_h[i]:
            j += 1
        if j > i + 1:
            avg = 0.5 * (i + j + 1)  # mean of ranks i+1..j
            ranks[order[i:j]] = avg
        i = j

    rmin = float(ranks.min())
    rmax = float(ranks.max())
    if rmax > rmin:
        # Mirror R's cut(x, breaks=n): equal-width breaks over [min, max] with
        # only the outermost two pushed out by 0.1% so endpoints fall inside
        # the leftmost / rightmost interval. Internal cut points stay exact.
        edges = np.linspace(rmin, rmax, n_slices + 1)
        slice_idx = np.searchsorted(edges[1:-1], ranks, side="left")  # 0..n_slices-1
    else:
        slice_idx = np.zeros(n_features, dtype=np.int64)

    selected = slice_idx == 0  # always-include lowest slice (R: slice == 1)
    for s in range(1, n_slices):
        mask = slice_idx == s
        if mask.any():
            threshold = np.quantile(rvar[mask], lts_quantile)
            selected |= mask & (rvar <= threshold)
    return selected


# ---------------------------------------------------------------------------
# L-BFGS-B wrapper for one ML fit.
# ---------------------------------------------------------------------------

# R-VSN defaults: factr=5e7, pgtol=2e-4, maxit=60000, m=5.  scipy's L-BFGS-B
# uses ftol = factr * eps, gtol = pgtol, maxcor = m.  Mirroring these gives
# parameter estimates that match R within optimizer noise.  The VSN
# profile likelihood has near-flat directions on small noisy inputs (any
# uniform-across-samples additive shift in h_ij gets absorbed by the
# profiled per-feature mean mu_i), so identical parameters are not
# achievable across L-BFGS-B implementations -- but the *transformed
# output* still agrees tightly on realistic-density datasets where data
# constrain the model well.
_FTOL_DEFAULT = 5e7 * np.finfo(np.float64).eps
_GTOL_DEFAULT = 2e-4
_MAXITER_DEFAULT = 60000
_MAXCOR_DEFAULT = 5  # match R's lbfgsb default memory size


def _ml_fit(
    Y: np.ndarray,
    pstart: np.ndarray,
    *,
    ftol: float = float(_FTOL_DEFAULT),
    gtol: float = _GTOL_DEFAULT,
    maxiter: int = _MAXITER_DEFAULT,
) -> tuple[np.ndarray, bool]:
    """Run a single ML fit on ``Y`` starting from ``pstart``.

    Returns
    -------
    theta : np.ndarray
        Optimized parameters.
    converged : bool
        True iff scipy's L-BFGS-B reported success.
    """
    n_samples = Y.shape[1]
    bounds = [(None, None)] * n_samples + [(-100.0, 100.0)] * n_samples

    res = minimize(
        _objective,
        pstart,
        args=(Y,),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "ftol": ftol,
            "gtol": gtol,
            "maxiter": maxiter,
            "maxfun": maxiter,
            "maxcor": _MAXCOR_DEFAULT,
        },
    )
    return res.x.copy(), bool(res.success)


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def fit_vsn(
    X: np.ndarray,
    *,
    lts_quantile: float = 0.9,
    cvg_niter: int = 7,
    ftol: float = float(_FTOL_DEFAULT),
    gtol: float = _GTOL_DEFAULT,
    maxiter: int = _MAXITER_DEFAULT,
) -> VSNFit:
    """Fit unstratified affine VSN on ``X``.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        Raw intensity matrix in R orientation. ``VSNNormalizer`` transposes
        from its (n_samples, n_features) public API before calling here.
    lts_quantile : float, default 0.9
        Quantile for the LTS robust-fit step. ``1.0`` disables LTS.
    cvg_niter : int, default 7
        Maximum number of LTS iterations.

    Returns
    -------
    VSNFit
    """
    if X.ndim != 2:
        raise ValueError(f"VSN expects a 2D matrix; got shape {X.shape}.")
    n_features, n_samples = X.shape
    if n_samples < 2:
        raise ValueError("VSN needs at least 2 samples (columns) when no reference is supplied.")
    if n_features < 2:
        raise ValueError("VSN needs at least 2 features (rows) for the per-feature mean to have meaning.")
    if not (0.0 < lts_quantile <= 1.0):
        raise ValueError("lts_quantile must lie in (0, 1].")

    # Initial parameters: a = 0, b_log = 1 (matches pstartHeuristic in R).
    pstart = np.empty(2 * n_samples, dtype=np.float64)
    pstart[:n_samples] = 0.0
    pstart[n_samples:] = 1.0

    whsel = np.ones(n_features, dtype=bool)
    no_lts = abs(lts_quantile - 1.0) < np.sqrt(np.finfo(np.float64).eps)

    converged = False
    iters_run = 0
    theta = pstart.copy()
    for iter_idx in range(cvg_niter):
        iters_run = iter_idx + 1
        Y_sub = X if iter_idx == 0 else X[whsel]
        theta, converged = _ml_fit(
            Y_sub,
            pstart,
            ftol=ftol,
            gtol=gtol,
            maxiter=maxiter,
        )
        if no_lts:
            break

        a_fit = theta[:n_samples]
        b_log_fit = theta[n_samples:]
        b_fit = np.exp(b_log_fit)
        Y_lin = b_fit[np.newaxis, :] * X + a_fit[np.newaxis, :]
        hy_full = np.arcsinh(Y_lin)

        # Warm start the next iteration from the current parameters.
        pstart = theta.copy()

        # Recompute LTS subset.
        new_whsel = _lts_select(hy_full, lts_quantile)
        # Match R's behavior: if the subset stops changing exactly we could
        # break, but R doesn't (cvg.eps default = 0). Keep iterating.
        whsel = new_whsel

    a_fit = theta[:n_samples]
    b_log_fit = theta[n_samples:]
    b_fit = np.exp(b_log_fit)

    # Compute final mu / sigsq on the natural arsinh scale, restricted to the
    # rows actually used in the last fit (matches R's bookkeeping).
    Y_used = X if iters_run == 1 else X[whsel]
    Y_lin_used = b_fit[np.newaxis, :] * Y_used + a_fit[np.newaxis, :]
    h_used = np.arcsinh(Y_lin_used)
    mu_used = h_used.mean(axis=1)
    sigsq = float(np.mean((h_used - mu_used[:, np.newaxis]) ** 2))

    if iters_run == 1:
        mu_full = mu_used
    else:
        mu_full = np.full(n_features, np.nan, dtype=np.float64)
        mu_full[whsel] = mu_used

    hoffset = float(np.log2(2.0 * np.exp(b_log_fit.mean())))
    coef = np.empty((1, n_samples, 2), dtype=np.float64)
    coef[0, :, 0] = a_fit
    coef[0, :, 1] = b_log_fit

    return VSNFit(
        a=a_fit,
        b_log=b_log_fit,
        mu=mu_full,
        sigsq=sigsq,
        hoffset=hoffset,
        coefficients=coef,
        converged=converged,
        n_lts_iter=iters_run,
    )


def transform(X: np.ndarray, fit: VSNFit) -> np.ndarray:
    """Apply a fitted VSN model to ``X`` in R orientation.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
    fit : VSNFit

    Returns
    -------
    np.ndarray, shape (n_features, n_samples)
        Transformed matrix on the log2-like scale, with ``hoffset`` applied
        so the output is comparable to ``log2(intensity)`` for large
        positive intensities.
    """
    if X.ndim != 2:
        raise ValueError(f"VSN expects a 2D matrix; got shape {X.shape}.")
    if X.shape[1] != fit.a.shape[0]:
        raise ValueError(f"Sample count mismatch: fit has {fit.a.shape[0]} samples, data has {X.shape[1]}.")
    b = np.exp(fit.b_log)
    Y_lin = b[np.newaxis, :] * X + fit.a[np.newaxis, :]
    h_natural = np.arcsinh(Y_lin)
    return h_natural / np.log(2.0) - fit.hoffset


def fit_transform(
    X: np.ndarray,
    *,
    lts_quantile: float = 0.9,
    cvg_niter: int = 7,
) -> tuple[np.ndarray, VSNFit]:
    """Fit VSN on ``X`` and return the transformed matrix together with the
    fitted parameters.

    Parameters
    ----------
    X : np.ndarray, shape (n_features, n_samples)
        Raw input in R orientation.
    """
    fit = fit_vsn(X, lts_quantile=lts_quantile, cvg_niter=cvg_niter)
    return transform(X, fit), fit
