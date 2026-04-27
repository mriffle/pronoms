"""
VSN (Variance Stabilizing Normalization) Normalizer for proteomics data.

This module provides ``VSNNormalizer``, a pure-Python implementation of
Huber et al.'s VSN algorithm. Earlier releases of pronoms used the
Bioconductor ``vsn`` R package via ``rpy2``; that dependency has been
replaced by a vectorized NumPy/SciPy implementation in
``pronoms.normalizers._vsn_engine``. The public class signature is
preserved.
"""

from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import plot_comparison_hexbin
from ..utils.validators import check_nan_inf, validate_input_data
from . import _vsn_engine


class VSNNormalizer:
    """Variance-stabilizing normalization (Huber et al., 2002).

    The fitted model is ``h_ij = arsinh(exp(beta_j) * y_ij + a_j)``, which
    behaves like ``log(y_ij)`` for large ``y_ij`` and is well-defined for
    zero or negative inputs. Per-sample offset ``a_j`` and log-scale
    factor ``beta_j`` are estimated by maximum profile likelihood with a
    least-trimmed-squares robustification step. The output is on a
    log2-comparable scale.

    Parameters
    ----------
    calib : str, optional
        Calibration method, by default ``"affine"``. ``"affine"`` is the
        only supported value; other values raise ``ValueError``.
    reference_sample : Optional[int], optional
        **Deprecated.** Always treated as ``None`` regardless of value.
        Pass ``None`` (the default) to silence the deprecation warning.
        Will be removed in a future major release.
    lts_quantile : float, optional
        Quantile for the Least-Trimmed-Squares robust step, by default
        ``0.75``. Must lie in (0, 1]. ``1.0`` disables the robust step.

    Attributes
    ----------
    vsn_params : Optional[dict]
        Fitted parameters as a plain Python dict. Only populated after
        ``normalize()``. Keys:

        ``coefficients`` : np.ndarray, shape ``(1, n_samples, 2)``
            Per-sample (a, beta) coefficients in R-VSN's layout. Slice
            ``[0, :, 0]`` is ``a``; ``[0, :, 1]`` is ``beta = log(b)``.
        ``a`` : np.ndarray, shape ``(n_samples,)``
            Per-sample offset.
        ``b_log`` : np.ndarray, shape ``(n_samples,)``
            Per-sample log-scale factor; the actual scaling is
            ``np.exp(b_log)``.
        ``sigsq`` : float
            Profiled variance on the natural ``arsinh`` scale.
        ``hoffset`` : float
            Offset applied during transform to put output on a log2-like
            scale.
        ``mu`` : np.ndarray, shape ``(n_features,)``
            Per-feature mean of the transformed data on the natural
            ``arsinh`` scale (rows trimmed by LTS in iterations >1
            appear as ``NaN``, matching R's bookkeeping).
        ``converged`` : bool
            Whether the inner L-BFGS-B converged on the final LTS
            iteration.
        ``n_lts_iter`` : int
            Number of LTS iterations executed (capped at 7).
    """

    def __init__(
        self,
        calib: str = "affine",
        reference_sample: int | None = None,
        lts_quantile: float = 0.75,
    ) -> None:
        if calib != "affine":
            raise ValueError(f"Only calib='affine' is supported by the native VSN engine; got {calib!r}.")
        self.calib = calib
        if reference_sample is not None:
            warnings.warn(
                "VSNNormalizer's 'reference_sample' parameter is deprecated and will "
                "be removed in a future major release. R-VSN's 'reference' argument "
                "expects a fitted vsn object, not a single-sample index, so this "
                "parameter cannot be implemented faithfully. Calls now ignore the "
                "value and use vsn2's default reference selection.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.reference_sample = reference_sample
        if not 0 < lts_quantile <= 1:
            raise ValueError("lts_quantile must be between 0 and 1")
        self.lts_quantile = lts_quantile
        self.vsn_params: dict[str, Any] | None = None

    def normalize(
        self,
        X: np.ndarray,
        protein_ids: list[str] | None = None,
        sample_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Fit VSN on ``X`` and return the transformed matrix.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape ``(n_samples, n_features)``.
        protein_ids : Optional[List[str]], optional
            Unused; retained for API compatibility with the previous
            R-backed implementation.
        sample_ids : Optional[List[str]], optional
            Unused; retained for API compatibility.

        Returns
        -------
        np.ndarray
            Normalized data with the same shape as ``X``, on a
            log2-comparable (variance-stabilized) scale.

        Raises
        ------
        ValueError
            If the input contains NaN or Inf, or if the native VSN
            engine fails to fit (e.g., fewer than 2 samples).
        """
        del protein_ids, sample_ids  # accepted for back-compat, not used
        X = validate_input_data(X)
        has_nan_inf, _ = check_nan_inf(X)
        if has_nan_inf:
            raise ValueError("Input data contains NaN or Inf values. Please handle these values before normalization.")

        # Native engine works in (n_features, n_samples) orientation.
        try:
            X_t = np.ascontiguousarray(X.T, dtype=np.float64)
            normalized_t, fit = _vsn_engine.fit_transform(
                X_t,
                lts_quantile=self.lts_quantile,
            )
        except Exception as e:
            raise ValueError(f"VSN normalization failed: {e!s}") from e

        self.vsn_params = {
            "coefficients": fit.coefficients,
            "a": fit.a,
            "b_log": fit.b_log,
            "sigsq": fit.sigsq,
            "hoffset": fit.hoffset,
            "mu": fit.mu,
            "converged": fit.converged,
            "n_lts_iter": fit.n_lts_iter,
        }
        return np.ascontiguousarray(normalized_t.T)

    def plot_comparison(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        figsize: tuple[int, int] = (8, 8),
        gridsize: int = 50,
        cmap: str = "viridis",
    ) -> plt.Figure:
        """Plot a hexbin comparison of raw vs. VSN-normalized intensities.

        Parameters
        ----------
        before_data : np.ndarray
            Data before normalization.
        after_data : np.ndarray
            Data after normalization (output of ``normalize``).
        figsize : Tuple[int, int], optional
        gridsize : int, optional
        cmap : str, optional

        Returns
        -------
        plt.Figure
        """
        return plot_comparison_hexbin(
            before_data=before_data,
            after_data=after_data,
            title="VSN Normalization Comparison (glog2 vs log2)",
            figsize=figsize,
            gridsize=gridsize,
            cmap=cmap,
            transform_original="log2",
            log_axes=False,
        )
