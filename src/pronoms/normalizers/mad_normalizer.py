import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import create_hexbin_comparison
from ..utils.validators import check_nan_inf, validate_input_data

# Consistency constant making MAD a σ-equivalent estimator under normality
# (1 / Φ⁻¹(0.75) ≈ 1.4826). Matches R's ``mad()`` default.
MAD_SIGMA_CONSTANT = 1.4826

_SCALE_DEFAULT_UNSPECIFIED = object()


class MADNormalizer:
    """
    Median Absolute Deviation (MAD) Normalizer.

    Centers each sample (row) by subtracting its median and scales it by its
    Median Absolute Deviation (MAD).

    Optionally performs calculations on log2-transformed data (default) to
    stabilize variance and handle typical intensity distributions.

    If `log_transform=True` (default):
        Calculations (median, MAD) are performed on `log2(X + 1)`.
        Normalization: `(log2(X + 1) - median_log) / (k * MAD_log)`
    If `log_transform=False`:
        Calculations are performed directly on `X`.
        Normalization: `(X - median) / (k * MAD)`

    Where ``k`` is the consistency constant set by ``scale_to_sigma``:

    - ``scale_to_sigma=True``: ``k = 1.4826`` (``MAD_SIGMA_CONSTANT``). The
      output is a robust z-score: per-row spread ≈ 1 σ for normal data.
      Matches R's ``mad()`` default and ``statsmodels.robust.scale.mad``.
    - ``scale_to_sigma=False``: ``k = 1`` (raw MAD divisor). Per-row spread
      is ≈ 1.4826 × what a true robust z-score would give. Use this if you
      explicitly want raw-MAD output and have not standardized to σ.

    .. deprecated::
        Calling without ``scale_to_sigma`` emits a ``DeprecationWarning``;
        the implicit default (raw MAD) will be replaced by
        ``scale_to_sigma=True`` in a future major release. Pass the
        argument explicitly to lock in your intended behavior across
        versions.

    Attributes
    ----------
    log_transform : bool
        Whether log2 transformation was applied before normalization.
    scale_to_sigma : bool
        Whether the divisor is ``MAD_SIGMA_CONSTANT * MAD`` (σ-equivalent)
        rather than raw MAD.
    row_medians : np.ndarray
        Median of the (potentially log2-transformed) data for each sample.
    row_mads : np.ndarray
        Raw Median Absolute Deviation (MAD) of the (potentially
        log2-transformed) data for each sample. Always the unscaled MAD,
        regardless of ``scale_to_sigma``.
    """

    def __init__(self, log_transform: bool = True, scale_to_sigma: bool = _SCALE_DEFAULT_UNSPECIFIED):  # type: ignore[assignment]
        """
        Initializes the MADNormalizer.

        Parameters
        ----------
        log_transform : bool, optional
            Whether to apply log2(X+1) transformation before calculating
            median and MAD, by default True.
        scale_to_sigma : bool, optional
            If True, divide by ``1.4826 * MAD`` so the output is a robust
            z-score (σ-equivalent under normality). If False, divide by raw
            MAD. The current implicit default is False for backward
            compatibility but will flip to True in a future major release;
            omitting the argument emits a ``DeprecationWarning``.
        """
        if scale_to_sigma is _SCALE_DEFAULT_UNSPECIFIED:
            warnings.warn(
                "MADNormalizer's default scaling will change in a future major "
                "release: the divisor will become 1.4826 * MAD (σ-equivalent, "
                "matching R's mad()) instead of raw MAD. Pass "
                "scale_to_sigma=True for the new behavior or "
                "scale_to_sigma=False to keep raw MAD.",
                DeprecationWarning,
                stacklevel=2,
            )
            scale_to_sigma = False

        self.log_transform = log_transform
        self.scale_to_sigma = bool(scale_to_sigma)
        self.row_medians: Optional[np.ndarray] = None
        self.row_mads: Optional[np.ndarray] = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Apply MAD normalization to the input data matrix X.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (n_samples, n_features).
            Must contain non-negative values if `log_transform=True`.

        Returns
        -------
        np.ndarray
            Normalized data matrix.

        Raises
        ------
        ValueError
            - If input is not a 2D array with at least one feature.
            - If input data contains NaN or Inf values.
            - If `log_transform=True` and input data contains negative values.
            - If MAD is zero for any sample (which prevents normalization).
        """
        # Validate input data type and shape first
        X_validated = validate_input_data(X)  # Use a different name to avoid modifying X if log_transform is False
        if X_validated.ndim != 2 or X_validated.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature (n_samples, n_features).")

        # Check for NaN or Inf values (on original data)
        has_nan_inf, _ = check_nan_inf(X_validated)
        if has_nan_inf:
            raise ValueError("Input data contains NaN or Inf values. Please handle these values before normalization.")

        data_to_process = X_validated
        scale_type = "original"

        if self.log_transform:
            # Check for negative values only if log transforming
            if np.any(X_validated < 0):
                raise ValueError("Input data contains negative values. Log2 transformation cannot be applied.")

            # Apply log2 transformation
            with np.errstate(divide="ignore", invalid="ignore"):
                log_X = np.log2(X_validated + 1)

            # Check for issues potentially introduced by log2
            if np.any(~np.isfinite(log_X)):
                raise ValueError(
                    "Non-finite values encountered after log2 transformation. Check input data near 0 or -1."
                )
            data_to_process = log_X
            scale_type = "log2(X+1)"

        # --- Calculations performed on data_to_process (either original or log2) ---

        # Calculate row-wise medians
        row_medians = np.median(data_to_process, axis=1, keepdims=True)

        # Calculate absolute deviations from the median
        abs_deviations = np.abs(data_to_process - row_medians)

        # Calculate row-wise MAD
        row_mads = np.median(abs_deviations, axis=1, keepdims=True)

        # Check for zero MAD values
        if np.any(row_mads == 0):
            zero_mad_indices = np.where(row_mads.flatten() == 0)[0]
            indices_repr = [int(i) for i in zero_mad_indices]
            raise ValueError(
                f"Cannot normalize: MAD of {scale_type} data is zero "
                f"for sample(s) at index/indices: {indices_repr}. "
                f"This usually means all {scale_type} values in the sample are identical."
            )

        # Store state (medians and *raw* MADs from the scale used).
        self.row_medians = row_medians.flatten()
        self.row_mads = row_mads.flatten()

        # Apply normalization: (data_to_process - median) / (k * MAD)
        # where k is the σ-consistency constant when scale_to_sigma is True.
        divisor = row_mads * MAD_SIGMA_CONSTANT if self.scale_to_sigma else row_mads
        normalized_data = (data_to_process - row_medians) / divisor

        return normalized_data

    def plot_comparison(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        figsize: tuple[int, int] = (10, 8),
        title: str = "MAD Normalization Comparison",
    ) -> plt.Figure:
        """
        Plot data before vs after normalization using a 2D hexbin density plot.

        Parameters
        ----------
        before_data : np.ndarray
            Data before normalization, shape (n_samples, n_features).
        after_data : np.ndarray
            Data after normalization, shape (n_samples, n_features).
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 8).
        title : str, optional
            Plot title, by default "MAD Normalization Comparison".

        Returns
        -------
        plt.Figure
            Figure object containing the hexbin density plot.
        """
        # Use the consistent utility function, but enable y-axis autoscaling
        # and add a horizontal line at y=0.
        fig = create_hexbin_comparison(
            before_data,
            after_data,
            figsize=figsize,
            title=title,
            xlabel="Original Data",
            ylabel=(
                "After MAD Normalization ("
                f"{'Standardized Log2 Scale' if self.log_transform else 'Standardized Original Scale'}"
                ")"
            ),
            autoscale_y=True,
            add_identity_line=False,
            add_center_line_y0=True,  # Centered around 0 in both cases
        )
        return fig
