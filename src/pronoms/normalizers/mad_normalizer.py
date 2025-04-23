import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison

class MADNormalizer:
    """
    Median Absolute Deviation (MAD) Normalizer.

    Centers each sample (row) by subtracting its median and scales it by its
    Median Absolute Deviation (MAD).

    Optionally performs calculations on log2-transformed data (default) to
    stabilize variance and handle typical intensity distributions.

    If `log_transform=True` (default):
        Calculations (median, MAD) are performed on `log2(X + 1)`.
        Normalization: `(log2(X + 1) - median_log) / MAD_log`
    If `log_transform=False`:
        Calculations are performed directly on `X`.
        Normalization: `(X - median) / MAD`

    Attributes
    ----------
    log_transform : bool
        Whether log2 transformation was applied before normalization.
    row_medians : np.ndarray
        Median of the (potentially log2-transformed) data for each sample.
    row_mads : np.ndarray
        Median Absolute Deviation (MAD) of the (potentially log2-transformed)
        data for each sample.
    """

    def __init__(self, log_transform: bool = True):
        """
        Initializes the MADNormalizer.

        Parameters
        ----------
        log_transform : bool, optional
            Whether to apply log2(X+1) transformation before calculating
            median and MAD, by default True.
        """
        self.log_transform = log_transform
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
        X_validated = validate_input_data(X) # Use a different name to avoid modifying X if log_transform is False
        if X_validated.ndim != 2 or X_validated.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature (n_samples, n_features).")

        # Check for NaN or Inf values (on original data)
        has_nan_inf, _ = check_nan_inf(X_validated)
        if has_nan_inf:
            raise ValueError(
                "Input data contains NaN or Inf values. Please handle these values before normalization."
            )

        data_to_process = X_validated
        scale_type = "original"

        if self.log_transform:
            # Check for negative values only if log transforming
            if np.any(X_validated < 0):
                raise ValueError("Input data contains negative values. Log2 transformation cannot be applied.")

            # Apply log2 transformation
            with np.errstate(divide='ignore', invalid='ignore'):
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
            raise ValueError(
                # Explicitly convert indices to int for consistent string formatting
                f"Cannot normalize: MAD of {scale_type} data is zero for sample(s) at index/indices: {[int(i) for i in zero_mad_indices]}. "
                f"This usually means all {scale_type} values in the sample are identical."
            )

        # Store state (medians and MADs from the scale used)
        self.row_medians = row_medians.flatten()
        self.row_mads = row_mads.flatten()

        # Apply normalization: (data_to_process - median) / MAD
        normalized_data = (data_to_process - row_medians) / row_mads

        return normalized_data

    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray,
                       figsize: Tuple[int, int] = (10, 8),
                       title: str = "MAD Normalization Comparison") -> plt.Figure:
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
            ylabel=f"After MAD Normalization ({'Standardized Log2 Scale' if self.log_transform else 'Standardized Original Scale'})",
            autoscale_y=True,
            add_identity_line=False,
            add_center_line_y0=True # Centered around 0 in both cases
        )
        return fig
