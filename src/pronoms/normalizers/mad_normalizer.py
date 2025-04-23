import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison

class MADNormalizer:
    """
    Normalizer that scales each sample based on its Median Absolute Deviation (MAD).

    This normalizer standardizes each sample (row) by subtracting its median
    and then dividing by its MAD. MAD is a robust measure of variability.

    Normalization formula for a sample `x`:
    `normalized_x = (x - median(x)) / MAD(x)`
    where `MAD(x) = median(|x - median(x)|)`

    Attributes
    ----------
    row_medians : Optional[np.ndarray]
        The median value calculated for each sample (row). Available after normalize().
    row_mads : Optional[np.ndarray]
        The Median Absolute Deviation (MAD) calculated for each sample (row).
        Available after normalize().
    """

    def __init__(self):
        """Initialize the MADNormalizer."""
        self.row_medians = None
        self.row_mads = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Median Absolute Deviation (MAD) normalization on input data X.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix with shape (n_samples, n_features).
            Each row represents a sample, each column represents a feature/protein.

        Returns
        -------
        np.ndarray
            Normalized data matrix with the same shape as X.

        Raises
        ------
        ValueError
            - If input is not a 2D array with at least one feature.
            - If input data contains NaN or Inf values.
            - If MAD is zero for any sample (which prevents normalization).
        """
        # Validate input data
        X = validate_input_data(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature (n_samples, n_features).")

        # Check for NaN or Inf values
        has_nan_inf, _ = check_nan_inf(X)
        if has_nan_inf:
            raise ValueError(
                "Input data contains NaN or Inf values. Please handle these values before normalization."
            )

        # Calculate row-wise medians
        row_medians = np.median(X, axis=1, keepdims=True)

        # Calculate absolute deviations from the median
        abs_deviations = np.abs(X - row_medians)

        # Calculate row-wise MAD
        row_mads = np.median(abs_deviations, axis=1, keepdims=True)

        # Check for zero MAD values
        if np.any(row_mads == 0):
            zero_mad_indices = np.where(row_mads.flatten() == 0)[0]
            raise ValueError(
                f"Cannot normalize: MAD is zero for sample(s) at index/indices: {list(zero_mad_indices)}. "
                "This usually means all values in the sample are identical."
            )

        # Store state
        self.row_medians = row_medians.flatten()
        self.row_mads = row_mads.flatten()

        # Apply normalization: (X - median) / MAD
        normalized_data = (X - row_medians) / row_mads

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
        # Note: While MAD normalization results are on a different scale,
        # we use the consistent utility function as requested.
        # Axis labels reflect the function's default behavior.
        fig = create_hexbin_comparison(
            before_data,
            after_data,
            figsize=figsize,
            title=title,
            xlabel="Before MAD Normalization",
            ylabel="After MAD Normalization"
        )
        return fig
