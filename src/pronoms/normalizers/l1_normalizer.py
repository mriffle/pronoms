"""
L1 Normalizer for proteomics data.

This module provides a class for L1 normalization of proteomics data.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import create_hexbin_comparison
from ..utils.validators import check_nan_inf, validate_input_data


class L1Normalizer:
    """
    Normalizer that scales each sample to have an L1 norm of 1.

    L1 normalization divides each value in a sample by the sum of absolute values
    in that sample. This is also known as "sum normalization" in proteomics.

    Attributes
    ----------
    scaling_factors : Optional[np.ndarray]
        Scaling factors used for normalization (L1 norm of each sample).
        Only available after calling normalize().
    mean_of_scaling_factors : Optional[float]
        Mean of scaling factors used to preserve original scale.
        Only available after calling normalize().
    """

    def __init__(self):
        """Initialize the L1Normalizer."""
        self.scaling_factors = None
        self.mean_of_scaling_factors = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Perform L1 normalization on input data X.

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
            If input data contains NaN or Inf values.
        """
        # Validate input data
        X = validate_input_data(X)

        # Check for NaN or Inf values
        has_nan_inf, _ = check_nan_inf(X)
        if has_nan_inf:
            raise ValueError("Input data contains NaN or Inf values. Please handle these values before normalization.")

        # Calculate L1 norm (sum of absolute values) for each sample (row)
        l1_norms = np.sum(np.abs(X), axis=1, keepdims=True)

        # Store the true norms and their unbiased mean for inspection.
        self.scaling_factors = l1_norms.flatten()
        self.mean_of_scaling_factors = float(np.mean(self.scaling_factors))

        # Zero-guard the divisor only; an all-zero row stays all-zero because
        # the numerator is also zero.
        divisor = np.where(l1_norms == 0, 1.0, l1_norms)
        normalized_data = (X / divisor) * self.mean_of_scaling_factors

        return normalized_data

    def plot_comparison(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        figsize: tuple[int, int] = (10, 8),
        title: str = "L1 Normalization Comparison",
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
            Plot title, by default "L1 Normalization Comparison".

        Returns
        -------
        plt.Figure
            Figure object containing the hexbin density plot.
        """
        # Validate input data
        before_data = validate_input_data(before_data)
        after_data = validate_input_data(after_data)

        # Create hexbin comparison plot
        fig = create_hexbin_comparison(
            before_data,
            after_data,
            figsize=figsize,
            title=title,
            xlabel="Before L1 Normalization",
            ylabel="After L1 Normalization",
        )

        return fig
