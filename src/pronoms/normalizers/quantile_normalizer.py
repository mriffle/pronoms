"""
Quantile Normalizer for proteomics data.

This module provides a class for quantile normalization of proteomics data.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from ..utils.plotting import create_hexbin_comparison
from ..utils.validators import check_nan_inf, validate_input_data


class QuantileNormalizer:
    """
    Normalizer that performs quantile normalization across samples.

    Quantile normalization makes the distribution of intensities for each sample
    identical by replacing each value with the mean of the corresponding quantiles
    across all samples.

    Tied values within a row receive the same normalized value (the average of
    the reference values at the tied ranks), following Bolstad et al. (2003).

    Attributes
    ----------
    reference_distribution : Optional[np.ndarray]
        The reference distribution used for normalization.
        Only available after calling normalize().
    """

    def __init__(self):
        """Initialize the QuantileNormalizer."""
        self.reference_distribution = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Perform quantile normalization on input data X.

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

        n_features = X.shape[1]

        # Reference distribution: column-mean of row-sorted data.
        reference = np.mean(np.sort(X, axis=1), axis=0)
        self.reference_distribution = reference

        # Tie-aware mapping: average ranks (1..N) → reference values.
        # Linear interpolation reproduces the Bolstad et al. tie rule, since for
        # a tie at integer half-ranks like 1.5 it returns (ref[0] + ref[1]) / 2.
        ranks = rankdata(X, method="average", axis=1)
        normalized_data = np.interp(ranks, np.arange(1, n_features + 1), reference)

        return normalized_data

    def plot_comparison(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        figsize: tuple[int, int] = (10, 8),
        title: str = "Quantile Normalization Comparison",
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
            Plot title, by default "Quantile Normalization Comparison".

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
            xlabel="Before Quantile Normalization",
            ylabel="After Quantile Normalization",
        )

        # If reference distribution is available, add a note about it
        if self.reference_distribution is not None:
            plt.figtext(
                0.01,
                0.01,
                "Quantile normalization transforms all samples\nto match a common reference distribution.",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
            )

        return fig
