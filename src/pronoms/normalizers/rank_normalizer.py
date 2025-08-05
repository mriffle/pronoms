"""
Rank Normalizer for proteomics data.

This module provides a class for rank transformation normalization of proteomics data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
from scipy.stats import rankdata

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison


class RankNormalizer:
    """
    Normalizer that transforms each sample's values to their ranks.
    
    This normalizer replaces each value in a sample with its rank, where the
    smallest value gets rank 1 and the largest gets rank N (number of features).
    Tied values are assigned the median rank of their group.
    
    Attributes
    ----------
    normalize_by_n : bool
        Whether to divide ranks by N (number of features) for comparability.
    ranks : Optional[np.ndarray]
        The rank-transformed data. Only available after calling normalize().
    """
    
    def __init__(self, normalize_by_n: bool = False):
        """
        Initialize the RankNormalizer.
        
        Parameters
        ----------
        normalize_by_n : bool, optional
            If True, divide all ranks by N (number of features) to get values
            between 1/N and 1. This makes ranks comparable across datasets
            with different numbers of features. By default False.
        """
        self.normalize_by_n = normalize_by_n
        self.ranks = None
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Perform rank transformation on input data X.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix with shape (n_samples, n_features).
            Each row represents a sample, each column represents a feature/protein.

        Returns
        -------
        np.ndarray
            Rank-transformed data matrix with the same shape as X.
            Values range from 1 to N (or 1/N to 1 if normalize_by_n=True).

        Raises
        ------
        ValueError
            - If input is not a 2D array with at least one feature.
            - If input data contains NaN or Inf values.
        """
        # Dimensionality guard
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature (n_samples, n_features).")

        # Validate input data (dtype conversion, etc.)
        X = validate_input_data(X)

        # Check for NaN or Inf values
        has_nan_inf, _ = check_nan_inf(X)
        if has_nan_inf:
            raise ValueError(
                "Input data contains NaN or Inf values. Please handle these values before normalization."
            )

        n_samples, n_features = X.shape
        rank_data = np.zeros_like(X, dtype=float)

        # Apply rank transformation to each sample (row)
        for i in range(n_samples):
            # Use scipy.stats.rankdata with method='average' for median rank of ties
            # This automatically handles tied values by assigning the average rank
            sample_ranks = rankdata(X[i, :], method='average')
            
            if self.normalize_by_n:
                # Normalize ranks to [1/N, 1] range
                sample_ranks = sample_ranks / n_features
            
            rank_data[i, :] = sample_ranks

        # Store the transformed data
        self.ranks = rank_data.copy()

        return rank_data
    
    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray, 
                       figsize: Tuple[int, int] = (10, 8),
                       title: str = "Rank Normalization Comparison",
                       log_x_axis: bool = True) -> plt.Figure:
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
            Plot title, by default "Rank Normalization Comparison".
        log_x_axis : bool, optional
            If True (default), plot log10 of the original values on the x-axis. 
            If False, plot raw original values. The y-axis always shows the 
            actual rank values from the normalization.
        
        Returns
        -------
        plt.Figure
            Figure object containing the hexbin density plot.
        """
        # Validate input data
        before_data = validate_input_data(before_data)
        after_data = validate_input_data(after_data)

        # Prepare x-axis data based on log_x_axis parameter
        if log_x_axis:
            # Log-transform the original data for the x-axis
            with np.errstate(divide='ignore', invalid='ignore'):
                x_data = np.log10(before_data + 1)
            xlabel = "Log10(Original Value + 1)"
            
            # Find min/max of log-transformed data for x-axis range
            x_min, x_max = np.min(x_data[np.isfinite(x_data)]), np.max(x_data[np.isfinite(x_data)])
        else:
            # Use raw original data for x-axis
            x_data = before_data
            xlabel = "Original Value"
            
            # Find min/max of original data for x-axis range
            x_min, x_max = np.min(x_data), np.max(x_data)
        
        # Add padding to x-axis range
        padding = (x_max - x_min) * 0.05  # 5% padding
        xlim = (x_min - padding, x_max + padding)
        
        # Set y-axis limits based on the actual range of normalized data
        # Check if normalize_by_n was used by examining the data range
        y_min, y_max = np.min(after_data), np.max(after_data)
        
        if y_max <= 1.0:  # normalize_by_n=True case
            ylim = (-0.05, 1.05)  # Small padding around [0, 1]
            ylabel = "Normalized Rank (0 to 1)"
        else:  # normalize_by_n=False case (ranks from 1 to N)
            n_features = before_data.shape[1]
            ylim = (0, n_features + 1)
            ylabel = f"Assigned Rank (1 to {n_features})"

        return create_hexbin_comparison(
            x_data,
            after_data,
            figsize=figsize,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            log_axes=False,  # We handle transformations manually
            xlim=xlim,
            ylim=ylim,
            autoscale_y=False,  # Use our custom y-axis limits
            add_identity_line=False  # Identity line is not meaningful here
        )
