"""
Median Normalizer for proteomics data.

This module provides a class for median normalization of proteomics data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison


class MedianNormalizer:
    """
    Normalizer that scales each sample by its median.
    
    This normalizer adjusts each sample (column) in the data matrix by dividing
    by the median value of that sample, effectively centering the distribution
    of each sample around a median of 1.0.
    
    Attributes
    ----------
    scaling_factors : Optional[np.ndarray]
        Scaling factors used for normalization (median of each sample).
        Only available after calling normalize().
    """
    
    def __init__(self):
        """Initialize the MedianNormalizer."""
        self.scaling_factors = None
        self.mean_of_medians = None
    
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Perform median normalization on input data X.
        
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
            raise ValueError(
                "Input data contains NaN or Inf values. Please handle these values before normalization."
            )
        
        # Calculate median for each sample (row) using faster partition method
        # This is O(n) instead of O(n log n) for large arrays
        n_features = X.shape[1]
        k = n_features // 2
        
        # Use np.partition which is much faster than sorting for finding medians
        if n_features % 2 == 1:  # Odd number of elements
            # For odd number of elements, median is the middle element
            medians = np.partition(X, k, axis=1)[:, k]
        else:  # Even number of elements
            # For even number of elements, median is the average of the two middle elements
            medians_high = np.partition(X, k, axis=1)[:, k]
            medians_low = np.partition(X, k-1, axis=1)[:, k-1]
            medians = (medians_high + medians_low) / 2
        
        # Calculate mean of medians to preserve original scale
        mean_of_medians = np.mean(medians)
        
        # Add keepdims for broadcasting
        medians = medians.reshape(-1, 1)
        
        # Avoid division by zero
        medians = np.where(medians == 0, 1.0, medians)
        
        # Store scaling factors
        self.scaling_factors = medians.flatten()
        self.mean_of_medians = mean_of_medians  # Store for reference
        
        # Normalize each sample by its median and multiply by mean of medians
        # to preserve the original scale of the data
        normalized_data = (X / medians) * mean_of_medians
        
        return normalized_data
    
    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray, 
                       figsize: Tuple[int, int] = (10, 8),
                       title: str = "Median Normalization Comparison") -> plt.Figure:
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
            Plot title, by default "Median Normalization Comparison".
        
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
            xlabel="Before Median Normalization",
            ylabel="After Median Normalization"
        )
        
        return fig
