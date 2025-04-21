"""
Median Normalizer for proteomics data.

This module provides a class for median normalization of proteomics data.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_comparison_plot


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
        
        # Add keepdims for broadcasting
        medians = medians.reshape(-1, 1)
        
        # Avoid division by zero
        medians = np.where(medians == 0, 1.0, medians)
        
        # Store scaling factors
        self.scaling_factors = medians.flatten()
        
        # Normalize each sample by its median
        normalized_data = X / medians
        return normalized_data
    
    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray, 
                       sample_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (15, 8),
                       title: str = "Median Normalization Comparison") -> plt.Figure:
        """
        Plot data before vs after normalization.
        
        Parameters
        ----------
        before_data : np.ndarray
            Data before normalization, shape (n_samples, n_features).
        after_data : np.ndarray
            Data after normalization, shape (n_samples, n_features).
        sample_names : Optional[List[str]], optional
            Names for the samples, by default None (uses indices).
        figsize : Tuple[int, int], optional
            Figure size, by default (15, 8).
        title : str, optional
            Plot title, by default "Median Normalization Comparison".
        
        Returns
        -------
        plt.Figure
            Figure object containing the comparison plots.
        """
        # Validate input data
        before_data = validate_input_data(before_data)
        after_data = validate_input_data(after_data)
        
        # Create comparison plot
        fig, _ = create_comparison_plot(
            before_data,
            after_data,
            figsize=figsize,
            title=title,
            before_label="Before Median Normalization",
            after_label="After Median Normalization",
            sample_names=sample_names,
        )
        
        # If scaling factors are available, add them to the plot
        if self.scaling_factors is not None:
            factor_text = "Scaling Factors:\n"
            for i, factor in enumerate(self.scaling_factors):
                sample_name = f"Sample {i}" if sample_names is None else sample_names[i]
                factor_text += f"{sample_name}: {factor:.3f}\n"
            
            # Add text box with scaling factors
            plt.figtext(
                0.01, 0.01, factor_text, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )
        
        return fig
