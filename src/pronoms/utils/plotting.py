"""
Plotting utilities for Pronoms.

This module provides functions for visualizing proteomics data before and after normalization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def create_hexbin_comparison(
    before_data: np.ndarray,
    after_data: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Before vs. After Normalization",
    xlabel: str = "Before Normalization",
    ylabel: str = "After Normalization",
    gridsize: int = 50,
    cmap: str = "viridis",
    add_identity_line: bool = True,
    transform_original: Optional[str] = None
) -> plt.Figure:
    """
    Create a 2D hexbin density plot comparing values before and after normalization.
    
    Parameters
    ----------
    before_data : np.ndarray
        Data before normalization, shape (n_samples, n_features).
    after_data : np.ndarray
        Data after normalization, shape (n_samples, n_features).
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8).
    title : str, optional
        Plot title, by default "Before vs. After Normalization".
    xlabel : str, optional
        X-axis label, by default "Before Normalization".
    ylabel : str, optional
        Y-axis label, by default "After Normalization".
    gridsize : int, optional
        Number of hexagons in the x-direction, by default 50.
    cmap : str, optional
        Colormap to use, by default "viridis".
    add_identity_line : bool, optional
        Whether to add an identity line (y=x), by default True.
    transform_original : Optional[str], optional
        Apply a transformation to the 'before_data' before plotting.
        String indicating transformation type (e.g., 'log2'). Currently only 'log2' is supported.
        If 'log2', np.log2(before_data + 1) is plotted on the x-axis.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with the hexbin plot.
    """
    # Check that data shapes match
    if before_data.shape != after_data.shape:
        raise ValueError(
            f"Data shapes must match: {before_data.shape} != {after_data.shape}"
        )
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Flatten the data for plotting
    x = before_data.flatten()
    y = after_data.flatten()

    # Apply transformation to 'before' data if specified
    x_label = xlabel
    if transform_original == 'log2':
        # Add 1 before log2 to handle zeros, filter out resulting NaNs/Infs if any
        # Use np.errstate to suppress warnings about invalid values (handled by filtering)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_transformed = np.log2(x + 1)
        
        # Filter out non-finite values that might arise from log2(-ve+1) or original NaNs/Infs
        valid_indices = np.isfinite(x_transformed) & np.isfinite(y)
        x = x_transformed[valid_indices]
        y = y[valid_indices]
        x_label = 'Log2(Original Value + 1)'
    else:
        # If no transformation, just filter out non-finite values from both
        valid_indices = np.isfinite(x) & np.isfinite(y)
        x = x[valid_indices]
        y = y[valid_indices]

    # Create hexbin plot
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1, bins='log')
    
    # Add colorbar
    cb = fig.colorbar(hb, ax=ax, label='log10(count)')
    
    # Add identity line (y=x) if requested
    if add_identity_line:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.7, zorder=0)
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Make axes equal
    ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Alias for backward compatibility and VSNNormalizer usage
plot_comparison_hexbin = create_hexbin_comparison
