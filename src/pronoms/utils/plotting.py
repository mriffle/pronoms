"""
Plotting utilities for Pronoms.

This module provides functions for visualizing proteomics data before and after normalization.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union


def create_boxplot(
    data: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Distribution of values",
    xlabel: str = "Samples",
    ylabel: str = "Values",
    sample_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_points: bool = True,
    color: str = "skyblue",
    point_alpha: float = 0.3,
) -> plt.Axes:
    """
    Create a boxplot of the data.
    
    Parameters
    ----------
    data : np.ndarray
        Data to plot, shape (n_proteins, n_samples).
    ax : Optional[plt.Axes], optional
        Matplotlib axes to plot on, by default None (creates new axes).
    title : str, optional
        Plot title, by default "Distribution of values".
    xlabel : str, optional
        X-axis label, by default "Samples".
    ylabel : str, optional
        Y-axis label, by default "Values".
    sample_names : Optional[List[str]], optional
        Names for the samples, by default None (uses indices).
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 6).
    show_points : bool, optional
        Whether to show individual data points, by default True.
    color : str, optional
        Color of the boxplots, by default "skyblue".
    point_alpha : float, optional
        Alpha value for the data points, by default 0.3.
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the boxplot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    # Transpose data for boxplot (samples on x-axis)
    data_t = data.T if data.shape[0] > data.shape[1] else data
    
    # Create boxplot
    boxprops = dict(linewidth=1.5, color=color)
    whiskerprops = dict(linewidth=1.5, color=color)
    medianprops = dict(linewidth=2, color='darkblue')
    
    bp = ax.boxplot(
        data_t,
        patch_artist=True,
        showfliers=False,
        boxprops=boxprops,
        whiskerprops=whiskerprops,
        medianprops=medianprops,
    )
    
    # Fill boxes with color
    for box in bp['boxes']:
        box.set(facecolor=color, alpha=0.6)
    
    # Add data points if requested
    if show_points:
        for i in range(data_t.shape[1]):
            # Add jitter to x position
            x = np.random.normal(i + 1, 0.04, size=data_t.shape[0])
            ax.scatter(x, data_t[:, i], alpha=point_alpha, s=3, color='black')
    
    # Set x-axis tick labels
    if sample_names is not None:
        ax.set_xticklabels(sample_names, rotation=45, ha='right')
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    return ax


def create_scatter_plot(
    data1: np.ndarray,
    data2: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Comparison",
    xlabel: str = "Before",
    ylabel: str = "After",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.5,
    color: str = "blue",
    add_identity_line: bool = True,
    add_regression_line: bool = True,
    sample_idx: Optional[int] = None,
) -> plt.Axes:
    """
    Create a scatter plot comparing two datasets.
    
    Parameters
    ----------
    data1 : np.ndarray
        First dataset, shape (n_proteins, n_samples).
    data2 : np.ndarray
        Second dataset, shape (n_proteins, n_samples).
    ax : Optional[plt.Axes], optional
        Matplotlib axes to plot on, by default None (creates new axes).
    title : str, optional
        Plot title, by default "Comparison".
    xlabel : str, optional
        X-axis label, by default "Before".
    ylabel : str, optional
        Y-axis label, by default "After".
    figsize : Tuple[int, int], optional
        Figure size, by default (8, 8).
    alpha : float, optional
        Alpha value for the scatter points, by default 0.5.
    color : str, optional
        Color of the scatter points, by default "blue".
    add_identity_line : bool, optional
        Whether to add y=x line, by default True.
    add_regression_line : bool, optional
        Whether to add regression line, by default True.
    sample_idx : Optional[int], optional
        Index of the sample to plot, by default None (plots all samples).
        
    Returns
    -------
    plt.Axes
        Matplotlib axes with the scatter plot.
        
    Raises
    ------
    ValueError
        If data shapes don't match or sample_idx is out of bounds.
    """
    if data1.shape != data2.shape:
        raise ValueError(
            f"Data shapes must match: {data1.shape} != {data2.shape}"
        )
    
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    
    # If sample_idx is provided, plot only that sample
    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= data1.shape[1]:
            raise ValueError(
                f"Sample index {sample_idx} out of bounds for data with {data1.shape[1]} samples"
            )
        x = data1[:, sample_idx]
        y = data2[:, sample_idx]
        ax.scatter(x, y, alpha=alpha, color=color)
    else:
        # Plot all samples
        for i in range(data1.shape[1]):
            ax.scatter(data1[:, i], data2[:, i], alpha=alpha, color=color)
    
    # Add identity line (y=x)
    if add_identity_line:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    
    # Add regression line
    if add_regression_line:
        # Flatten data for regression if using all samples
        if sample_idx is None:
            x = data1.flatten()
            y = data2.flatten()
        else:
            x = data1[:, sample_idx]
            y = data2[:, sample_idx]
        
        # Calculate regression line
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        y_line = m * x_line + b
        ax.plot(x_line, y_line, 'r-', alpha=0.75, zorder=1)
        
        # Add regression equation to plot
        eq_text = f"y = {m:.2f}x + {b:.2f}"
        ax.annotate(
            eq_text,
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
        )
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Make axes equal
    ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout
    plt.tight_layout()
    
    return ax


def create_comparison_plot(
    before_data: np.ndarray,
    after_data: np.ndarray,
    figsize: Tuple[int, int] = (15, 8),
    title: str = "Normalization Comparison",
    before_label: str = "Before Normalization",
    after_label: str = "After Normalization",
    sample_names: Optional[List[str]] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]:
    """
    Create a comprehensive comparison plot with boxplots and scatter plot.
    
    Parameters
    ----------
    before_data : np.ndarray
        Data before normalization, shape (n_proteins, n_samples).
    after_data : np.ndarray
        Data after normalization, shape (n_proteins, n_samples).
    figsize : Tuple[int, int], optional
        Figure size, by default (15, 8).
    title : str, optional
        Main figure title, by default "Normalization Comparison".
    before_label : str, optional
        Label for before data, by default "Before Normalization".
    after_label : str, optional
        Label for after data, by default "After Normalization".
    sample_names : Optional[List[str]], optional
        Names for the samples, by default None (uses indices).
        
    Returns
    -------
    Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes]]
        Figure and axes objects for the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Create boxplots
    create_boxplot(
        before_data,
        ax=axes[0],
        title=before_label,
        sample_names=sample_names,
    )
    
    create_boxplot(
        after_data,
        ax=axes[1],
        title=after_label,
        sample_names=sample_names,
    )
    
    # Create scatter plot
    create_scatter_plot(
        before_data,
        after_data,
        ax=axes[2],
        title="Before vs. After",
        xlabel=before_label,
        ylabel=after_label,
    )
    
    # Set main title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    
    return fig, axes
