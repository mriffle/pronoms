import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison

class MedianPolishNormalizer:
    """
    Normalizer based on Tukey's Median Polish algorithm.

    This algorithm iteratively removes median effects from rows (samples) and
    columns (features) of a matrix, typically applied to log-transformed data.
    It decomposes the data `X` into:
    `X[i, j] = overall_median + row_effect[i] + col_effect[j] + residual[i, j]`

    The normalized data returned is typically the `residuals + overall_median`,
    transformed back to the original scale if log transformation was used.

    Attributes
    ----------
    max_iterations : int
        Maximum number of iterations allowed for the algorithm.
    tolerance : float
        Convergence threshold. The algorithm stops if the sum of absolute changes
        in residuals is less than this value.
    epsilon : float
        Small constant added before log transformation to handle non-positive values.
    log_transform : bool
        Whether to apply log transformation before median polish and back-transform after.
    row_effects : Optional[np.ndarray]
        The calculated median effects for each row (sample). Available after normalize().
    col_effects : Optional[np.ndarray]
        The calculated median effects for each column (feature). Available after normalize().
    overall_median : Optional[float]
        The calculated overall median effect. Available after normalize().
    residuals : Optional[np.ndarray]
        The final residuals after removing row, column, and overall effects.
        Available after normalize().
    converged : Optional[bool]
        Whether the algorithm converged within max_iterations. Available after normalize().
    iterations_run : Optional[int]
        Number of iterations actually performed. Available after normalize().
    """
    def __init__(self, max_iterations: int = 10, tolerance: float = 0.01, epsilon: float = 1e-6, log_transform: bool = True):
        """
        Initialize the MedianPolishNormalizer.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations, by default 10.
        tolerance : float, optional
            Convergence tolerance based on sum of absolute changes in residuals, by default 0.01.
        epsilon : float, optional
            Small constant added before log transformation (if used), by default 1e-6.
            Only used if `log_transform` is True.
        log_transform : bool, optional
            Whether to log-transform the data before applying median polish and
            exponentiate after, by default True.
        """
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("tolerance must be a non-negative number")
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
             raise ValueError("epsilon must be a non-negative number")
        if not isinstance(log_transform, bool):
            raise ValueError("log_transform must be a boolean")

        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.log_transform = log_transform

        # Results attributes initialized to None
        self.row_effects: Optional[np.ndarray] = None
        self.col_effects: Optional[np.ndarray] = None
        self.overall_median: Optional[float] = None
        self.residuals: Optional[np.ndarray] = None
        self.converged: Optional[bool] = None
        self.iterations_run: Optional[int] = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Tukey's Median Polish normalization to the data.

        If log_transform is True, the input data is log-transformed before polishing.
        The method returns the normalized data defined as overall_median + residuals.
        Note: If log_transform was used, the returned data remains in log-space.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Normalized data matrix (overall_median + residuals).
            If log_transform=True, this matrix is in log-space.
        """
        X = validate_input_data(X)

        # log / shift handling ------------------------------------------------
        if self.log_transform:
            Xp = np.log(X + self.epsilon)
        else:
            Xp = X.copy()

        # second sanity check
        has_nan_inf, _ = check_nan_inf(Xp)
        if has_nan_inf:
            raise ValueError("Input contains NaN or Inf values.")

        n_rows, n_cols = Xp.shape
        # initialise effects --------------------------------------------------
        self.row_effects = np.zeros(n_rows)
        self.col_effects = np.zeros(n_cols)
        self.overall_median = 0.0
        resid = Xp.copy()

        # iterative polish ----------------------------------------------------
        self.converged = False
        for it in range(self.max_iterations):
            # ----- row step --------------------------------------------------
            row_med = np.median(resid, axis=1)
            resid -= row_med[:, None]
            self.row_effects += row_med

            # centre row effects and update overall
            rm = np.median(self.row_effects)
            self.row_effects -= rm
            self.overall_median += rm

            # ----- column step ----------------------------------------------
            col_med = np.median(resid, axis=0)
            resid -= col_med
            self.col_effects += col_med

            # centre column effects and update overall
            cm = np.median(self.col_effects)
            self.col_effects -= cm
            self.overall_median += cm

            # ----- convergence check ----------------------------------------
            max_change = max(np.abs(row_med).max(), np.abs(col_med).max())
            if max_change <= self.tolerance:
                self.converged = True
                break

        self.iterations_run = it + 1

        # store residuals
        self.residuals = resid

        # return log-space normalized matrix
        return self.overall_median + resid

    def plot_comparison(self, original_data: np.ndarray, normalized_data: np.ndarray, figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Generate a hexbin plot comparing original data (log scale) vs. normalized data.

        If log_transform was used during normalization, the normalized data (y-axis)
        will be in log-space. The original data (x-axis) is always plotted on a log scale
        for comparison consistency, especially when normalization involved log transform.

        Parameters
        ----------
        original_data : np.ndarray
            The raw data matrix (n_samples, n_features).
        normalized_data : np.ndarray
            The data matrix after normalization (n_samples, n_features).
            This will be in log-space if log_transform=True was used.
        figsize : Tuple[int, int], optional
            Figure size for the plot, by default (10, 8).

        Returns
        -------
        plt.Figure
            Matplotlib figure object containing the hexbin plot.
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Flatten data for hexbin plot
        x_flat = original_data.flatten()
        y_flat = normalized_data.flatten()

        # Filter out non-positive values for log scale on x-axis
        valid_indices = x_flat > 0
        x_filtered = x_flat[valid_indices]
        y_filtered = y_flat[valid_indices]

        if len(x_filtered) == 0:
            ax.text(0.5, 0.5, "No positive data to plot on log scale", ha='center', va='center')
            ax.set_title("Median Polish Normalization Comparison")
            ax.set_xlabel("Original Data (Log Scale)")
            ax.set_ylabel("Normalized Data")
            return fig

        # Create hexbin plot
        hb = ax.hexbin(x_filtered, y_filtered, gridsize=50, cmap='viridis', xscale='log')
        fig.colorbar(hb, ax=ax, label='Count in bin')

        # Add diagonal line (transformed appropriately if y is log-scale)
        # Determine if y is likely log-scale (heuristic: if log_transform was True)
        if self.log_transform:
            # If y is log, plot log(x) vs y
            min_val = np.log(x_filtered.min())
            max_val = np.log(x_filtered.max())
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='y = log(x)')
            ax.set_ylabel("Normalized Data (Log Scale)")
        else:
            # If y is original scale, plot x vs y
            min_val = x_filtered.min()
            max_val = x_filtered.max()
            ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1, label='y = x')
            ax.set_ylabel("Normalized Data")

        ax.set_title("Median Polish Normalization Comparison")
        ax.set_xlabel("Original Data (Log Scale)")
        ax.legend()
        fig.tight_layout()

        return fig
