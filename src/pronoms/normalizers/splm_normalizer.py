import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison

class SPLMNormalizer:
    """
    Normalizer based on Stable Protein Log-Mean Normalization (SPLM-Norm).

    Scales proteomics intensity data using a subset of stably expressed proteins
    (lowest coefficient of variation in log-space). It uses the mean of
    log-transformed intensities of these stable proteins per sample to define
    scaling factors, performs normalization in log-space, recenters, and then
    transforms back to the original scale.

    Attributes
    ----------
    num_stable_proteins : int
        Number of stable proteins used for calculating scaling factors.
    epsilon : float
        Small constant added before log transformation to avoid log(0).
    stable_protein_indices : Optional[np.ndarray]
        Indices of the proteins identified as stable. Available after normalize().
    log_scaling_factors : Optional[np.ndarray]
        The per-sample log-space scaling factors derived from stable proteins. Available after normalize().
    grand_mean_log_scaling_factor : Optional[float]
        The mean of the log_scaling_factors across all samples. Available after normalize().
    """
    def __init__(self, num_stable_proteins: int = 100, epsilon: float = 1e-6):
        """
        Initialize the SPLMNormalizer.

        Parameters
        ----------
        num_stable_proteins : int, optional
            Number of proteins with the lowest log-space CV to use as stable references, by default 100.
        epsilon : float, optional
            Small constant added to intensities before log transformation to avoid log(0), by default 1e-6.
        """
        if not isinstance(num_stable_proteins, int) or num_stable_proteins <= 0:
            raise ValueError("num_stable_proteins must be a positive integer.")
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError("epsilon must be a non-negative number.")

        self.num_stable_proteins = num_stable_proteins
        self.epsilon = epsilon
        self.stable_protein_indices = None
        self.log_scaling_factors = None
        self.grand_mean_log_scaling_factor = None

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Stable Protein Log-Mean Normalization on input data X.

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
            - If num_stable_proteins is greater than the number of features in X.
            - If stable proteins cannot be determined (e.g., all proteins have zero variance).
        """
        # 1. Validate input
        X = validate_input_data(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature (n_samples, n_features).")
        has_nan_inf, _ = check_nan_inf(X)
        if has_nan_inf:
            raise ValueError("Input data contains NaN or Inf values.")
        if self.num_stable_proteins > X.shape[1]:
            raise ValueError(f"num_stable_proteins ({self.num_stable_proteins}) cannot be greater than the number of features ({X.shape[1]}).")

        # 2. Log-transform
        # Add epsilon before log to avoid log(0) or log(negative) if data wasn't strictly positive
        X_log = np.log(X + self.epsilon)

        # 3. Compute protein-wise CV in log space
        log_means = np.mean(X_log, axis=0)
        log_stds = np.std(X_log, axis=0)

        # Handle potential division by zero (assign Inf CV to proteins with zero mean or std dev)
        # Proteins with zero std dev (constant log value) should have CV=0 and be preferred.
        log_cvs = np.full_like(log_means, np.inf) # Default to Inf
        # Calculate CV only where mean is non-zero (and implicitly std>0 for finite CV)
        valid_mask = (log_means != 0)
        # Avoid division by zero warning for log_stds == 0 cases
        non_zero_std_mask = (log_stds != 0)
        final_mask = valid_mask & non_zero_std_mask
        log_cvs[final_mask] = log_stds[final_mask] / log_means[final_mask]
        # Assign CV=0 to proteins with zero standard deviation (constant proteins)
        log_cvs[log_stds == 0] = 0

        if np.all(np.isinf(log_cvs)):
             raise ValueError("Could not compute valid CVs for any protein. Check input data variance.")

        # 4. Select stable reference proteins
        # Use partition instead of sort for efficiency if only top N are needed
        # partition puts the k-th smallest element in its sorted position
        partitioned_indices = np.argpartition(log_cvs, self.num_stable_proteins - 1)
        self.stable_protein_indices = partitioned_indices[:self.num_stable_proteins]
        # Ensure the indices are sorted for consistent testing/debugging if needed
        self.stable_protein_indices = np.sort(self.stable_protein_indices)

        # 5. Compute sample-wise log-scaling factors
        stable_log_data = X_log[:, self.stable_protein_indices]
        self.log_scaling_factors = np.mean(stable_log_data, axis=1)

        # 6. Normalize in log space
        # Reshape factors for broadcasting (n_samples,) -> (n_samples, 1)
        log_factors_reshaped = self.log_scaling_factors[:, np.newaxis]
        X_log_norm = X_log - log_factors_reshaped

        # 7. Recenter the data
        self.grand_mean_log_scaling_factor = np.mean(self.log_scaling_factors)
        X_log_recentered = X_log_norm + self.grand_mean_log_scaling_factor

        # 8. Back-transform to linear space
        X_norm = np.exp(X_log_recentered) - self.epsilon
        # Ensure non-negativity after subtracting epsilon
        X_norm = np.maximum(0, X_norm)

        return X_norm

    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray,
                       figsize: Tuple[int, int] = (10, 8),
                       title: str = "SPLM Normalization Comparison") -> plt.Figure:
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
            Plot title, by default "SPLM Normalization Comparison".

        Returns
        -------
        plt.Figure
            Figure object containing the hexbin density plot.
        """
        before_data = validate_input_data(before_data)
        after_data = validate_input_data(after_data)

        fig = create_hexbin_comparison(
            before_data,
            after_data,
            figsize=figsize,
            title=title,
            xlabel="Before SPLM Normalization",
            ylabel="After SPLM Normalization"
        )
        return fig
