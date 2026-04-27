import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import create_hexbin_comparison
from ..utils.validators import check_nan_inf, validate_input_data


class SPLMNormalizer:
    """
    Normalizer based on Stable Protein Log-Mean Normalization (SPLM-Norm).

    Scales proteomics intensity data using a subset of stably expressed proteins
    (lowest coefficient of variation, computed in linear space as
    ``std(X) / mean(X)`` per protein across samples). It uses the mean of
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
            Number of proteins with the lowest linear-space CV (``std/mean``)
            to use as stable references, by default 100.
        epsilon : float, optional
            Small constant added to intensities before log transformation to avoid log(0), by default 1e-6.
        """
        if not isinstance(num_stable_proteins, int) or num_stable_proteins <= 0:
            raise ValueError("num_stable_proteins must be a positive integer.")
        if not isinstance(epsilon, (int, float)) or epsilon < 0:
            raise ValueError("epsilon must be a non-negative number.")

        self.num_stable_proteins = num_stable_proteins
        self.epsilon = epsilon
        self.stable_protein_indices: np.ndarray | None = None
        self.log_scaling_factors: np.ndarray | None = None
        self.grand_mean_log_scaling_factor: float | None = None

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
            raise ValueError(
                f"num_stable_proteins ({self.num_stable_proteins}) "
                f"cannot be greater than the number of features ({X.shape[1]})."
            )

        # 2. Compute per-protein CV in linear space (std/mean across samples).
        # Constant proteins (std == 0) get CV=0 and are most preferred.
        # Proteins with zero mean would give an undefined CV; treat those as +inf
        # so they are deprioritized rather than crashing the selection.
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        cvs = np.full_like(means, np.inf)
        nonzero_mean = means != 0
        cvs[nonzero_mean] = stds[nonzero_mean] / means[nonzero_mean]
        cvs[stds == 0] = 0.0

        if np.all(np.isinf(cvs)):
            raise ValueError("Could not compute valid CVs for any protein. Check input data variance.")

        # 3. Select stable reference proteins (smallest CVs).
        partitioned_indices = np.argpartition(cvs, self.num_stable_proteins - 1)
        self.stable_protein_indices = np.sort(partitioned_indices[: self.num_stable_proteins])

        # 4. Log-transform for the centering step.
        # Epsilon guards against log(0) when intensities reach the floor.
        X_log = np.log(X + self.epsilon)

        # 5. Compute sample-wise log-scaling factors
        stable_log_data = X_log[:, self.stable_protein_indices]
        self.log_scaling_factors = np.mean(stable_log_data, axis=1)

        # 6. Normalize in log space
        # Reshape factors for broadcasting (n_samples,) -> (n_samples, 1)
        log_factors_reshaped = self.log_scaling_factors[:, np.newaxis]
        X_log_norm = X_log - log_factors_reshaped

        # 7. Recenter the data
        self.grand_mean_log_scaling_factor = float(np.mean(self.log_scaling_factors))
        X_log_recentered = X_log_norm + self.grand_mean_log_scaling_factor

        # 8. Back-transform to linear space
        X_norm = np.exp(X_log_recentered) - self.epsilon
        # Ensure non-negativity after subtracting epsilon
        X_norm = np.maximum(0, X_norm)

        return X_norm

    def plot_comparison(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        figsize: tuple[int, int] = (10, 8),
        title: str = "SPLM Normalization Comparison",
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
            ylabel="After SPLM Normalization",
        )
        return fig
