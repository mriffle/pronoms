"""
VSN (Variance Stabilizing Normalization) Normalizer for proteomics data.

This module provides a class for VSN normalization of proteomics data,
which is implemented using the vsn R package.
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..utils.plotting import plot_comparison_hexbin
from ..utils.r_interface import run_r_script, setup_r_environment
from ..utils.validators import check_nan_inf, validate_input_data


class VSNNormalizer:
    """
    Normalizer that performs Variance Stabilizing Normalization using the R vsn package.

    VSN is a normalization method that stabilizes the variance across the intensity range,
    making the variance independent of the mean intensity. This is particularly useful
    for proteomics data where the variance often increases with the mean.

    Attributes
    ----------
    vsn_params : Optional[Dict[str, Any]]
        Parameters from the VSN normalization process.
        Only available after calling normalize().
    """

    def __init__(self, calib: str = "affine", reference_sample: Optional[int] = None, lts_quantile: float = 0.75):
        """
        Initialize the VSNNormalizer.

        Parameters
        ----------
        calib : str, optional
            Calibration method, by default "affine".
            Options: "affine", "none", "shift", "maximum".
        reference_sample : Optional[int], optional
            **Deprecated.** Always treated as ``None`` regardless of value.
            R-VSN's ``reference`` argument expects a previously fitted ``vsn``
            object, not a single-sample index, so the parameter as exposed
            here cannot be implemented faithfully and will be removed in a
            future major release. Pass ``None`` (the default) to silence the
            deprecation warning.
        lts_quantile : float, optional
            Quantile for the resistant regression (Linear Threshold Shift), by default 0.75.
            Controls the robustness of the normalization. Must be between 0 and 1.
        """
        self.calib = calib
        if reference_sample is not None:
            warnings.warn(
                "VSNNormalizer's 'reference_sample' parameter is deprecated and will "
                "be removed in a future major release. R-VSN's 'reference' argument "
                "expects a fitted vsn object, not a single-sample index, so this "
                "parameter cannot be implemented faithfully. Calls now ignore the "
                "value and use vsn2's default reference selection.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.reference_sample = reference_sample
        if not 0 <= lts_quantile <= 1:
            raise ValueError("lts_quantile must be between 0 and 1")
        self.lts_quantile = lts_quantile
        self.vsn_params = None

        # Check R environment and required packages
        self._check_r_dependencies()

    def _check_r_dependencies(self):
        """Check if required R packages are installed."""
        try:
            setup_r_environment(["vsn"])
        except Exception as e:
            print(f"Warning: {e!s}")
            print("VSN normalization will not be available.")

    def normalize(
        self, X: np.ndarray, protein_ids: Optional[list[str]] = None, sample_ids: Optional[list[str]] = None
    ) -> np.ndarray:
        """
        Perform VSN normalization on input data X.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix with shape (n_samples, n_features).
            Each row represents a sample, each column represents a feature/protein.
        protein_ids : Optional[List[str]], optional
            Protein/feature identifiers, by default None (uses column indices).
        sample_ids : Optional[List[str]], optional
            Sample identifiers, by default None (uses row indices).

        Returns
        -------
        np.ndarray
            Normalized data matrix with the same shape as X.

        Raises
        ------
        ValueError
            If input data contains NaN or Inf values or if R integration fails.
        """
        # Validate input data
        X = validate_input_data(X)

        # Check for NaN or Inf values
        has_nan_inf, _ = check_nan_inf(X)
        if has_nan_inf:
            raise ValueError("Input data contains NaN or Inf values. Please handle these values before normalization.")

        # Generate default IDs if not provided
        if protein_ids is None:
            protein_ids = [f"Protein_{i}" for i in range(X.shape[1])]

        if sample_ids is None:
            sample_ids = [f"Sample_{i}" for i in range(X.shape[0])]

        # Create R script for VSN normalization
        r_script = self._create_vsn_script()

        try:
            # Transpose data for R script since VSN expects proteins as rows
            # and samples as columns in the R environment
            X_transposed = X.T

            # Run R script with transposed data
            results = run_r_script(r_script, data=X_transposed, row_names=protein_ids, col_names=sample_ids)

            # Extract normalized data and transpose back to original orientation
            if "normalized_data" in results:
                normalized_data = results["normalized_data"].T
            else:
                raise ValueError("VSN normalization failed to return normalized data")

            # Store VSN parameters
            if "parameters" in results:
                # Directly store the parameters dict from results
                self.vsn_params = results["parameters"]

            return normalized_data

        except Exception as e:
            raise ValueError(f"VSN normalization failed: {e!s}") from e

    def _create_vsn_script(self) -> str:
        """
        Create R script for VSN normalization using ExpressionSet.

        Returns
        -------
        str
            R script for VSN normalization.
        """
        # Base R script parts
        script_start = """
        # Load required packages
        library(vsn)
        library(Biobase)

        # Check if input_data exists
        if (!exists("input_data")) {
            stop("Input data matrix not provided")
        }

        # Create ExpressionSet (should inherit names from input_data)
        eset <- tryCatch({
            ExpressionSet(assayData = input_data)
        }, error = function(e) {
            stop(paste("Error creating ExpressionSet:", e$message))
        })
        """

        # ``reference_sample`` is deprecated; the script always emits the
        # default vsn2 call. See __init__ for the rationale.
        vsn2_call = f"vsn2(eset, minDataPointsPerStratum = 3, lts.quantile = {self.lts_quantile})"

        script_end = f"""
        # Run VSN normalization on the ExpressionSet
        vsn_fit <- tryCatch({{
            {vsn2_call}
        }}, error = function(e) {{
            stop(paste("Error during vsn2 execution:", e$message))
        }})

        # Get normalized data
        normalized_data <- exprs(vsn_fit)

        # Extract parameters (Note: @stdev removed as it's not a standard slot for vsn object here)
        parameters <- list(
            coefficients = vsn_fit@coefficients
        )
        """

        return script_start + script_end

    def plot_comparison(
        self,
        before_data: np.ndarray,
        after_data: np.ndarray,
        figsize: tuple[int, int] = (8, 8),
        gridsize: int = 50,
        cmap: str = "viridis",
    ) -> plt.Figure:
        """
        Plot comparison using hexbin plot.
        For VSN, this plots log2(Original + 1) vs Normalized (glog2).

        Parameters
        ----------
        before_data : np.ndarray
            Data before normalization.
        after_data : np.ndarray
            Data after normalization (normalized using this instance).
        figsize : Tuple[int, int], optional
            Figure size, by default (8, 8).
        gridsize : int, optional
            Number of hexagons in the x-direction, by default 50.
        cmap : str, optional
            Colormap for the hexbins, by default 'viridis'.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        return plot_comparison_hexbin(
            before_data=before_data,
            after_data=after_data,
            title="VSN Normalization Comparison (glog2 vs log2)",
            figsize=figsize,
            gridsize=gridsize,
            cmap=cmap,
            transform_original="log2",
            log_axes=False,
        )
