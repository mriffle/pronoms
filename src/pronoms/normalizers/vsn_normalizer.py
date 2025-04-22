"""
VSN (Variance Stabilizing Normalization) Normalizer for proteomics data.

This module provides a class for VSN normalization of proteomics data,
which is implemented using the vsn R package.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison
from ..utils.r_interface import setup_r_environment, run_r_script


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
    
    def __init__(self, calib: str = "affine", reference_sample: Optional[int] = None):
        """
        Initialize the VSNNormalizer.
        
        Parameters
        ----------
        calib : str, optional
            Calibration method, by default "affine".
            Options: "affine", "none", "shift", "maximum".
        reference_sample : Optional[int], optional
            Index of the reference sample to calibrate against, by default None.
            If None, the vsn package will choose a reference automatically.
        """
        self.calib = calib
        self.reference_sample = reference_sample
        self.vsn_params = None
        
        # Check R environment and required packages
        self._check_r_dependencies()
    
    def _check_r_dependencies(self):
        """Check if required R packages are installed."""
        try:
            setup_r_environment(["vsn"])
        except Exception as e:
            print(f"Warning: {str(e)}")
            print("VSN normalization will not be available.")
    
    def normalize(self, X: np.ndarray,
                 protein_ids: Optional[List[str]] = None,
                 sample_ids: Optional[List[str]] = None) -> np.ndarray:
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
            raise ValueError(
                "Input data contains NaN or Inf values. Please handle these values before normalization."
            )
        
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
            results = run_r_script(
                r_script,
                data=X_transposed,
                row_names=protein_ids,
                col_names=sample_ids
            )
            
            # Extract normalized data and transpose back to original orientation
            if 'normalized_data' in results:
                normalized_data = results['normalized_data'].T
            else:
                raise ValueError("VSN normalization failed to return normalized data")
            
            # Store VSN parameters
            if 'parameters' in results:
                # Convert the R list-like object to a Python dict for easier access
                r_params = results['parameters']
                self.vsn_params = {}
                # Access elements using string keys after conversion
                # Note: Accessing R list elements might require checking names
                param_names = list(r_params.names)
                if 'coefficients' in param_names:
                     self.vsn_params['coefficients'] = np.array(r_params.rx2('coefficients'))
        
            return normalized_data
            
        except Exception as e:
            raise ValueError(f"VSN normalization failed: {str(e)}")
    
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
        
        # Conditionally add reference parameter to vsn2 call
        vsn2_call_base = "vsn2(eset, minDataPointsPerStratum = 3, lts.quantile = 0.75"
        if self.reference_sample is not None:
            ref_index = self.reference_sample + 1 # R is 1-indexed
            vsn2_call = f"{vsn2_call_base}, reference = {ref_index})"
        else:
            vsn2_call = f"{vsn2_call_base})" # No reference argument
            
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
    
    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray, 
                       figsize: Tuple[int, int] = (10, 8),
                       title: str = "VSN Normalization Comparison") -> plt.Figure:
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
            Plot title, by default "VSN Normalization Comparison".
        
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
            xlabel="Before VSN Normalization",
            ylabel="After VSN Normalization"
        )
        
        # If VSN parameters are available, add them to the plot
        if self.vsn_params is not None:
            # Create a string with the VSN parameters
            param_text = "VSN Parameters:\n"
            
            # Add only the most important parameters to avoid cluttering
            if 'stdev' in self.vsn_params:
                stdev = self.vsn_params['stdev']
                param_text += f"Stdev: {stdev:.3f}\n"
            
            if 'reference' in self.vsn_params:
                ref = self.vsn_params['reference']
                param_text += f"Reference: {ref}\n"
            
            # Add text box with VSN parameters
            plt.figtext(
                0.01, 0.01, param_text, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )
        
        return fig
