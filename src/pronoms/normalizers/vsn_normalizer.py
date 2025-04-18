"""
VSN (Variance Stabilizing Normalization) Normalizer for proteomics data.

This module provides a class for VSN normalization of proteomics data,
which is implemented using the vsn R package.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple, Dict, Any

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_comparison_plot
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
    
    def normalize(self, X: Union[np.ndarray, List[List[float]]],
                 protein_ids: Optional[List[str]] = None,
                 sample_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Perform VSN normalization on input data X.
        
        Parameters
        ----------
        X : Union[np.ndarray, List[List[float]]]
            Input data matrix with shape (n_proteins, n_samples).
            Each column represents a sample, each row represents a protein.
        protein_ids : Optional[List[str]], optional
            Protein identifiers, by default None (uses row indices).
        sample_ids : Optional[List[str]], optional
            Sample identifiers, by default None (uses column indices).
        
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
            protein_ids = [f"Protein_{i}" for i in range(X.shape[0])]
        
        if sample_ids is None:
            sample_ids = [f"Sample_{i}" for i in range(X.shape[1])]
        
        # Create R script for VSN normalization
        r_script = self._create_vsn_script()
        
        try:
            # Run R script
            results = run_r_script(
                r_script,
                data=X,
                row_names=protein_ids,
                col_names=sample_ids
            )
            
            # Extract normalized data
            if 'normalized_data' in results:
                normalized_data = results['normalized_data']
            else:
                raise ValueError("VSN normalization failed to return normalized data")
            
            # Store VSN parameters
            if 'parameters' in results:
                self.vsn_params = results['parameters']
            
            return normalized_data
            
        except Exception as e:
            raise ValueError(f"VSN normalization failed: {str(e)}")
    
    def _create_vsn_script(self) -> str:
        """
        Create R script for VSN normalization.
        
        Returns
        -------
        str
            R script for VSN normalization.
        """
        # Set reference sample parameter
        ref_sample = "NULL"
        if self.reference_sample is not None:
            ref_sample = str(self.reference_sample + 1)  # R is 1-indexed
        
        script = """
        # Load required packages
        library(vsn)
        
        # Check if input_data exists in the environment
        if (!exists("input_data")) {
            stop("Input data not provided")
        }
        
        # Run VSN normalization
        vsn_fit <- vsn2(
            input_data,
            calib.strat = "{calib}",
            reference = {ref_sample}
        )
        
        # Get normalized data
        normalized_data <- predict(vsn_fit, input_data)
        
        # Extract parameters
        parameters <- list(
            coefficients = vsn_fit@coefficients,
            stdev = vsn_fit@stdev,
            reference = vsn_fit@reference,
            h_parameters = vsn_fit@h.parameters
        )
        """.format(
            calib=self.calib,
            ref_sample=ref_sample
        )
        
        return script
    
    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray, 
                       sample_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (15, 8),
                       title: str = "VSN Normalization Comparison") -> plt.Figure:
        """
        Plot data before vs after normalization.
        
        Parameters
        ----------
        before_data : np.ndarray
            Data before normalization, shape (n_proteins, n_samples).
        after_data : np.ndarray
            Data after normalization, shape (n_proteins, n_samples).
        sample_names : Optional[List[str]], optional
            Names for the samples, by default None (uses indices).
        figsize : Tuple[int, int], optional
            Figure size, by default (15, 8).
        title : str, optional
            Plot title, by default "VSN Normalization Comparison".
        
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
            before_label="Before VSN Normalization",
            after_label="After VSN Normalization",
            sample_names=sample_names,
        )
        
        # Add additional plot showing mean-variance relationship
        # This is a key diagnostic for VSN normalization
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate mean and standard deviation for each protein before normalization
        mean_before = np.nanmean(before_data, axis=1)
        std_before = np.nanstd(before_data, axis=1)
        
        # Calculate mean and standard deviation for each protein after normalization
        mean_after = np.nanmean(after_data, axis=1)
        std_after = np.nanstd(after_data, axis=1)
        
        # Plot mean vs standard deviation before normalization
        ax1.scatter(mean_before, std_before, alpha=0.5, s=5)
        ax1.set_title("Before VSN: Mean-Variance Relationship")
        ax1.set_xlabel("Mean")
        ax1.set_ylabel("Standard Deviation")
        
        # Plot mean vs standard deviation after normalization
        ax2.scatter(mean_after, std_after, alpha=0.5, s=5, color='green')
        ax2.set_title("After VSN: Mean-Variance Relationship")
        ax2.set_xlabel("Mean")
        ax2.set_ylabel("Standard Deviation")
        
        # Add a horizontal line at the median standard deviation after normalization
        median_std = np.nanmedian(std_after)
        ax2.axhline(median_std, color='red', linestyle='--', alpha=0.7)
        ax2.text(
            0.05, 0.95, 
            f"Median SD: {median_std:.3f}",
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        # Add explanation text
        fig2.suptitle("VSN Normalization Effect on Variance", fontsize=14)
        plt.figtext(
            0.5, 0.01,
            "VSN aims to stabilize variance across the intensity range.\n"
            "After successful normalization, the standard deviation should be\n"
            "approximately constant and independent of the mean.",
            ha='center', fontsize=10
        )
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        return fig, fig2
