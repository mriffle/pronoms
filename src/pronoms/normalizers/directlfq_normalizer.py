"""
DirectLFQ Normalizer for proteomics data.

This module provides a class for DirectLFQ normalization of proteomics data,
which is implemented using the DirectLFQ R package.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple, Dict, Any

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_comparison_plot
from ..utils.r_interface import setup_r_environment, run_r_script


class DirectLFQNormalizer:
    """
    Normalizer that performs DirectLFQ normalization using the R package.
    
    DirectLFQ is a label-free quantification method for proteomics data that
    performs normalization and missing value imputation.
    
    Attributes
    ----------
    normalization_stats : Optional[Dict[str, Any]]
        Statistics from the normalization process.
        Only available after calling normalize().
    """
    
    def __init__(self, impute_missing: bool = True, min_peptides: int = 2):
        """
        Initialize the DirectLFQNormalizer.
        
        Parameters
        ----------
        impute_missing : bool, optional
            Whether to impute missing values, by default True.
        min_peptides : int, optional
            Minimum number of peptides required for protein quantification, by default 2.
        """
        self.impute_missing = impute_missing
        self.min_peptides = min_peptides
        self.normalization_stats = None
        
        # Check R environment and required packages
        self._check_r_dependencies()
    
    def _check_r_dependencies(self):
        """Check if required R packages are installed."""
        try:
            setup_r_environment(["DirectLFQ"])
        except Exception as e:
            print(f"Warning: {str(e)}")
            print("DirectLFQ normalization will not be available.")
    
    def normalize(self, X: Union[np.ndarray, List[List[float]]],
                 protein_ids: Optional[List[str]] = None,
                 sample_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Perform DirectLFQ normalization on input data X.
        
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
            If input data contains Inf values or if R integration fails.
        """
        # Validate input data
        X = validate_input_data(X)
        
        # Check for Inf values (NaN values are allowed for DirectLFQ)
        has_inf = np.any(np.isinf(X))
        if has_inf:
            raise ValueError(
                "Input data contains Inf values. Please handle these values before normalization."
            )
        
        # Generate default IDs if not provided
        if protein_ids is None:
            protein_ids = [f"Protein_{i}" for i in range(X.shape[0])]
        
        if sample_ids is None:
            sample_ids = [f"Sample_{i}" for i in range(X.shape[1])]
        
        # Create R script for DirectLFQ normalization
        r_script = self._create_directlfq_script()
        
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
                raise ValueError("DirectLFQ normalization failed to return normalized data")
            
            # Store normalization statistics
            if 'stats' in results:
                self.normalization_stats = results['stats']
            
            return normalized_data
            
        except Exception as e:
            raise ValueError(f"DirectLFQ normalization failed: {str(e)}")
    
    def _create_directlfq_script(self) -> str:
        """
        Create R script for DirectLFQ normalization.
        
        Returns
        -------
        str
            R script for DirectLFQ normalization.
        """
        script = """
        # Load required packages
        library(DirectLFQ)
        
        # Check if input_data exists in the environment
        if (!exists("input_data")) {
            stop("Input data not provided")
        }
        
        # Convert any NaN to NA for R compatibility
        input_data[is.nan(input_data)] <- NA
        
        # Run DirectLFQ normalization
        result <- directLFQ(
            x = input_data,
            annotation = NULL,  # No additional annotation
            measure.cols = 1:ncol(input_data),
            id.cols = NULL,  # No ID columns
            peptide.cols = NULL,  # No peptide columns
            protein.col = NULL,  # No protein column
            tolerance = 0.001,
            max.pep.per.prot = 100,
            min.pep.per.prot = {min_peptides},
            min.common.peptides = 1,
            filter.proteins = TRUE,
            norm = TRUE,
            req.obs = 1,
            impute = {impute}
        )
        
        # Extract normalized data
        normalized_data <- result$estimate
        
        # Extract statistics
        stats <- list(
            num_proteins = nrow(result$estimate),
            num_samples = ncol(result$estimate),
            missing_values_before = sum(is.na(input_data)),
            missing_values_after = sum(is.na(result$estimate)),
            normalization_factors = result$norm.factors
        )
        """.format(
            min_peptides=self.min_peptides,
            impute="TRUE" if self.impute_missing else "FALSE"
        )
        
        return script
    
    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray, 
                       sample_names: Optional[List[str]] = None,
                       figsize: Tuple[int, int] = (15, 8),
                       title: str = "DirectLFQ Normalization Comparison") -> plt.Figure:
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
            Plot title, by default "DirectLFQ Normalization Comparison".
        
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
            before_label="Before DirectLFQ Normalization",
            after_label="After DirectLFQ Normalization",
            sample_names=sample_names,
        )
        
        # If normalization stats are available, add them to the plot
        if self.normalization_stats is not None:
            # Extract statistics
            stats = self.normalization_stats
            
            # Create text for statistics
            stats_text = "DirectLFQ Statistics:\n"
            
            if 'normalization_factors' in stats:
                stats_text += "Normalization Factors:\n"
                for i, factor in enumerate(stats['normalization_factors']):
                    sample_name = f"Sample {i}" if sample_names is None else sample_names[i]
                    stats_text += f"  {sample_name}: {factor:.3f}\n"
            
            if 'missing_values_before' in stats and 'missing_values_after' in stats:
                stats_text += f"\nMissing Values:\n"
                stats_text += f"  Before: {stats['missing_values_before']}\n"
                stats_text += f"  After: {stats['missing_values_after']}\n"
                
                if stats['missing_values_before'] > 0:
                    pct_imputed = 100 * (1 - stats['missing_values_after'] / stats['missing_values_before'])
                    stats_text += f"  Imputed: {pct_imputed:.1f}%\n"
            
            # Add text box with statistics
            plt.figtext(
                0.01, 0.01, stats_text, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
            )
        
        return fig
