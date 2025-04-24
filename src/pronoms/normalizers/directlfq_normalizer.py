"""
DirectLFQ Normalizer for proteomics data.

This module provides a class for DirectLFQ normalization of proteomics data,
using the directlfq library.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import directlfq.config as dlcfg
import directlfq.utils   as dlu
import directlfq.normalization as dlnorm
import directlfq.protein_intensity_estimation as dlprot
from typing import Optional, List, Tuple

from ..utils.validators import validate_input_data, check_nan_inf
from ..utils.plotting import create_hexbin_comparison


class DirectLFQNormalizer:
    """
    Normalizer using the DirectLFQ algorithm for in-memory processing.

    This normalizer wraps the external `directlfq` library to perform
    intensity normalization directly on NumPy arrays without intermediate
    file I/O. It processes peptide-level data to produce normalized
    protein-level and peptide-level intensities.

    Parameters
    ----------
    do_between_sample_norm : bool, optional
        Whether to perform between-sample normalization (median centering
        based on selected stable proteins), by default True.
    n_quad_samples : int, optional
        Number of samples used for quadratic stabilization during
        between-sample normalization, by default 50.
    n_quad_ions : int, optional
        Number of ions used for quadratic stabilization during protein
        intensity estimation, by default 10.
    min_nonan : int, optional
        Minimum number of non-NaN values required per protein for its
        intensity to be estimated, by default 1.
    num_cores : int | None, optional
        Number of CPU cores to use for parallel processing in directlfq.
        If None, directlfq attempts to use all available cores, by default None.

    Attributes
    ----------
    do_between_sample_norm : bool
        Flag indicating if between-sample normalization is enabled.
    n_quad_samples : int
        Number of samples for quadratic stabilization (sample norm).
    n_quad_ions : int
        Number of ions for quadratic stabilization (protein estimation).
    min_nonan : int
        Minimum non-NaN values required per protein.
    num_cores : Optional[int]
        Number of cores used by directlfq.
    """
    def __init__(self,
                 do_between_sample_norm: bool = True,
                 n_quad_samples: int = 50,
                 n_quad_ions: int = 10,
                 min_nonan: int = 1,
                 num_cores: int | None = None):
        self.do_between_sample_norm = do_between_sample_norm
        self.n_quad_samples = n_quad_samples
        self.n_quad_ions = n_quad_ions
        self.min_nonan = min_nonan
        self.num_cores = num_cores

    def normalize(
        self,
        X: np.ndarray,
        proteins: list[str],
        peptides: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run DirectLFQ on the given peptide-level intensity matrix in memory.

        This method orchestrates the DirectLFQ workflow:
        1. Constructs a DataFrame in the format required by `directlfq`.
        2. Applies preprocessing steps (log transform, sorting, NaN removal).
        3. Optionally performs between-sample normalization.
        4. Estimates protein intensities.
        5. Extracts normalized protein and ion matrices and their corresponding IDs.

        Parameters
        ----------
        X : np.ndarray
            Input data matrix with shape (n_samples, n_features), where features
            typically represent peptides or ions.
        proteins : list[str]
            List of protein identifiers corresponding to each feature (column) in X.
            The length must equal `X.shape[1]`.
        peptides : list[str]
            List of peptide or ion identifiers corresponding to each feature (column)
            in X. The length must equal `X.shape[1]`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing four NumPy arrays:
          - protein_matrix: Normalized protein intensities (shape: n_samples, n_proteins).
          - ion_matrix: Normalized peptide/ion intensities (shape: n_samples, n_peptides).
          - protein_ids: Array of unique protein identifiers corresponding to the
            columns of `protein_matrix` (shape: n_proteins,).
          - peptide_ids: Array of unique peptide/ion identifiers corresponding to the
            columns of `ion_matrix` (shape: n_peptides,).

        Raises
        ------
        ValueError
            - If input `X` is not 2-dimensional.
            - If lengths of `proteins` or `peptides` do not match `X.shape[1]`.
            - If `X` contains NaN or infinite values.
            - If internal DataFrame processing or ID extraction fails.
        ImportError
            If the 'directlfq' library is not installed.
        """
        # ----------------- Input validation -----------------
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array (samples × features).")
        if len(proteins) != X.shape[1] or len(peptides) != X.shape[1]:
            raise ValueError("Lengths of 'proteins' and 'peptides' must equal n_features.")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("DirectLFQ cannot handle NaN or Inf values.")

        # ----------------- Construct DataFrame ---------------
        n_samples, _ = X.shape
        sample_cols = [f"sample_{i+1}" for i in range(n_samples)]
        df = pd.DataFrame({
            "protein": proteins,
            "ion": peptides,
            **{sample_cols[i]: X[i, :] for i in range(n_samples)}
        })

        # ----------------- DirectLFQ Configuration -----------
        dlcfg.set_global_protein_and_ion_id(protein_id="protein", quant_id="ion")
        dlcfg.check_wether_to_copy_numpy_arrays_derived_from_pandas()

        # ----------------- Preprocessing ---------------------
        df = dlu.sort_input_df_by_protein_and_quant_id(df)
        df = dlu.index_and_log_transform_input_df(df)
        df = dlu.remove_allnan_rows_input_df(df)

        if self.do_between_sample_norm:
            df = dlnorm.NormalizationManagerSamplesOnSelectedProteins(
                df,
                num_samples_quadratic=self.n_quad_samples,
                selected_proteins_file=None,
            ).complete_dataframe

        # ----------------- Protein inference -----------------
        prot_df, ion_df = dlprot.estimate_protein_intensities(
            df,
            min_nonan=self.min_nonan,
            num_samples_quadratic=self.n_quad_ions,
            num_cores=self.num_cores,
        )

        # ----------------- Extract IDs -----------------------
        protein_ids = (
            prot_df["protein"].to_numpy(dtype=str, copy=False)
            if "protein" in prot_df.columns
            else np.array(prot_df.index, dtype=str)
        )
        peptide_ids = (
            ion_df["ion"].to_numpy(dtype=str, copy=False)
            if "ion" in ion_df.columns
            else ion_df.index.get_level_values("ion").to_numpy(dtype=str)
        )

        # ----------------- Drop ID columns -------------------
        prot_numeric = prot_df.drop(columns=["protein"], errors="ignore")
        ion_numeric  = ion_df.drop(columns=["protein", "ion"], errors="ignore")

        # ----------------- To NumPy --------------------------
        protein_matrix = prot_numeric.T.to_numpy(dtype=np.float64, copy=False)
        ion_matrix     = ion_numeric.T.to_numpy(dtype=np.float64, copy=False)

        # ----------------- Sanity check ----------------------
        if ion_matrix.shape[1] != peptide_ids.shape[0]:
            raise ValueError("Ion matrix shape does not match number of returned peptide IDs.")

        return protein_matrix, ion_matrix, protein_ids, peptide_ids

    def plot_comparison(self, before_data: np.ndarray, after_data: np.ndarray,
                       figsize: Tuple[int, int] = (10, 8),
                       title: str = "DirectLFQ Protein Normalization Comparison") -> plt.Figure:
        """
        Plot protein data before vs after DirectLFQ normalization using a hexbin plot.

        Note: This plots the *protein* level intensities. DirectLFQ computes these
        from the input peptide/ion intensities.

        Parameters
        ----------
        before_data : np.ndarray
            Protein intensity data *before* normalization, shape (n_samples, n_proteins).
            This needs to be calculated/provided separately if the input to
            `normalize` was peptide-level.
        after_data : np.ndarray
            Normalized protein intensity data *after* normalization, shape (n_samples, n_proteins).
            Typically the first element returned by the `normalize` method.
        figsize : Tuple[int, int], optional
            Figure size, by default (10, 8).
        title : str, optional
            Plot title, by default "DirectLFQ Protein Normalization Comparison".

        Returns
        -------
        plt.Figure
            Figure object containing the hexbin density plot.
        """
        # Validate input data
        before_data = validate_input_data(before_data)
        after_data = validate_input_data(after_data)

        if before_data.shape != after_data.shape:
             print(f"Warning: Shape mismatch in plot_comparison: before={before_data.shape}, after={after_data.shape}. Plotting may be misleading.")

        # Create hexbin comparison plot
        fig = create_hexbin_comparison(
            before_data,
            after_data,
            figsize=figsize,
            title=title,
            xlabel="Before DirectLFQ (Protein Intensity)",
            ylabel="After DirectLFQ (Protein Intensity)"
        )

        return fig
