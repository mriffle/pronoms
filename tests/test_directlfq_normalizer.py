"""
Unit tests for the DirectLFQ Normalizer.
"""

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from unittest.mock import patch, MagicMock, call
import tempfile
import os

# Module to test
from pronoms.normalizers.directlfq_normalizer import DirectLFQNormalizer

# --- Fixtures ---

@pytest.fixture
def raw_data():
    """Provides sample raw intensity data (samples x features)."""
    # 3 samples, 5 features (peptides/ions)
    return np.array([
        [10, 20, 30, 40, 50],
        [12, 22, 32, 42, 52],
        [ 8, 18, 28, 38, 48]
    ], dtype=float)

@pytest.fixture
def protein_list():
    """Provides sample protein names for features."""
    # Corresponding to 5 features
    return ['ProtA', 'ProtA', 'ProtB', 'ProtC', 'ProtC']

@pytest.fixture
def peptide_list():
    """Provides sample peptide/ion names for features."""
    # Corresponding to 5 features
    return ['PepA1', 'PepA2', 'PepB1', 'PepC1', 'PepC2']

@pytest.fixture
def mock_directlfq_output_protein():
    """Mock DataFrame for protein output from directlfq."""
    data = {'sample_1': [100.0, 300.0], 'sample_2': [120.0, 320.0], 'sample_3': [80.0, 280.0]}
    index = ['ProtA', 'ProtC']
    return pd.DataFrame(data, index=index)

@pytest.fixture
def mock_directlfq_output_ion():
    """Mock DataFrame for ion output from directlfq."""
    data = {'sample_1': [10.0, 20.0, 40.0, 50.0],
            'sample_2': [12.0, 22.0, 42.0, 52.0],
            'sample_3': [8.0, 18.0, 38.0, 48.0]}
    index = ['PepA1', 'PepA2', 'PepC1', 'PepC2']
    return pd.DataFrame(data, index=index)

# --- Test Class ---

class TestDirectLFQNormalizer:
    """
    Test suite for the DirectLFQNormalizer.
    """

    def test_initialization(self):
        """Test DirectLFQNormalizer initialization with default and custom cores."""
        # Default initialization
        normalizer_default = DirectLFQNormalizer()
        assert normalizer_default.num_cores is None
        assert normalizer_default.do_between_sample_norm is True
        assert normalizer_default.min_nonan == 1
        assert normalizer_default.n_quad_ions == 10
        assert normalizer_default.n_quad_samples == 50

        # Initialization with specific cores
        normalizer_cores = DirectLFQNormalizer(num_cores=4)
        assert normalizer_cores.num_cores == 4

        # Initialization with custom parameters
        normalizer_custom = DirectLFQNormalizer(
            num_cores=2,
            do_between_sample_norm=False,
            min_nonan=2,
            n_quad_ions=5,
            n_quad_samples=20
        )
        assert normalizer_custom.num_cores == 2
        assert normalizer_custom.do_between_sample_norm is False
        assert normalizer_custom.min_nonan == 2
        assert normalizer_custom.n_quad_ions == 5
        assert normalizer_custom.n_quad_samples == 20

    def test_normalize_invalid_input_shapes(self, raw_data, protein_list, peptide_list):
        """Test normalize with inconsistent input shapes."""
        normalizer = DirectLFQNormalizer()

        # Incorrect protein list length
        with pytest.raises(
            ValueError,
            match="Lengths of 'proteins' and 'peptides' must equal n_features."
        ):
            normalizer.normalize(raw_data, protein_list[:-1], peptide_list)

        # Incorrect peptide list length
        with pytest.raises(
            ValueError,
            match="Lengths of 'proteins' and 'peptides' must equal n_features."
        ):
            normalizer.normalize(raw_data, protein_list, peptide_list[:-1])

        # 1D array
        with pytest.raises(ValueError, match=r"X must be a 2-D array \(samples × features\)\."):
            normalizer.normalize(raw_data[0, :], protein_list, peptide_list)

    def test_normalize_nan_inf_input(self, protein_list, peptide_list):
        """Test normalize with NaN or Inf values in input."""
        normalizer = DirectLFQNormalizer()

        # NaN input
        X_nan = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]])
        proteins_nan = protein_list[:3]
        peptides_nan = peptide_list[:3]
        with pytest.raises(ValueError, match="DirectLFQ cannot handle NaN or Inf"):
            normalizer.normalize(X_nan, proteins_nan, peptides_nan)

        # Inf input
        X_inf = np.array([[1.0, np.inf, 3.0], [4.0, 5.0, 6.0]])
        proteins_inf = protein_list[:3]
        peptides_inf = peptide_list[:3]
        with pytest.raises(ValueError, match="DirectLFQ cannot handle NaN or Inf"):
            normalizer.normalize(X_inf, proteins_inf, peptides_inf)

    @patch('pronoms.normalizers.directlfq_normalizer.dlcfg.set_global_protein_and_ion_id')
    @patch('pronoms.normalizers.directlfq_normalizer.dlcfg.check_wether_to_copy_numpy_arrays_derived_from_pandas')
    @patch('pronoms.normalizers.directlfq_normalizer.dlu.index_and_log_transform_input_df')
    @patch('pronoms.normalizers.directlfq_normalizer.dlu.sort_input_df_by_protein_and_quant_id')
    @patch('pronoms.normalizers.directlfq_normalizer.dlu.remove_allnan_rows_input_df')
    @patch('pronoms.normalizers.directlfq_normalizer.dlnorm.NormalizationManagerSamplesOnSelectedProteins')
    @patch('pronoms.normalizers.directlfq_normalizer.dlprot.estimate_protein_intensities')
    def test_normalize_success(self,
                             mock_estimate_protein_intensities,
                             mock_norm_manager,
                             mock_remove_nan_rows,
                             mock_sort_df,
                             mock_log_transform,
                             mock_check_copy,
                             mock_set_global_ids,
                              raw_data, protein_list, peptide_list,
                              mock_directlfq_output_protein, mock_directlfq_output_ion):
        """Test successful normalization flow with mocked dependencies."""
        # Setup mocks
        # Construct the expected DataFrame that normalize will create
        n_samples, _ = raw_data.shape
        sample_cols = [f"sample_{i+1}" for i in range(n_samples)]
        df_expected_input = pd.DataFrame({
            'protein': protein_list,
            'ion': peptide_list,
            **{s: raw_data[i, :] for i, s in enumerate(sample_cols)}
        })

        # Mock the directlfq function return values
        # Preprocessing steps return the initial df (now without index)
        # Ensure order matches normalize: sort -> log_transform -> remove_nan
        mock_sort_df.return_value = df_expected_input       # sort is called first
        mock_log_transform.return_value = df_expected_input # log_transform called second
        mock_remove_nan_rows.return_value = df_expected_input # remove_nan called third
        mock_norm_instance = MagicMock()
        mock_norm_instance.complete_dataframe = df_expected_input # Result of sample norm
        mock_norm_manager.return_value = mock_norm_instance

        # Mock estimate_protein_intensities to return DFs *with* IDs
        # These should mimic the structure *before* IDs are dropped in normalize()
        mock_protein_df_with_ids = mock_directlfq_output_protein.reset_index()
        mock_protein_df_with_ids.rename(columns={'index': 'protein'}, inplace=True)
        mock_ion_df_with_ids = mock_directlfq_output_ion.reset_index()
        mock_ion_df_with_ids.rename(columns={'index': 'ion'}, inplace=True)
        # Need to add the protein column back to ion_df as estimate_protein_intensities includes it
        protein_map = df_expected_input.set_index('ion')['protein']
        mock_ion_df_with_ids['protein'] = mock_ion_df_with_ids['ion'].map(protein_map)
        # Reorder columns to match typical directlfq output before processing
        ion_cols = ['protein', 'ion'] + [c for c in mock_ion_df_with_ids.columns if c not in ['protein', 'ion']]
        mock_ion_df_with_ids = mock_ion_df_with_ids[ion_cols]

        mock_estimate_protein_intensities.return_value = (
            mock_protein_df_with_ids,
            mock_ion_df_with_ids
        )

        # Instantiate normalizer with default params
        normalizer = DirectLFQNormalizer(num_cores=4) # Use fixed cores for testing

        # Run normalization
        norm_prot, norm_ion, ret_prot_ids, ret_pep_ids = normalizer.normalize(
            raw_data, protein_list, peptide_list
        )

        # --- Assertions ---
        # 1. Config functions called
        mock_set_global_ids.assert_called_once_with(protein_id='protein', quant_id='ion')
        mock_check_copy.assert_called_once()

        # 2. Preprocessing called in correct order: sort -> log_transform -> remove_nan
        mock_sort_df.assert_called_once()
        assert_frame_equal(mock_sort_df.call_args[0][0], df_expected_input) # Called with initial df

        mock_log_transform.assert_called_once()
        # Called with result of sort (which is mocked to be df_expected_input)
        assert_frame_equal(mock_log_transform.call_args[0][0], df_expected_input)

        mock_remove_nan_rows.assert_called_once()
        # Called with result of log_transform (mocked to be df_expected_input)
        assert_frame_equal(mock_remove_nan_rows.call_args[0][0], df_expected_input)

        # 3. NormalizationManager called correctly
        call_args_norm = mock_norm_manager.call_args
        assert_frame_equal(call_args_norm[0][0], df_expected_input) # Called with result of remove_nan
        assert call_args_norm.kwargs['num_samples_quadratic'] == 50 # Default
        assert call_args_norm.kwargs['selected_proteins_file'] is None

        # 4. Assert estimate_protein_intensities called correctly
        call_args_est = mock_estimate_protein_intensities.call_args
        assert_frame_equal(call_args_est[0][0], df_expected_input)
        assert call_args_est.kwargs['min_nonan'] == 1 # Default
        assert call_args_est.kwargs['num_samples_quadratic'] == 10 # Default n_quad_ions
        assert call_args_est.kwargs['num_cores'] == 4 # From initializer

        # 5. Check final outputs
        expected_norm_prot = mock_directlfq_output_protein.T.to_numpy(dtype=np.float64)
        expected_norm_ion = mock_directlfq_output_ion.T.to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(norm_prot, expected_norm_prot)
        np.testing.assert_array_equal(norm_ion, expected_norm_ion)

        # 6. Check returned IDs
        expected_prot_ids = np.array(['ProtA', 'ProtC'], dtype=str)
        expected_pep_ids = np.array(['PepA1', 'PepA2', 'PepC1', 'PepC2'], dtype=str)
        np.testing.assert_array_equal(ret_prot_ids, expected_prot_ids)
        np.testing.assert_array_equal(ret_pep_ids, expected_pep_ids)

    @patch('pronoms.normalizers.directlfq_normalizer.dlcfg.set_global_protein_and_ion_id')
    @patch('pronoms.normalizers.directlfq_normalizer.dlcfg.check_wether_to_copy_numpy_arrays_derived_from_pandas')
    @patch('pronoms.normalizers.directlfq_normalizer.dlu.index_and_log_transform_input_df')
    @patch('pronoms.normalizers.directlfq_normalizer.dlu.sort_input_df_by_protein_and_quant_id')
    @patch('pronoms.normalizers.directlfq_normalizer.dlu.remove_allnan_rows_input_df')
    @patch('pronoms.normalizers.directlfq_normalizer.dlnorm.NormalizationManagerSamplesOnSelectedProteins')
    @patch('pronoms.normalizers.directlfq_normalizer.dlprot.estimate_protein_intensities')
    def test_normalize_success_custom_params(self,
                             mock_estimate_protein_intensities,
                             mock_norm_manager,
                             mock_remove_nan_rows,
                             mock_sort_df,
                             mock_log_transform,
                             mock_check_copy,
                             mock_set_global_ids,
                              raw_data, protein_list, peptide_list,
                              mock_directlfq_output_protein, mock_directlfq_output_ion):
        """Test successful normalization flow with non-default parameters."""
        # Setup mocks (similar to test_normalize_success)
        n_samples, _ = raw_data.shape
        sample_cols = [f"sample_{i+1}" for i in range(n_samples)]
        df_expected_input = pd.DataFrame({
            'protein': protein_list,
            'ion': peptide_list,
            **{s: raw_data[i, :] for i, s in enumerate(sample_cols)}
        })

        # Ensure order matches normalize: sort -> log_transform -> remove_nan
        mock_sort_df.return_value = df_expected_input       # sort is called first
        mock_log_transform.return_value = df_expected_input # log_transform called second
        mock_remove_nan_rows.return_value = df_expected_input # remove_nan called third
        mock_norm_instance = MagicMock()
        mock_norm_instance.complete_dataframe = df_expected_input # Not expected to be called here
        mock_norm_manager.return_value = mock_norm_instance # Not expected to be called here

        # Mock estimate_protein_intensities to return DFs *with* IDs
        mock_protein_df_with_ids = mock_directlfq_output_protein.reset_index()
        mock_protein_df_with_ids.rename(columns={'index': 'protein'}, inplace=True)
        mock_ion_df_with_ids = mock_directlfq_output_ion.reset_index()
        mock_ion_df_with_ids.rename(columns={'index': 'ion'}, inplace=True)
        protein_map = df_expected_input.set_index('ion')['protein']
        mock_ion_df_with_ids['protein'] = mock_ion_df_with_ids['ion'].map(protein_map)
        ion_cols = ['protein', 'ion'] + [c for c in mock_ion_df_with_ids.columns if c not in ['protein', 'ion']]
        mock_ion_df_with_ids = mock_ion_df_with_ids[ion_cols]

        mock_estimate_protein_intensities.return_value = (
            mock_protein_df_with_ids,
            mock_ion_df_with_ids
        )

        # Instantiate with non-default parameters
        custom_params = {
            "num_cores": 2,
            "do_between_sample_norm": False,
            "min_nonan": 3,
            "n_quad_ions": 7,
            "n_quad_samples": 25 # This won't be used as do_between_sample_norm=False
        }
        normalizer = DirectLFQNormalizer(**custom_params)

        # Run normalization
        norm_prot, norm_ion, ret_prot_ids, ret_pep_ids = normalizer.normalize(
            raw_data, protein_list, peptide_list
        )

        # --- Assertions for non-default behavior ---
        # 1. Config functions called (same as default)
        mock_set_global_ids.assert_called_once_with(protein_id='protein', quant_id='ion')
        mock_check_copy.assert_called_once()

        # 2. Preprocessing called (order: sort -> log_transform -> remove_nan)
        mock_sort_df.assert_called_once()
        assert_frame_equal(mock_sort_df.call_args[0][0], df_expected_input)
        mock_log_transform.assert_called_once()
        assert_frame_equal(mock_log_transform.call_args[0][0], df_expected_input)
        mock_remove_nan_rows.assert_called_once()
        assert_frame_equal(mock_remove_nan_rows.call_args[0][0], df_expected_input)

        # 3. NormalizationManager NOT called
        mock_norm_manager.assert_not_called()

        # 4. Estimate protein intensities called with correct non-default params
        #    Input DF should be the result of mock_remove_nan_rows directly (no index)
        call_args_est = mock_estimate_protein_intensities.call_args
        assert_frame_equal(call_args_est[0][0], df_expected_input)
        assert call_args_est.kwargs['min_nonan'] == custom_params['min_nonan']
        assert call_args_est.kwargs['num_samples_quadratic'] == custom_params['n_quad_ions'] # Uses n_quad_ions
        assert call_args_est.kwargs['num_cores'] == custom_params['num_cores']

        # 5. Check final outputs (same as default for this mock data)
        expected_norm_prot = mock_directlfq_output_protein.T.to_numpy(dtype=np.float64)
        expected_norm_ion = mock_directlfq_output_ion.T.to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(norm_prot, expected_norm_prot)
        np.testing.assert_array_equal(norm_ion, expected_norm_ion)

        # 6. Check returned IDs (same as default for this mock data)
        expected_prot_ids = np.array(['ProtA', 'ProtC'], dtype=str)
        expected_pep_ids = np.array(['PepA1', 'PepA2', 'PepC1', 'PepC2'], dtype=str)
        np.testing.assert_array_equal(ret_prot_ids, expected_prot_ids)
        np.testing.assert_array_equal(ret_pep_ids, expected_pep_ids)

    @patch('pronoms.normalizers.directlfq_normalizer.create_hexbin_comparison')
    def test_plot_comparison(self, mock_plot_util, raw_data):
        """Test the plot_comparison method calls the plotting utility."""
        normalizer = DirectLFQNormalizer()

        # Prepare some dummy data matching expected protein output shape
        # Use shapes derived from fixtures for consistency
        n_samples = raw_data.shape[0]
        n_proteins = 2 # Based on mock_directlfq_output_protein fixture
        before_protein = np.random.rand(n_samples, n_proteins)
        after_protein = np.random.rand(n_samples, n_proteins)

        # Custom title and figsize
        custom_title = "My DirectLFQ Plot"
        custom_figsize = (12, 6)

        # Call the method
        fig = normalizer.plot_comparison(
            before_protein,
            after_protein,
            title=custom_title,
            figsize=custom_figsize
        )

        # Assert the underlying plotting utility was called correctly
        mock_plot_util.assert_called_once()
        call_args = mock_plot_util.call_args
        assert call_args is not None
        
        pos_args = call_args.args
        kw_args = call_args.kwargs

        np.testing.assert_array_equal(pos_args[0], before_protein)
        np.testing.assert_array_equal(pos_args[1], after_protein)
        assert kw_args['figsize'] == custom_figsize
        assert kw_args['title'] == custom_title
        assert 'xlabel' in kw_args and kw_args['xlabel'] == "Before DirectLFQ (Protein Intensity)"
        assert 'ylabel' in kw_args and kw_args['ylabel'] == "After DirectLFQ (Protein Intensity)"
        # The returned figure should be the mock object itself
        assert isinstance(fig, MagicMock)

    @patch('pronoms.normalizers.directlfq_normalizer.create_hexbin_comparison')
    @patch('builtins.print') # Mock print to check warning
    def test_plot_comparison_shape_mismatch(self, mock_print, mock_plot_util):
        """Test plot_comparison with shape mismatch (should still plot but warn)."""
        normalizer = DirectLFQNormalizer()
        before = np.random.rand(3, 5)
        after = np.random.rand(3, 4) # Mismatched feature count

        normalizer.plot_comparison(before, after)

        # Check if warning printed - use pytest.approx for flexibility with exact string format
        mock_print.assert_called_once_with('Warning: Shape mismatch in plot_comparison: before=(3, 5), after=(3, 4). Plotting may be misleading.')

        # Check if plotting utility was called (even with mismatch)
        mock_plot_util.assert_called_once()
