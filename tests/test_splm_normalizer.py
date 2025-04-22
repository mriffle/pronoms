import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from pronoms.normalizers import SPLMNormalizer
# Import needed for mocking check, adjust if plotting utils change
from pronoms.utils.plotting import create_hexbin_comparison

class TestSPLMNormalizer:
    """Test suite for SPLMNormalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple dataset: 5 samples, 10 features
        # Features 0, 1, 2: Stable (low CV in log space)
        # Features 3, 4, 5: Moderate CV
        # Features 6, 7, 8, 9: High CV
        # Use values that result in distinct log-CVs to avoid ambiguity
        self.data = np.array([
            # Stable (around log(100) ~ 4.6) - low std dev
            [100, 101, 99,  50, 60, 70,  10, 200, 30, 500],  # Sample 1
            [102, 100, 98,  55, 65, 75,  15, 250, 35, 550],  # Sample 2
            [98,  99, 101, 45, 55, 65,  5,  150, 25, 450],  # Sample 3
            [101, 102, 100, 52, 62, 72,  12, 220, 32, 520],  # Sample 4
            [99,  98, 102, 48, 58, 68,  8,  180, 28, 480]   # Sample 5
        ], dtype=float)

        # Expected stable indices (0, 1, 2) if num_stable_proteins=3
        # These should have the lowest log-CVs
        self.expected_stable_indices_n3 = [0, 1, 2]

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        normalizer = SPLMNormalizer()
        assert normalizer.num_stable_proteins == 100
        assert normalizer.epsilon == 1e-6
        assert normalizer.stable_protein_indices is None
        assert normalizer.log_scaling_factors is None
        assert normalizer.grand_mean_log_scaling_factor is None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        normalizer = SPLMNormalizer(num_stable_proteins=50, epsilon=1e-5)
        assert normalizer.num_stable_proteins == 50
        assert normalizer.epsilon == 1e-5

    def test_init_invalid_params(self):
        """Test initialization raises ValueError for invalid parameters."""
        with pytest.raises(ValueError, match="num_stable_proteins must be a positive integer"):
            SPLMNormalizer(num_stable_proteins=0)
        with pytest.raises(ValueError, match="num_stable_proteins must be a positive integer"):
            SPLMNormalizer(num_stable_proteins=-10)
        with pytest.raises(ValueError, match="num_stable_proteins must be a positive integer"):
            SPLMNormalizer(num_stable_proteins=10.5)
        with pytest.raises(ValueError, match="epsilon must be a non-negative number"):
            SPLMNormalizer(epsilon=-1e-6)

    def test_normalize_shape_dtype(self):
        """Test output shape and dtype."""
        normalizer = SPLMNormalizer(num_stable_proteins=3)
        normalized_data = normalizer.normalize(self.data)
        assert normalized_data.shape == self.data.shape
        assert normalized_data.dtype == np.float64

    def test_normalize_nan_inf(self):
        """Test ValueError for NaN/Inf input."""
        normalizer = SPLMNormalizer()
        data_nan = self.data.copy()
        data_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values"):
            normalizer.normalize(data_nan)

        data_inf = self.data.copy()
        data_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values"):
            normalizer.normalize(data_inf)

    def test_normalize_too_few_proteins(self):
        """Test ValueError if num_stable_proteins > number of features."""
        normalizer = SPLMNormalizer(num_stable_proteins=self.data.shape[1] + 1)
        with pytest.raises(ValueError, match="cannot be greater than the number of features"):
            normalizer.normalize(self.data)

    def test_normalize_logic_n3(self):
        """Test the normalization logic with num_stable_proteins=3."""
        # Use a smaller number of stable proteins for easier verification
        n_stable = 3
        epsilon = 1e-6
        normalizer = SPLMNormalizer(num_stable_proteins=n_stable, epsilon=epsilon)
        normalized_data = normalizer.normalize(self.data)

        # 1. Check stored state
        assert normalizer.stable_protein_indices is not None
        # Check the correct indices were selected (should be 0, 1, 2 based on data setup)
        np.testing.assert_array_equal(np.sort(normalizer.stable_protein_indices),
                                       np.sort(self.expected_stable_indices_n3))
        assert normalizer.log_scaling_factors is not None
        assert normalizer.log_scaling_factors.shape == (self.data.shape[0],)
        assert normalizer.grand_mean_log_scaling_factor is not None

        # 2. Manual calculation for verification (simplified)
        X_log = np.log(self.data + epsilon)
        stable_log_data = X_log[:, self.expected_stable_indices_n3]
        expected_log_factors = np.mean(stable_log_data, axis=1)
        expected_grand_mean = np.mean(expected_log_factors)

        np.testing.assert_allclose(normalizer.log_scaling_factors, expected_log_factors, rtol=1e-5)
        assert abs(normalizer.grand_mean_log_scaling_factor - expected_grand_mean) < 1e-5

        # 3. Check output properties 
        # Check that the mean log intensity of stable proteins is constant across samples after normalization
        normalized_log = np.log(normalized_data + epsilon)
        norm_stable_log_data = normalized_log[:, normalizer.stable_protein_indices]
        norm_sample_means_stable = np.mean(norm_stable_log_data, axis=1)
        # These should all be very close to the grand mean log scaling factor
        np.testing.assert_allclose(norm_sample_means_stable, normalizer.grand_mean_log_scaling_factor, rtol=1e-5)

        # Check that non-negative values are returned
        assert np.all(normalized_data >= 0)

    def test_normalize_all_proteins_stable(self):
        """Test when num_stable_proteins equals the total number of features."""
        n_stable = self.data.shape[1]
        normalizer = SPLMNormalizer(num_stable_proteins=n_stable)
        normalized_data = normalizer.normalize(self.data)
        assert normalized_data.shape == self.data.shape
        # All indices should be selected
        assert len(normalizer.stable_protein_indices) == n_stable
        np.testing.assert_array_equal(np.sort(normalizer.stable_protein_indices), np.arange(n_stable))

    def test_normalize_constant_data(self):
        """Test with data where all proteins are constant (zero log-CV)."""
        constant_data = np.full_like(self.data, 100.0)
        n_stable = 3
        normalizer = SPLMNormalizer(num_stable_proteins=n_stable)
        # Expect near-identity transformation (up to epsilon effects and float precision)
        normalized_data = normalizer.normalize(constant_data)
        np.testing.assert_allclose(normalized_data, constant_data, rtol=1e-5, atol=1e-5)
        # Check that the first n_stable indices are selected as stable (since all have CV=0)
        assert len(normalizer.stable_protein_indices) == n_stable
        assert np.all(np.isin(normalizer.stable_protein_indices, np.arange(self.data.shape[1])))

    def test_normalize_zero_variance_protein(self):
        """Test data with a protein having zero variance (should have CV=0)."""
        data_with_const = self.data.copy()
        data_with_const[:, 4] = 50.0 # Make feature 4 constant
        normalizer = SPLMNormalizer(num_stable_proteins=1)
        normalized_data = normalizer.normalize(data_with_const)
        # The constant protein (index 4) should be selected as the most stable
        assert normalizer.stable_protein_indices[0] == 4

    @patch('pronoms.normalizers.splm_normalizer.create_hexbin_comparison')
    def test_plot_comparison(self, mock_create_hexbin):
        """Test plot_comparison method."""
        normalizer = SPLMNormalizer(num_stable_proteins=3)
        # Need to run normalize first to have data for plotting
        normalized_data = normalizer.normalize(self.data)

        mock_fig = MagicMock()
        mock_create_hexbin.return_value = mock_fig

        fig = normalizer.plot_comparison(self.data, normalized_data)

        mock_create_hexbin.assert_called_once()
        call_args, call_kwargs = mock_create_hexbin.call_args
        # Validate arguments passed to the plotting function
        np.testing.assert_array_equal(call_args[0], self.data)
        np.testing.assert_array_equal(call_args[1], normalized_data)
        assert call_kwargs.get('title') == "SPLM Normalization Comparison"
        assert call_kwargs.get('xlabel') == "Before SPLM Normalization"
        assert call_kwargs.get('ylabel') == "After SPLM Normalization"
        assert call_kwargs.get('figsize') == (10, 8) # Default figsize

        assert fig == mock_fig
