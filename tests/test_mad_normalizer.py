import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from pronoms.normalizers.mad_normalizer import MADNormalizer

class TestMADNormalizer:
    """Test suite for MADNormalizer."""

    @pytest.fixture(scope="class")
    def data(self):
        """Provides common test data."""
        # Simple data where median and MAD are easy to calculate
        # Row 0: Median=3, MAD=median(|[1,2,3,4,5]-3|) = median(|[-2,-1,0,1,2]|) = median([0,1,1,2,2]) = 1
        # Row 1: Median=8, MAD=median(|[6,7,8,9,10]-8|) = median(|[-2,-1,0,1,2]|) = median([0,1,1,2,2]) = 1
        # Row 2: Median=3, MAD=median(|[1,2,3,4,100]-3|) = median(|[-2,-1,0,1,97]|) = median([0,1,1,2,97]) = 1
        # Row 3: All same, Median=5, MAD=0
        return np.array([
            [1., 2., 3., 4., 5.],
            [6., 7., 8., 9., 10.],
            [1., 2., 3., 4., 100.],
            [5., 5., 5., 5., 5.]
        ])

    @pytest.fixture(scope="class")
    def expected_medians(self):
        return np.array([3., 8., 3., 5.])

    @pytest.fixture(scope="class")
    def expected_mads(self):
        # MAD calculation: median(abs(x - median(x)))
        return np.array([1., 1., 1., 0.])

    @pytest.fixture(scope="class")
    def expected_normalized_data(self):
        # (X - median) / MAD
        # Row 0: ([1,2,3,4,5] - 3) / 1 = [-2, -1, 0, 1, 2]
        # Row 1: ([6,7,8,9,10] - 8) / 1 = [-2, -1, 0, 1, 2]
        # Row 2: ([1,2,3,4,100] - 3) / 1 = [-2, -1, 0, 1, 97]
        # Row 3: Will raise error due to MAD=0
        return np.array([
            [-2., -1., 0., 1., 2.],
            [-2., -1., 0., 1., 2.],
            [-2., -1., 0., 1., 97.]
            # Row 3 excluded as it causes error
        ])

    def test_init(self):
        """Test normalizer initialization."""
        normalizer = MADNormalizer()
        assert normalizer.row_medians is None
        assert normalizer.row_mads is None

    def test_normalize_logic(self, data, expected_normalized_data, expected_medians, expected_mads):
        """Test the core normalization logic."""
        normalizer = MADNormalizer()
        # Exclude the row with MAD=0 for this test
        data_subset = data[:3, :]
        normalized_data = normalizer.normalize(data_subset)

        np.testing.assert_allclose(normalized_data, expected_normalized_data, atol=1e-6)
        # Check stored state
        np.testing.assert_allclose(normalizer.row_medians, expected_medians[:3], atol=1e-6)
        np.testing.assert_allclose(normalizer.row_mads, expected_mads[:3], atol=1e-6)

    def test_normalize_zero_mad(self, data):
        """Test ValueError when MAD is zero."""
        normalizer = MADNormalizer()
        # Use raw string literal (r"") for the regex pattern to avoid SyntaxWarning
        expected_error_msg = r"Cannot normalize: MAD is zero for sample\(s\) at index/indices: \[3\]"
        with pytest.raises(ValueError, match=expected_error_msg):
            normalizer.normalize(data)

    def test_normalize_nan_inf(self, data):
        """Test ValueError for NaN/Inf input."""
        normalizer = MADNormalizer()
        data_nan = data.copy()
        data_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values."):
            normalizer.normalize(data_nan)

        data_inf = data.copy()
        data_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values."):
            normalizer.normalize(data_inf)

    def test_normalize_invalid_dim(self):
        """Test ValueError for invalid input dimensions."""
        normalizer = MADNormalizer()

        # Test 1D array
        with pytest.raises(ValueError, match="Input data must be a 2D array"):
            normalizer.normalize(np.array([1, 2, 3]))

        # Test empty 1D array (also caught by ndim != 2 check)
        with pytest.raises(ValueError, match="Input data must be a 2D array"):
            normalizer.normalize(np.array([]))

        # Test empty 2D array (caught by size == 0 check in validator)
        with pytest.raises(ValueError, match="Input data cannot be empty"):
             normalizer.normalize(np.array([[], []]))

    @patch('pronoms.normalizers.mad_normalizer.create_hexbin_comparison')
    def test_plot_comparison(self, mock_create_hexbin, data):
        """Test plot_comparison calls the plotting utility."""
        normalizer = MADNormalizer()
        # Need to run normalize first to have data for plotting (use valid subset)
        # Exclude the row with zero MAD for plotting test
        data_subset = data[:3, :]
        normalized_data = normalizer.normalize(data_subset)

        # Setup mock figure
        mock_fig = MagicMock(spec=plt.Figure)
        mock_create_hexbin.return_value = mock_fig

        # Call the plotting function
        fig = normalizer.plot_comparison(data_subset, normalized_data)

        # Verify mock call
        mock_create_hexbin.assert_called_once()
        call_args, call_kwargs = mock_create_hexbin.call_args
        np.testing.assert_array_equal(call_args[0], data_subset)
        np.testing.assert_array_equal(call_args[1], normalized_data)
        assert call_kwargs.get('figsize') == (10, 8)
        assert call_kwargs.get('title') == "MAD Normalization Comparison"
        assert call_kwargs.get('xlabel') == "Before MAD Normalization"
        assert call_kwargs.get('ylabel') == "After MAD Normalization"

        # Check returned figure
        assert fig == mock_fig
        plt.close(fig) # Close figure if mock didn't prevent creation
