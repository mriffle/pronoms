import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from pronoms.normalizers import MADNormalizer
from pronoms.utils import validate_input_data # Ensure this is imported if needed indirectly

# Sample data for testing (needs to be non-negative for log2)
@pytest.fixture
def data():
    # Sample data for testing (needs to be non-negative for log2)
    # Row 3: Should have non-zero MAD after log2(X+1)
    # Row 4: Should have zero MAD after log2(X+1) because log2(5+1) is constant
    return np.array([
        [1.,   2.,   3.,   4.,   5. ], # median=3, MAD=1
        [6.,   7.,   8.,   9.,  10. ], # median=8, MAD=1
        [1.,   2.,   3.,   4., 100.], # median=3, MAD=1 (approx)
        [5.,   5.,   5.,   5.,   5. ]  # median=5, MAD=0 -> log2(6), MAD=0
    ], dtype=float)

class TestMADNormalizer:

    @pytest.mark.parametrize("log_transform", [True, False])
    def test_normalize_basic(self, data, log_transform):
        """Test basic MAD normalization with and without log transform."""
        normalizer = MADNormalizer(log_transform=log_transform)
        data_subset = data[:3, :] # Exclude row with zero MAD

        if log_transform:
            # Expected values calculated manually based on log2(X+1)
            data_processed = np.log2(data_subset + 1)
        else:
            # Expected values calculated manually based on original data
            data_processed = data_subset

        # Calculate expected values based on the processed data
        row_medians = np.median(data_processed, axis=1, keepdims=True)
        abs_deviations = np.abs(data_processed - row_medians)
        row_mads = np.median(abs_deviations, axis=1, keepdims=True)

        # Handle potential division by zero if MAD is zero (though excluded here)
        # This is more for robustness if the test setup changes
        row_mads[row_mads == 0] = 1 # Avoid division by zero in test calc if fixture changes

        expected_output = (data_processed - row_medians) / row_mads

        # Actual normalization
        normalized_data = normalizer.normalize(data_subset)

        assert normalized_data.shape == data_subset.shape
        np.testing.assert_allclose(normalized_data, expected_output, rtol=1e-5)

        # Check stored state matches calculations on the processed data
        np.testing.assert_allclose(normalizer.row_medians, row_medians.flatten(), rtol=1e-5)
        # Recalculate MADs without the zero-replacement for assertion
        row_mads_assert = np.median(abs_deviations, axis=1)
        np.testing.assert_allclose(normalizer.row_mads, row_mads_assert, rtol=1e-5)
        assert normalizer.log_transform == log_transform

    @pytest.mark.parametrize(
        "log_transform, scale_type, expected_index",
        [(True, "log2\\(X\\+1\\)", 3), (False, "original", 3)]
    )
    def test_normalize_zero_mad(self, data, log_transform, scale_type, expected_index):
        """Test ValueError when MAD is zero (log or original)."""
        normalizer = MADNormalizer(log_transform=log_transform)
        # Row 3 has identical values, leading to MAD=0 on both original and log2(X+1) scales
        expected_error_msg = (
            f"Cannot normalize: MAD of {scale_type} data is zero for sample\\(s\\) "
            f"at index/indices: \\[{expected_index}\\]"
        )
        with pytest.raises(ValueError, match=expected_error_msg):
            normalizer.normalize(data)

    @pytest.mark.parametrize("log_transform", [True, False])
    def test_normalize_nan_inf(self, data, log_transform):
        """Test ValueError for NaN/Inf input."""
        normalizer = MADNormalizer(log_transform=log_transform)
        data_with_nan = data.copy()
        data_with_nan[0, 0] = np.nan
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values."):
            normalizer.normalize(data_with_nan)

        data_with_inf = data.copy()
        data_with_inf[0, 0] = np.inf
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values."):
            normalizer.normalize(data_with_inf)

    @pytest.mark.parametrize("log_transform", [True, False])
    def test_normalize_invalid_dim(self, log_transform):
        """Test error handling for invalid input dimensions."""
        normalizer = MADNormalizer(log_transform=log_transform)
        with pytest.raises(ValueError, match="must be a 2D array"):
            normalizer.normalize(np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="Input data cannot be empty"):
             normalizer.normalize(np.array([[], []]))

    # No parametrization needed - check happens only if log_transform=True
    def test_normalize_negative_values_log_true(self):
        """Test ValueError for negative values when log_transform=True."""
        normalizer = MADNormalizer(log_transform=True)
        data_neg = np.array([[1., 2.], [-1., 4.]])
        with pytest.raises(ValueError, match="Input data contains negative values. Log2 transformation cannot be applied."):
            normalizer.normalize(data_neg)

    def test_normalize_negative_values_log_false(self):
        """Test negative values are allowed when log_transform=False."""
        normalizer = MADNormalizer(log_transform=False)
        data_neg = np.array([[1., 2., 3.], [-1., -2., -3.]])
        # Calculate expected (median = 2, MAD = 1; median = -2, MAD = 1)
        expected = np.array([[-1., 0., 1.], [1., 0., -1.]])
        try:
            normalized = normalizer.normalize(data_neg)
            np.testing.assert_allclose(normalized, expected)
            # Check state
            np.testing.assert_allclose(normalizer.row_medians, [2.0, -2.0])
            np.testing.assert_allclose(normalizer.row_mads, [1.0, 1.0])
            assert not normalizer.log_transform
        except ValueError:
            pytest.fail("Normalization failed unexpectedly for negative values when log_transform=False")

    @pytest.mark.parametrize("log_transform", [True, False])
    @patch('pronoms.normalizers.mad_normalizer.create_hexbin_comparison')
    def test_plot_comparison(self, mock_create_hexbin, data, log_transform):
        """Test plot_comparison calls the plotting utility with correct params."""
        normalizer = MADNormalizer(log_transform=log_transform)
        # Need to run normalize first to set internal state (like log_transform)
        # Use valid subset
        data_subset = data[:3, :]
        normalized_data = normalizer.normalize(data_subset)

        # Setup mock figure (no spec needed)
        mock_fig = MagicMock()
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
        assert call_kwargs.get('xlabel') == "Original Data"
        expected_ylabel = f"After MAD Normalization ({'Standardized Log2 Scale' if log_transform else 'Standardized Original Scale'})"
        assert call_kwargs.get('ylabel') == expected_ylabel
        assert call_kwargs.get('autoscale_y') is True
        assert call_kwargs.get('add_identity_line') is False
        assert call_kwargs.get('add_center_line_y0') is True

        # Check returned figure instance is the mock
        assert fig is mock_fig
