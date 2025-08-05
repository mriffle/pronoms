"""
Tests for RankNormalizer.
"""

import numpy as np
import pytest
from unittest.mock import patch
from pronoms.normalizers import RankNormalizer


class TestRankNormalizer:
    """Test cases for RankNormalizer."""

    def test_init_default(self):
        """Test default initialization."""
        normalizer = RankNormalizer()
        assert normalizer.normalize_by_n is False
        assert normalizer.ranks is None

    def test_init_with_normalize_by_n(self):
        """Test initialization with normalize_by_n=True."""
        normalizer = RankNormalizer(normalize_by_n=True)
        assert normalizer.normalize_by_n is True
        assert normalizer.ranks is None

    def test_normalize_basic(self):
        """Test basic rank normalization."""
        # Simple test case: [3, 1, 2] should become [3, 1, 2] (ranks)
        X = np.array([[3.0, 1.0, 2.0]])
        normalizer = RankNormalizer()
        result = normalizer.normalize(X)
        
        expected = np.array([[3.0, 1.0, 2.0]])
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(normalizer.ranks, expected)

    def test_normalize_with_ties(self):
        """Test rank normalization with tied values."""
        # Test case: [1, 2, 2, 3] should become [1, 2.5, 2.5, 4]
        # Tied values at rank 2,3 get median rank 2.5
        X = np.array([[1.0, 2.0, 2.0, 3.0]])
        normalizer = RankNormalizer()
        result = normalizer.normalize(X)
        
        expected = np.array([[1.0, 2.5, 2.5, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_normalize_multiple_samples(self):
        """Test rank normalization with multiple samples."""
        X = np.array([
            [3.0, 1.0, 2.0],  # Should become [3, 1, 2]
            [5.0, 5.0, 1.0]   # Should become [2.5, 2.5, 1] (ties at rank 2,3)
        ])
        normalizer = RankNormalizer()
        result = normalizer.normalize(X)
        
        expected = np.array([
            [3.0, 1.0, 2.0],
            [2.5, 2.5, 1.0]
        ])
        np.testing.assert_array_equal(result, expected)

    def test_normalize_by_n_true(self):
        """Test rank normalization with normalize_by_n=True."""
        X = np.array([[3.0, 1.0, 2.0]])  # 3 features
        normalizer = RankNormalizer(normalize_by_n=True)
        result = normalizer.normalize(X)
        
        # Ranks [3, 1, 2] divided by 3 = [1.0, 0.333..., 0.666...]
        expected = np.array([[1.0, 1.0/3.0, 2.0/3.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_by_n_with_ties(self):
        """Test normalize_by_n with tied values."""
        X = np.array([[1.0, 2.0, 2.0, 3.0]])  # 4 features
        normalizer = RankNormalizer(normalize_by_n=True)
        result = normalizer.normalize(X)
        
        # Ranks [1, 2.5, 2.5, 4] divided by 4 = [0.25, 0.625, 0.625, 1.0]
        expected = np.array([[0.25, 0.625, 0.625, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_invalid_input_1d(self):
        """Test that 1D input raises ValueError."""
        X = np.array([1, 2, 3])
        normalizer = RankNormalizer()
        
        with pytest.raises(ValueError, match="X must be a 2D array"):
            normalizer.normalize(X)

    def test_normalize_invalid_input_empty_features(self):
        """Test that input with no features raises ValueError."""
        X = np.array([]).reshape(2, 0)
        normalizer = RankNormalizer()
        
        with pytest.raises(ValueError, match="X must be a 2D array"):
            normalizer.normalize(X)

    def test_normalize_nan_values(self):
        """Test that NaN values raise ValueError."""
        X = np.array([[1.0, np.nan, 3.0]])
        normalizer = RankNormalizer()
        
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values"):
            normalizer.normalize(X)

    def test_normalize_inf_values(self):
        """Test that Inf values raise ValueError."""
        X = np.array([[1.0, np.inf, 3.0]])
        normalizer = RankNormalizer()
        
        with pytest.raises(ValueError, match="Input data contains NaN or Inf values"):
            normalizer.normalize(X)

    def test_normalize_preserves_shape(self):
        """Test that normalization preserves input shape."""
        X = np.random.rand(5, 10)
        normalizer = RankNormalizer()
        result = normalizer.normalize(X)
        
        assert result.shape == X.shape

    def test_normalize_rank_range(self):
        """Test that ranks are in correct range."""
        X = np.random.rand(3, 7)  # 7 features
        normalizer = RankNormalizer()
        result = normalizer.normalize(X)
        
        # Each row should have ranks from 1 to 7
        for i in range(result.shape[0]):
            row_ranks = result[i, :]
            assert np.min(row_ranks) >= 1.0
            assert np.max(row_ranks) <= 7.0
            # Should contain all ranks from 1 to 7 (allowing for ties)
            unique_ranks = np.unique(row_ranks)
            assert len(unique_ranks) <= 7  # Can be less due to ties

    def test_normalize_rank_range_normalized(self):
        """Test that normalized ranks are in correct range."""
        X = np.random.rand(3, 7)  # 7 features
        normalizer = RankNormalizer(normalize_by_n=True)
        result = normalizer.normalize(X)
        
        # Each row should have ranks from 1/7 to 1
        for i in range(result.shape[0]):
            row_ranks = result[i, :]
            assert np.min(row_ranks) >= 1.0/7.0
            assert np.max(row_ranks) <= 1.0

    @patch('pronoms.normalizers.rank_normalizer.create_hexbin_comparison')
    def test_plot_comparison(self, mock_create_hexbin):
        """Test plot_comparison method with default log_x_axis=True."""
        normalizer = RankNormalizer()
        before_data = np.array([[1.0, 10.0, 100.0]])
        after_data = np.array([[1.0, 2.0, 3.0]])  # Ranks 1 to N
        
        normalizer.plot_comparison(before_data, after_data)

        # Check that the mock was called exactly once
        mock_create_hexbin.assert_called_once()

        # Get the arguments from the call
        call_args, call_kwargs = mock_create_hexbin.call_args

        # Expected log-transformed data (default behavior)
        expected_x_data = np.log10(before_data + 1)

        # Assert on the positional arguments (numpy arrays)
        np.testing.assert_allclose(call_args[0], expected_x_data)
        np.testing.assert_allclose(call_args[1], after_data)

        # Assert on the keyword arguments
        n_features = before_data.shape[1]
        x_min, x_max = np.min(expected_x_data), np.max(expected_x_data)
        padding = (x_max - x_min) * 0.05
        expected_xlim = (x_min - padding, x_max + padding)
        
        assert call_kwargs['figsize'] == (10, 8)
        assert call_kwargs['title'] == "Rank Normalization Comparison"
        assert call_kwargs['xlabel'] == "Log10(Original Value + 1)"
        assert call_kwargs['ylabel'] == f"Assigned Rank (1 to {n_features})"
        assert call_kwargs['log_axes'] is False
        np.testing.assert_allclose(call_kwargs['xlim'], expected_xlim)
        assert call_kwargs['ylim'] == (0, n_features + 1)
        assert call_kwargs['autoscale_y'] is False
        assert call_kwargs['add_identity_line'] is False

    @patch('pronoms.normalizers.rank_normalizer.create_hexbin_comparison')
    def test_plot_comparison_custom_params(self, mock_create_hexbin):
        """Test plot_comparison with custom parameters."""
        normalizer = RankNormalizer()
        before_data = np.array([[1.0, 10.0, 100.0]])
        after_data = np.array([[1.0, 2.0, 3.0]])
        
        normalizer.plot_comparison(
            before_data, 
            after_data,
            figsize=(12, 10),
            title="Custom Title",
            log_x_axis=False  # Test with raw x-axis values
        )
        
        # Check that the mock was called exactly once
        mock_create_hexbin.assert_called_once()

        # Get the arguments from the call
        call_args, call_kwargs = mock_create_hexbin.call_args

        # Expected raw data (no log transformation)
        expected_x_data = before_data

        # Assert on the positional arguments (numpy arrays)
        np.testing.assert_allclose(call_args[0], expected_x_data)
        np.testing.assert_allclose(call_args[1], after_data)

        # Assert on the keyword arguments
        n_features = before_data.shape[1]
        x_min, x_max = np.min(expected_x_data), np.max(expected_x_data)
        padding = (x_max - x_min) * 0.05
        expected_xlim = (x_min - padding, x_max + padding)
        
        assert call_kwargs['figsize'] == (12, 10)
        assert call_kwargs['title'] == "Custom Title"
        assert call_kwargs['xlabel'] == "Original Value"  # Raw values, not log
        assert call_kwargs['ylabel'] == f"Assigned Rank (1 to {n_features})"
        assert call_kwargs['log_axes'] is False
        np.testing.assert_allclose(call_kwargs['xlim'], expected_xlim)
        assert call_kwargs['ylim'] == (0, n_features + 1)
        assert call_kwargs['autoscale_y'] is False
        assert call_kwargs['add_identity_line'] is False

    @patch('pronoms.normalizers.rank_normalizer.create_hexbin_comparison')
    def test_plot_comparison_normalized_ranks(self, mock_create_hexbin):
        """Test plot_comparison with normalized ranks (normalize_by_n=True)."""
        normalizer = RankNormalizer(normalize_by_n=True)
        before_data = np.array([[1.0, 10.0, 100.0]])
        # Normalized ranks should be in [0, 1] range
        after_data = np.array([[0.25, 0.5, 0.75]])  # Ranks divided by N
        
        normalizer.plot_comparison(before_data, after_data)

        # Check that the mock was called exactly once
        mock_create_hexbin.assert_called_once()

        # Get the arguments from the call
        call_args, call_kwargs = mock_create_hexbin.call_args

        # Expected log-transformed data (default behavior)
        expected_x_data = np.log10(before_data + 1)

        # Assert on the positional arguments (numpy arrays)
        np.testing.assert_allclose(call_args[0], expected_x_data)
        np.testing.assert_allclose(call_args[1], after_data)

        # Assert on the keyword arguments
        x_min, x_max = np.min(expected_x_data), np.max(expected_x_data)
        padding = (x_max - x_min) * 0.05
        expected_xlim = (x_min - padding, x_max + padding)
        
        assert call_kwargs['figsize'] == (10, 8)
        assert call_kwargs['title'] == "Rank Normalization Comparison"
        assert call_kwargs['xlabel'] == "Log10(Original Value + 1)"
        assert call_kwargs['ylabel'] == "Normalized Rank (0 to 1)"  # Different for normalized ranks
        assert call_kwargs['log_axes'] is False
        np.testing.assert_allclose(call_kwargs['xlim'], expected_xlim)
        assert call_kwargs['ylim'] == (-0.05, 1.05)  # Range for normalized ranks
        assert call_kwargs['autoscale_y'] is False
        assert call_kwargs['add_identity_line'] is False

    def test_complex_tie_scenario(self):
        """Test complex scenario with multiple groups of ties."""
        # Test the specific example from the user's description
        # If we have values that would normally get ranks 100-104 and they're all tied,
        # they should all get rank 102 (median of 100,101,102,103,104)
        X = np.array([[1, 2, 2, 2, 2, 2, 3]])  # 5 tied values in the middle
        normalizer = RankNormalizer()
        result = normalizer.normalize(X)
        
        # Expected: [1, 4, 4, 4, 4, 4, 7] (median of ranks 2,3,4,5,6 is 4)
        expected = np.array([[1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0]])
        np.testing.assert_array_equal(result, expected)

    def test_all_same_values(self):
        """Test case where all values in a sample are the same."""
        X = np.array([[5.0, 5.0, 5.0, 5.0]])
        normalizer = RankNormalizer()
        result = normalizer.normalize(X)
        
        # All values should get the median rank: (1+2+3+4)/4 = 2.5
        expected = np.array([[2.5, 2.5, 2.5, 2.5]])
        np.testing.assert_array_equal(result, expected)
