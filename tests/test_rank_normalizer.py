"""
Tests for RankNormalizer.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
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
    def test_plot_comparison_default(self, mock_create_hexbin):
        """Test plot_comparison with default parameters (log_axes=False)."""
        normalizer = RankNormalizer()
        before_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        after_data = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        
        # Mock the return value
        mock_fig = MagicMock()
        mock_create_hexbin.return_value = mock_fig
        
        # Call the method
        result = normalizer.plot_comparison(before_data, after_data)
        
        # Verify the mock was called
        mock_create_hexbin.assert_called_once()
        args, kwargs = mock_create_hexbin.call_args
        
        # Check that raw data was passed (log_axes=False by default)
        np.testing.assert_allclose(args[0], before_data)
        np.testing.assert_allclose(args[1], after_data)
        
        # Check other parameters
        assert kwargs['title'] == "Rank Normalization Comparison"
        assert kwargs['xlabel'] == "Original Value"
        assert kwargs['ylabel'] == "Assigned Rank (1 to 3)"
        assert kwargs['log_axes'] == False
        assert kwargs['autoscale_y'] == True
        assert kwargs['add_identity_line'] == False
        
        # Check that xlim and ylim were set
        assert 'xlim' in kwargs
        assert 'ylim' in kwargs
        assert kwargs['ylim'] == (0, 4)  # n_features + 1
        
        assert result == mock_fig

    @patch('pronoms.normalizers.rank_normalizer.create_hexbin_comparison')
    def test_plot_comparison_log_axes_true(self, mock_create_hexbin):
        """Test plot_comparison with log_axes=True."""
        normalizer = RankNormalizer()
        before_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        after_data = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        
        # Mock the return value
        mock_fig = MagicMock()
        mock_create_hexbin.return_value = mock_fig
        
        # Call the method with log_axes=True
        result = normalizer.plot_comparison(
            before_data, after_data, 
            figsize=(12, 10), 
            title="Custom Title",
            log_axes=True
        )
        
        # Verify the mock was called
        mock_create_hexbin.assert_called_once()
        args, kwargs = mock_create_hexbin.call_args
        
        # Check that log-transformed data was passed (log_axes=True)
        expected_x_data = np.log10(before_data + 1)
        np.testing.assert_allclose(args[0], expected_x_data)
        np.testing.assert_allclose(args[1], after_data)
        
        # Check other parameters
        assert kwargs['figsize'] == (12, 10)
        assert kwargs['title'] == "Custom Title"
        assert kwargs['xlabel'] == "Log10(Original Value + 1)"
        assert kwargs['ylabel'] == "Assigned Rank (1 to 3)"
        assert kwargs['log_axes'] == False  # Always False since data is pre-transformed
        assert kwargs['autoscale_y'] == True
        assert kwargs['add_identity_line'] == False
        
        assert result == mock_fig

    @patch('pronoms.normalizers.rank_normalizer.create_hexbin_comparison')
    def test_plot_comparison_log_axes_false(self, mock_create_hexbin):
        """Test plot_comparison with log_axes=False explicitly."""
        normalizer = RankNormalizer()
        before_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        after_data = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        
        # Mock the return value
        mock_fig = MagicMock()
        mock_create_hexbin.return_value = mock_fig
        
        # Call the method with log_axes=False explicitly
        result = normalizer.plot_comparison(
            before_data, after_data, 
            log_axes=False
        )
        
        # Verify the mock was called
        mock_create_hexbin.assert_called_once()
        args, kwargs = mock_create_hexbin.call_args
        
        # Check that raw data was passed (log_axes=False)
        np.testing.assert_allclose(args[0], before_data)
        np.testing.assert_allclose(args[1], after_data)
        
        # Check other parameters
        assert kwargs['title'] == "Rank Normalization Comparison"
        assert kwargs['xlabel'] == "Original Value"
        assert kwargs['ylabel'] == "Assigned Rank (1 to 3)"
        assert kwargs['log_axes'] == False
        assert kwargs['autoscale_y'] == True
        assert kwargs['add_identity_line'] == False
        
        assert result == mock_fig

    @patch('pronoms.normalizers.rank_normalizer.create_hexbin_comparison')
    def test_plot_comparison_normalized_ranks(self, mock_create_hexbin):
        """Test plot_comparison with normalized ranks (normalize_by_n=True)."""
        normalizer = RankNormalizer(normalize_by_n=True)
        before_data = np.array([[1.0, 10.0, 100.0]])
        # Normalized ranks should be in [0, 1] range
        after_data = np.array([[0.25, 0.5, 0.75]])  # Ranks divided by N
        
        # Mock the return value
        mock_fig = MagicMock()
        mock_create_hexbin.return_value = mock_fig
        
        result = normalizer.plot_comparison(before_data, after_data)

        # Check that the mock was called exactly once
        mock_create_hexbin.assert_called_once()

        # Get the arguments from the call
        call_args, call_kwargs = mock_create_hexbin.call_args

        # Assert on the positional arguments (raw data by default, log_axes=False)
        np.testing.assert_allclose(call_args[0], before_data)
        np.testing.assert_allclose(call_args[1], after_data)

        # Assert on the keyword arguments
        # Note: Uses n_features for y-axis limits
        n_features = before_data.shape[1]
        assert call_kwargs['figsize'] == (10, 8)
        assert call_kwargs['title'] == "Rank Normalization Comparison"
        assert call_kwargs['xlabel'] == "Original Value"  # Raw data by default
        assert call_kwargs['ylabel'] == f"Assigned Rank (1 to {n_features})"
        assert call_kwargs['log_axes'] is False
        assert 'xlim' in call_kwargs
        assert call_kwargs['ylim'] == (0, n_features + 1)
        assert call_kwargs['autoscale_y'] is True
        assert call_kwargs['add_identity_line'] is False
        
        assert result == mock_fig

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
