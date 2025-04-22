"""
Tests for the MedianNormalizer class.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import MedianNormalizer


class TestMedianNormalizer:
    """Test suite for MedianNormalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        self.data = np.array([
            [10, 20, 30, 40, 50],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150]
        ])
        
        # Calculate medians for each row: [30, 60, 90]
        # After normalization, all rows should have median 1.0
        
        # Create normalizer
        self.normalizer = MedianNormalizer()
    
    def test_normalize_numpy_array(self):
        """Test normalization with numpy array input."""
        # Normalize data
        normalized = self.normalizer.normalize(self.data)
        
        # Check that the result is a numpy array
        assert isinstance(normalized, np.ndarray)
        
        # Check that the shape is preserved
        assert normalized.shape == self.data.shape
        
        # Check that the scaling factors were stored
        assert self.normalizer.scaling_factors is not None
        assert_allclose(self.normalizer.scaling_factors, [30, 60, 90])
        
        # Calculate the expected rescaled data
        medians = np.median(self.data, axis=1, keepdims=True)
        normalized_before_rescaling = self.data / medians
        mean_median = np.mean([30, 60, 90]) # 60
        expected_normalized = normalized_before_rescaling * mean_median

        # Check the normalized values against the expected rescaled values
        assert_allclose(normalized, expected_normalized, rtol=1e-10)
        
        # Check specific values
        expected = np.array([
            [10/30, 20/30, 30/30, 40/30, 50/30],
            [20/60, 40/60, 60/60, 80/60, 100/60],
            [30/90, 60/90, 90/90, 120/90, 150/90]
        ])
        expected = expected * mean_median
        assert_allclose(normalized, expected, rtol=1e-10)
    
    # test_normalize_pandas_dataframe removed as we now only support numpy arrays
    
    def test_normalize_with_zeros(self):
        """Test normalization with zeros in the data."""
        # Create data with zeros
        data_with_zeros = np.array([
            [0, 10, 20],
            [0, 20, 40],
            [0, 30, 60]
        ])
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_zeros)
        
        # Check that the scaling factors were stored
        assert self.normalizer.scaling_factors is not None
        assert_allclose(self.normalizer.scaling_factors, [10, 20, 30])
        
        # Calculate the expected rescaled data
        medians = np.median(data_with_zeros, axis=1, keepdims=True)
        normalized_before_rescaling = data_with_zeros / medians
        mean_median = np.mean([10, 20, 30]) # 20
        expected_normalized = normalized_before_rescaling * mean_median

        # Check the normalized values against the expected rescaled values
        assert_allclose(normalized, expected_normalized, rtol=1e-10)
        
        # Check specific values
        expected = np.array([
            [0/10, 10/10, 20/10],
            [0/20, 20/20, 40/20],
            [0/30, 30/30, 60/30]
        ])
        expected = expected * mean_median
        assert_allclose(normalized, expected, rtol=1e-10)
    
    def test_normalize_with_zero_row(self):
        """Test normalization with a row of all zeros should raise an error."""
        data_with_zero_row = np.array([
            [0, 0, 0, 0, 0],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150]
        ])
        with pytest.raises(ValueError, match="All sample medians must be > 0"):  # Zero median row
            self.normalizer.normalize(data_with_zero_row)
    
    def test_normalize_even_features(self):
        """Test normalization with an even number of features."""
        # Create data with an even number of features
        data_with_even_features = np.array([
            [10, 20, 30, 40],
            [20, 40, 60, 80],
            [30, 60, 90, 120]
        ])
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_even_features)
        
        # Check that the scaling factors were stored
        assert self.normalizer.scaling_factors is not None
        assert_allclose(self.normalizer.scaling_factors, [25, 50, 75])
        
        # Calculate the expected rescaled data
        medians = np.median(data_with_even_features, axis=1, keepdims=True)
        normalized_before_rescaling = data_with_even_features / medians
        mean_median = np.mean([25, 50, 75]) # 50
        expected_normalized = normalized_before_rescaling * mean_median

        # Check the normalized values against the expected rescaled values
        assert_allclose(normalized, expected_normalized, rtol=1e-10)
        
        # Check specific values
        expected = np.array([
            [10/25, 20/25, 30/25, 40/25],
            [20/50, 40/50, 60/50, 80/50],
            [30/75, 60/75, 90/75, 120/75]
        ])
        expected = expected * mean_median
        assert_allclose(normalized, expected, rtol=1e-10)
    
    def test_normalize_with_nan_values(self):
        """Test that normalization raises an error with NaN values."""
        # Create data with NaN values
        data_with_nan = np.array([
            [10, 20, 30],
            [20, np.nan, 60],
            [30, 60, 90]
        ])
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="NaN or Inf"):
            self.normalizer.normalize(data_with_nan)
    
    def test_normalize_with_inf_values(self):
        """Test that normalization raises an error with Inf values."""
        # Create data with Inf values
        data_with_inf = np.array([
            [10, 20, 30],
            [20, np.inf, 60],
            [30, 60, 90]
        ])
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="NaN or Inf"):
            self.normalizer.normalize(data_with_inf)
    
    def test_plot_comparison(self):
        """Test that plot_comparison returns a figure."""
        # Normalize data
        normalized = self.normalizer.normalize(self.data)
        
        # Create plot
        fig = self.normalizer.plot_comparison(self.data, normalized)
        
        # Check that the result is a matplotlib figure
        assert fig is not None
        
        # Check that scaling factors were included in the plot
        # (This is a bit hard to test directly, so we just check that the plot was created)
        assert hasattr(fig, 'axes')
        assert len(fig.axes) > 0
