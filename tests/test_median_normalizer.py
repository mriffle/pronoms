"""
Tests for the MedianNormalizer class.
"""

import numpy as np
import pandas as pd
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
        
        # Check that the medians of normalized data are all 1.0
        medians = np.median(normalized, axis=1)
        assert_allclose(medians, [1.0, 1.0, 1.0], rtol=1e-10)
        
        # Check specific values
        expected = np.array([
            [10/30, 20/30, 30/30, 40/30, 50/30],
            [20/60, 40/60, 60/60, 80/60, 100/60],
            [30/90, 60/90, 90/90, 120/90, 150/90]
        ])
        assert_allclose(normalized, expected, rtol=1e-10)
    
    def test_normalize_pandas_dataframe(self):
        """Test normalization with pandas DataFrame input."""
        # Convert data to DataFrame
        df = pd.DataFrame(self.data, columns=['A', 'B', 'C', 'D', 'E'])
        
        # Normalize data
        normalized = self.normalizer.normalize(df)
        
        # Check that the result is a numpy array
        assert isinstance(normalized, np.ndarray)
        
        # Check that the shape is preserved
        assert normalized.shape == self.data.shape
        
        # Check that the medians of normalized data are all 1.0
        medians = np.median(normalized, axis=1)
        assert_allclose(medians, [1.0, 1.0, 1.0], rtol=1e-10)
    
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
        
        # Check specific values
        expected = np.array([
            [0/10, 10/10, 20/10],
            [0/20, 20/20, 40/20],
            [0/30, 30/30, 60/30]
        ])
        assert_allclose(normalized, expected, rtol=1e-10)
    
    def test_normalize_with_zero_row(self):
        """Test normalization with a row of all zeros."""
        # Create data with a row having zero median
        data_with_zero_row = np.array([
            [0, 0, 0, 0, 0],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150]
        ])
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_zero_row)
        
        # Check that the scaling factors were stored
        # The first row has median 0, but should be replaced with 1.0 to avoid division by zero
        assert self.normalizer.scaling_factors is not None
        assert_allclose(self.normalizer.scaling_factors, [1, 60, 90])
        
        # Check specific values for rows with non-zero medians
        assert_allclose(normalized[1, :], data_with_zero_row[1, :] / 60, rtol=1e-10)
        assert_allclose(normalized[2, :], data_with_zero_row[2, :] / 90, rtol=1e-10)
        
        # For the row with zero median, the values should be equal to the original values
        # since we replace the zero median with 1.0
        assert_allclose(normalized[0, :], data_with_zero_row[0, :], rtol=1e-10)
    
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
