"""
Tests for the L1Normalizer class.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import L1Normalizer


class TestL1Normalizer:
    """Test suite for L1Normalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        self.data = np.array([
            [10, 20, 30],
            [20, 40, 60],
            [30, 60, 90],
            [40, 80, 120],
            [50, 100, 150]
        ])
        
        # Calculate L1 norms for each column: [150, 300, 450]
        # After normalization, all columns should have L1 norm 1.0
        
        # Create normalizer
        self.normalizer = L1Normalizer()
    
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
        assert_allclose(self.normalizer.scaling_factors, [150, 300, 450])
        
        # Check that the L1 norms of normalized data are all 1.0
        l1_norms = np.sum(np.abs(normalized), axis=0)
        assert_allclose(l1_norms, [1.0, 1.0, 1.0], rtol=1e-10)
        
        # Check specific values
        expected = np.array([
            [10/150, 20/300, 30/450],
            [20/150, 40/300, 60/450],
            [30/150, 60/300, 90/450],
            [40/150, 80/300, 120/450],
            [50/150, 100/300, 150/450]
        ])
        assert_allclose(normalized, expected, rtol=1e-10)
    
    def test_normalize_pandas_dataframe(self):
        """Test normalization with pandas DataFrame input."""
        # Convert data to DataFrame
        df = pd.DataFrame(self.data, columns=['A', 'B', 'C'])
        
        # Normalize data
        normalized = self.normalizer.normalize(df)
        
        # Check that the result is a numpy array
        assert isinstance(normalized, np.ndarray)
        
        # Check that the shape is preserved
        assert normalized.shape == self.data.shape
        
        # Check that the L1 norms of normalized data are all 1.0
        l1_norms = np.sum(np.abs(normalized), axis=0)
        assert_allclose(l1_norms, [1.0, 1.0, 1.0], rtol=1e-10)
    
    def test_normalize_with_zeros(self):
        """Test normalization with zeros in the data."""
        # Create data with zeros
        data_with_zeros = np.array([
            [0, 0, 0],
            [10, 20, 30],
            [20, 40, 60]
        ])
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_zeros)
        
        # Check that the scaling factors were stored
        assert self.normalizer.scaling_factors is not None
        assert_allclose(self.normalizer.scaling_factors, [30, 60, 90])
        
        # Check specific values
        expected = np.array([
            [0/30, 0/60, 0/90],
            [10/30, 20/60, 30/90],
            [20/30, 40/60, 60/90]
        ])
        assert_allclose(normalized, expected, rtol=1e-10)
    
    def test_normalize_with_zero_column(self):
        """Test normalization with a column of all zeros."""
        # Create data with a column of all zeros
        data_with_zero_column = np.array([
            [0, 20, 30],
            [0, 40, 60],
            [0, 60, 90],
            [0, 80, 120],
            [0, 100, 150]
        ])
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_zero_column)
        
        # Check that the scaling factors were stored
        # The first column has L1 norm 0, but should be replaced with 1.0 to avoid division by zero
        assert self.normalizer.scaling_factors is not None
        assert_allclose(self.normalizer.scaling_factors, [0, 300, 450])
        
        # Check specific values for columns with non-zero L1 norms
        assert_allclose(normalized[:, 1], data_with_zero_column[:, 1] / 300, rtol=1e-10)
        assert_allclose(normalized[:, 2], data_with_zero_column[:, 2] / 450, rtol=1e-10)
        
        # For the column with zero L1 norm, the values should be equal to the original values
        # since we replace the zero L1 norm with 1.0
        assert_allclose(normalized[:, 0], data_with_zero_column[:, 0], rtol=1e-10)
    
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
    
    def test_normalize_with_negative_values(self):
        """Test normalization with negative values."""
        # Create data with negative values
        data_with_negatives = np.array([
            [-10, 20, -30],
            [20, -40, 60],
            [30, 60, -90]
        ])
        
        # Calculate expected L1 norms (sum of absolute values)
        l1_norms = np.sum(np.abs(data_with_negatives), axis=0)
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_negatives)
        
        # Check that the scaling factors were stored correctly
        assert_allclose(self.normalizer.scaling_factors, l1_norms, rtol=1e-10)
        
        # Check that the L1 norms of normalized data are all 1.0
        normalized_l1_norms = np.sum(np.abs(normalized), axis=0)
        assert_allclose(normalized_l1_norms, [1.0, 1.0, 1.0], rtol=1e-10)
    
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
