"""
Tests for the QuantileNormalizer class.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import QuantileNormalizer


class TestQuantileNormalizer:
    """Test suite for QuantileNormalizer."""

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
        
        # Create normalizer
        self.normalizer = QuantileNormalizer()
    
    def test_normalize_numpy_array(self):
        """Test normalization with numpy array input."""
        # Normalize data
        normalized = self.normalizer.normalize(self.data)
        
        # Check that the result is a numpy array
        assert isinstance(normalized, np.ndarray)
        
        # Check that the shape is preserved
        assert normalized.shape == self.data.shape
        
        # Check that the reference distribution was stored
        assert self.normalizer.reference_distribution is not None
        assert len(self.normalizer.reference_distribution) == self.data.shape[0]
        
        # Check that all columns have the same sorted values
        for i in range(self.data.shape[1]):
            assert_allclose(np.sort(normalized[:, i]), self.normalizer.reference_distribution, rtol=1e-10)
        
        # Check that the relative ordering within each column is preserved
        for i in range(self.data.shape[1]):
            original_order = np.argsort(np.argsort(self.data[:, i]))
            normalized_order = np.argsort(np.argsort(normalized[:, i]))
            assert_allclose(original_order, normalized_order, rtol=1e-10)
    
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
        
        # Check that all columns have the same sorted values
        for i in range(self.data.shape[1]):
            assert_allclose(np.sort(normalized[:, i]), self.normalizer.reference_distribution, rtol=1e-10)
    
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
        
        # Check that the reference distribution was stored
        assert self.normalizer.reference_distribution is not None
        assert len(self.normalizer.reference_distribution) == data_with_zeros.shape[0]
        
        # Check that all columns have the same sorted values
        for i in range(data_with_zeros.shape[1]):
            assert_allclose(np.sort(normalized[:, i]), self.normalizer.reference_distribution, rtol=1e-10)
    
    def test_normalize_with_identical_values(self):
        """Test normalization with identical values in a column."""
        # Create data with identical values in a column
        data_with_identical = np.array([
            [10, 20, 30],
            [10, 40, 60],
            [10, 60, 90]
        ])
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_identical)
        
        # Check that the reference distribution was stored
        assert self.normalizer.reference_distribution is not None
        
        # Check that all columns have the same sorted values
        for i in range(data_with_identical.shape[1]):
            assert_allclose(np.sort(normalized[:, i]), self.normalizer.reference_distribution, rtol=1e-10)
        
        # Check that identical values in the original data map to identical values in normalized data
        # For the first column, all values are identical, so they should all map to the same value
        assert_allclose(normalized[0, 0], normalized[1, 0], rtol=1e-10)
        assert_allclose(normalized[1, 0], normalized[2, 0], rtol=1e-10)
    
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
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_negatives)
        
        # Check that the reference distribution was stored
        assert self.normalizer.reference_distribution is not None
        
        # Check that all columns have the same sorted values
        for i in range(data_with_negatives.shape[1]):
            assert_allclose(np.sort(normalized[:, i]), self.normalizer.reference_distribution, rtol=1e-10)
    
    def test_normalize_different_shapes(self):
        """Test normalization with different data shapes."""
        # Create data with different shapes
        data_wide = np.random.rand(10, 5)  # 10 proteins, 5 samples
        data_tall = np.random.rand(20, 3)  # 20 proteins, 3 samples
        
        # Normalize data
        normalized_wide = self.normalizer.normalize(data_wide)
        
        # Check shapes
        assert normalized_wide.shape == data_wide.shape
        
        # Reset normalizer and normalize tall data
        self.normalizer = QuantileNormalizer()
        normalized_tall = self.normalizer.normalize(data_tall)
        
        # Check shapes
        assert normalized_tall.shape == data_tall.shape
        
        # Check that reference distributions have different lengths
        assert len(self.normalizer.reference_distribution) == data_tall.shape[0]
    
    def test_plot_comparison(self):
        """Test that plot_comparison returns figures."""
        # Normalize data
        normalized = self.normalizer.normalize(self.data)
        
        # Create plot
        result = self.normalizer.plot_comparison(self.data, normalized)
        
        # Check that the result contains figures
        assert result is not None
        
        # If result is a tuple, it should contain two figures
        if isinstance(result, tuple):
            fig, fig2 = result
            assert hasattr(fig, 'axes')
            assert hasattr(fig2, 'axes')
        else:
            # Otherwise, it should be a single figure
            assert hasattr(result, 'axes')
