"""
Tests for the L1Normalizer class.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import L1Normalizer


class TestL1Normalizer:
    """Test suite for L1Normalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple test dataset
        self.data = np.array([
            [10, 20, 30, 40, 50],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150]
        ])
        
        # Calculate L1 norms for each row: [150, 300, 450]
        # After normalization, all rows should have L1 norm 1.0
        
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
        
        # Calculate the expected rescaled data
        l1_norms = np.sum(np.abs(self.data), axis=1, keepdims=True)
        normalized_before_rescaling = self.data / l1_norms
        mean_l1_norm = np.mean([150, 300, 450]) # 300
        expected_normalized = normalized_before_rescaling * mean_l1_norm

        # Check the normalized values against the expected rescaled values
        assert_allclose(normalized, expected_normalized, rtol=1e-10)
    
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
        assert_allclose(self.normalizer.scaling_factors, [30, 60, 90])
        
        # Calculate the expected rescaled data
        l1_norms = np.sum(np.abs(data_with_zeros), axis=1, keepdims=True)
        normalized_before_rescaling = data_with_zeros / l1_norms
        mean_l1_norm = np.mean([30, 60, 90]) # 60
        expected_normalized = normalized_before_rescaling * mean_l1_norm

        # Check the normalized values against the expected rescaled values
        assert_allclose(normalized, expected_normalized, rtol=1e-10)
    
    def test_normalize_with_zero_row(self):
        """Test normalization with a row of all zeros."""
        # Create data with a row of all zeros
        data_with_zero_row = np.array([
            [0, 0, 0, 0, 0],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150]
        ])
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_zero_row)
        
        # Check that the scaling factors were stored
        # The first row has L1 norm 0, but should be replaced with 1.0 to avoid division by zero
        assert self.normalizer.scaling_factors is not None
        assert_allclose(self.normalizer.scaling_factors, [1, 300, 450])
        
        # Calculate the expected rescaled data
        l1_norms = np.sum(np.abs(data_with_zero_row), axis=1, keepdims=True)
        # Manually handle division by zero for the first row
        l1_norms_safe = np.where(l1_norms == 0, 1, l1_norms)
        normalized_before_rescaling = data_with_zero_row / l1_norms_safe
        mean_l1_norm = np.mean([1, 300, 450]) # 583/3
        expected_normalized = normalized_before_rescaling * mean_l1_norm
        # The zero row should remain zero after normalization and rescaling
        expected_normalized[0, :] = 0.0

        # Check the normalized values against the expected rescaled values
        assert_allclose(normalized, expected_normalized, rtol=1e-10)
        
        # Check that the first row (originally all zeros) remains all zeros
        assert_allclose(normalized[0, :], [0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-10)
    
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
        l1_norms = np.sum(np.abs(data_with_negatives), axis=1)
        
        # Normalize data
        normalized = self.normalizer.normalize(data_with_negatives)
        
        # Check that the scaling factors were stored correctly
        assert_allclose(self.normalizer.scaling_factors, l1_norms, rtol=1e-10)
        
        # Calculate the expected rescaled data
        normalized_before_rescaling = data_with_negatives / l1_norms[:, np.newaxis]
        mean_l1_norm = np.mean(l1_norms) # 60
        expected_normalized = normalized_before_rescaling * mean_l1_norm

        # Check the normalized values against the expected rescaled values
        assert_allclose(normalized, expected_normalized, rtol=1e-10)
    
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
