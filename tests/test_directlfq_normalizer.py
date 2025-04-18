"""
Tests for the DirectLFQNormalizer class.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from pronoms.normalizers import DirectLFQNormalizer
from pronoms.utils.r_interface import RInterfaceError


class TestDirectLFQNormalizer:
    """Test suite for DirectLFQNormalizer."""

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
        
        # Sample names
        self.sample_names = ['Sample1', 'Sample2', 'Sample3']
        
        # Protein IDs
        self.protein_ids = ['Protein1', 'Protein2', 'Protein3', 'Protein4', 'Protein5']
        
        # Mock normalization stats
        self.mock_stats = {
            'num_proteins': 5,
            'num_samples': 3,
            'missing_values_before': 0,
            'missing_values_after': 0,
            'normalization_factors': [0.9, 1.0, 1.1]
        }
    
    @pytest.mark.parametrize("impute_missing", [True, False])
    def test_init(self, impute_missing):
        """Test initialization with different parameters."""
        # Create normalizer with different parameters
        normalizer = DirectLFQNormalizer(impute_missing=impute_missing, min_peptides=3)
        
        # Check that parameters were stored
        assert normalizer.impute_missing == impute_missing
        assert normalizer.min_peptides == 3
        assert normalizer.normalization_stats is None
    
    @patch('pronoms.normalizers.directlfq_normalizer.setup_r_environment')
    def test_check_r_dependencies(self, mock_setup):
        """Test R dependencies check."""
        # Create normalizer
        normalizer = DirectLFQNormalizer()
        
        # Check that setup_r_environment was called with the correct package
        mock_setup.assert_called_once_with(['DirectLFQ'])
        
        # Test with setup_r_environment raising an exception
        mock_setup.side_effect = RInterfaceError("R package not found")
        
        # This should not raise an exception, just print a warning
        normalizer = DirectLFQNormalizer()
    
    @patch('pronoms.normalizers.directlfq_normalizer.run_r_script')
    def test_normalize(self, mock_run):
        """Test normalization with mocked R interface."""
        # Mock the result of run_r_script
        mock_result = {
            'normalized_data': np.array([
                [11, 20, 27],
                [22, 40, 55],
                [33, 60, 82],
                [44, 80, 109],
                [55, 100, 136]
            ]),
            'stats': self.mock_stats
        }
        mock_run.return_value = mock_result
        
        # Create normalizer
        normalizer = DirectLFQNormalizer()
        
        # Normalize data
        normalized = normalizer.normalize(
            self.data,
            protein_ids=self.protein_ids,
            sample_ids=self.sample_names
        )
        
        # Check that run_r_script was called
        mock_run.assert_called_once()
        
        # Check that the result is the mocked normalized data
        np.testing.assert_array_equal(normalized, mock_result['normalized_data'])
        
        # Check that normalization stats were stored
        assert normalizer.normalization_stats == self.mock_stats
    
    @patch('pronoms.normalizers.directlfq_normalizer.run_r_script')
    def test_normalize_with_default_ids(self, mock_run):
        """Test normalization with default IDs."""
        # Mock the result of run_r_script
        mock_run.return_value = {
            'normalized_data': np.array([
                [11, 20, 27],
                [22, 40, 55],
                [33, 60, 82],
                [44, 80, 109],
                [55, 100, 136]
            ]),
            'stats': self.mock_stats
        }
        
        # Create normalizer
        normalizer = DirectLFQNormalizer()
        
        # Normalize data without providing IDs
        normalized = normalizer.normalize(self.data)
        
        # Check that run_r_script was called with default IDs
        args, kwargs = mock_run.call_args
        assert 'row_names' in kwargs
        assert 'col_names' in kwargs
        assert len(kwargs['row_names']) == 5  # Number of proteins
        assert len(kwargs['col_names']) == 3  # Number of samples
    
    @patch('pronoms.normalizers.directlfq_normalizer.run_r_script')
    def test_normalize_with_inf_values(self, mock_run):
        """Test that normalization raises an error with Inf values."""
        # Create data with Inf values
        data_with_inf = np.array([
            [10, 20, 30],
            [20, np.inf, 60],
            [30, 60, 90]
        ])
        
        # Create normalizer
        normalizer = DirectLFQNormalizer()
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="Inf values"):
            normalizer.normalize(data_with_inf)
        
        # Check that run_r_script was not called
        mock_run.assert_not_called()
    
    @patch('pronoms.normalizers.directlfq_normalizer.run_r_script')
    def test_normalize_with_nan_values(self, mock_run):
        """Test normalization with NaN values (should be allowed)."""
        # Create data with NaN values
        data_with_nan = np.array([
            [10, 20, 30],
            [20, np.nan, 60],
            [30, 60, 90]
        ])
        
        # Mock the result of run_r_script
        mock_run.return_value = {
            'normalized_data': np.array([
                [11, 20, 27],
                [22, np.nan, 55],
                [33, 60, 82]
            ]),
            'stats': {
                'num_proteins': 3,
                'num_samples': 3,
                'missing_values_before': 1,
                'missing_values_after': 1,
                'normalization_factors': [0.9, 1.0, 1.1]
            }
        }
        
        # Create normalizer
        normalizer = DirectLFQNormalizer()
        
        # Normalize data with NaN values
        normalized = normalizer.normalize(data_with_nan)
        
        # Check that run_r_script was called
        mock_run.assert_called_once()
    
    @patch('pronoms.normalizers.directlfq_normalizer.run_r_script')
    def test_normalize_with_run_r_script_error(self, mock_run):
        """Test that normalization handles run_r_script errors."""
        # Mock run_r_script to raise an exception
        mock_run.side_effect = Exception("R script error")
        
        # Create normalizer
        normalizer = DirectLFQNormalizer()
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="DirectLFQ normalization failed"):
            normalizer.normalize(self.data)
    
    @patch('pronoms.normalizers.directlfq_normalizer.run_r_script')
    def test_normalize_with_missing_result(self, mock_run):
        """Test that normalization handles missing result."""
        # Mock run_r_script to return a result without normalized_data
        mock_run.return_value = {'stats': self.mock_stats}
        
        # Create normalizer
        normalizer = DirectLFQNormalizer()
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="failed to return normalized data"):
            normalizer.normalize(self.data)
    
    @patch('matplotlib.pyplot.Figure')
    @patch('pronoms.normalizers.directlfq_normalizer.create_comparison_plot')
    def test_plot_comparison(self, mock_create_plot, mock_figure):
        """Test plot_comparison method."""
        # Mock the result of create_comparison_plot
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_create_plot.return_value = (mock_fig, mock_axes)
        
        # Create normalizer with mock stats
        normalizer = DirectLFQNormalizer()
        normalizer.normalization_stats = self.mock_stats
        
        # Call plot_comparison
        fig = normalizer.plot_comparison(self.data, self.data)
        
        # Check that create_comparison_plot was called
        mock_create_plot.assert_called_once()
        
        # Check that the result is the mocked figure
        assert fig == mock_fig
