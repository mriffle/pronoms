"""
Tests for the VSNNormalizer class.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from pronoms.normalizers import VSNNormalizer
from pronoms.utils.r_interface import RInterfaceError


class TestVSNNormalizer:
    """Test suite for VSNNormalizer."""

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
        
        # Mock VSN parameters
        self.mock_params = {
            'coefficients': [1.0, 0.5, 0.2],
            'stdev': 0.1,
            'reference': 1,
            'h_parameters': [0.01, 0.001]
        }
    
    @pytest.mark.parametrize("calib", ["affine", "none", "shift", "maximum"])
    def test_init(self, calib):
        """Test initialization with different parameters."""
        # Create normalizer with different parameters
        normalizer = VSNNormalizer(calib=calib, reference_sample=1)
        
        # Check that parameters were stored
        assert normalizer.calib == calib
        assert normalizer.reference_sample == 1
        assert normalizer.vsn_params is None
    
    @patch('pronoms.normalizers.vsn_normalizer.setup_r_environment')
    def test_check_r_dependencies(self, mock_setup):
        """Test R dependencies check."""
        # Create normalizer
        normalizer = VSNNormalizer()
        
        # Check that setup_r_environment was called with the correct package
        mock_setup.assert_called_once_with(['vsn'])
        
        # Test with setup_r_environment raising an exception
        mock_setup.side_effect = RInterfaceError("R package not found")
        
        # This should not raise an exception, just print a warning
        normalizer = VSNNormalizer()
    
    @patch('pronoms.normalizers.vsn_normalizer.run_r_script')
    def test_normalize(self, mock_run):
        """Test normalization with mocked R interface."""
        # Mock the result of run_r_script
        mock_result = {
            'normalized_data': np.array([
                [2.3, 3.0, 3.4],
                [3.0, 3.7, 4.1],
                [3.4, 4.1, 4.5],
                [3.7, 4.4, 4.8],
                [3.9, 4.6, 5.0]
            ]),
            'parameters': self.mock_params
        }
        mock_run.return_value = mock_result
        
        # Create normalizer
        normalizer = VSNNormalizer()
        
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
        
        # Check that VSN parameters were stored
        assert normalizer.vsn_params == self.mock_params
    
    @patch('pronoms.normalizers.vsn_normalizer.run_r_script')
    def test_normalize_with_default_ids(self, mock_run):
        """Test normalization with default IDs."""
        # Mock the result of run_r_script
        mock_run.return_value = {
            'normalized_data': np.array([
                [2.3, 3.0, 3.4],
                [3.0, 3.7, 4.1],
                [3.4, 4.1, 4.5],
                [3.7, 4.4, 4.8],
                [3.9, 4.6, 5.0]
            ]),
            'parameters': self.mock_params
        }
        
        # Create normalizer
        normalizer = VSNNormalizer()
        
        # Normalize data without providing IDs
        normalized = normalizer.normalize(self.data)
        
        # Check that run_r_script was called with default IDs
        args, kwargs = mock_run.call_args
        assert 'row_names' in kwargs
        assert 'col_names' in kwargs
        assert len(kwargs['row_names']) == 5  # Number of proteins
        assert len(kwargs['col_names']) == 3  # Number of samples
    
    @patch('pronoms.normalizers.vsn_normalizer.run_r_script')
    def test_normalize_with_nan_inf_values(self, mock_run):
        """Test that normalization raises an error with NaN or Inf values."""
        # Create data with NaN values
        data_with_nan = np.array([
            [10, 20, 30],
            [20, np.nan, 60],
            [30, 60, 90]
        ])
        
        # Create normalizer
        normalizer = VSNNormalizer()
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="NaN or Inf"):
            normalizer.normalize(data_with_nan)
        
        # Check that run_r_script was not called
        mock_run.assert_not_called()
        
        # Create data with Inf values
        data_with_inf = np.array([
            [10, 20, 30],
            [20, np.inf, 60],
            [30, 60, 90]
        ])
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="NaN or Inf"):
            normalizer.normalize(data_with_inf)
    
    @patch('pronoms.normalizers.vsn_normalizer.run_r_script')
    def test_normalize_with_run_r_script_error(self, mock_run):
        """Test that normalization handles run_r_script errors."""
        # Mock run_r_script to raise an exception
        mock_run.side_effect = Exception("R script error")
        
        # Create normalizer
        normalizer = VSNNormalizer()
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="VSN normalization failed"):
            normalizer.normalize(self.data)
    
    @patch('pronoms.normalizers.vsn_normalizer.run_r_script')
    def test_normalize_with_missing_result(self, mock_run):
        """Test that normalization handles missing result."""
        # Mock run_r_script to return a result without normalized_data
        mock_run.return_value = {'parameters': self.mock_params}
        
        # Create normalizer
        normalizer = VSNNormalizer()
        
        # Check that normalization raises a ValueError
        with pytest.raises(ValueError, match="failed to return normalized data"):
            normalizer.normalize(self.data)
    
    @patch('matplotlib.pyplot.Figure')
    @patch('pronoms.normalizers.vsn_normalizer.create_comparison_plot')
    def test_plot_comparison(self, mock_create_plot, mock_figure):
        """Test plot_comparison method."""
        # Mock the result of create_comparison_plot
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_create_plot.return_value = (mock_fig, mock_axes)
        
        # Mock plt.subplots
        mock_fig2 = MagicMock()
        mock_axes2 = (MagicMock(), MagicMock())
        
        with patch('matplotlib.pyplot.subplots', return_value=(mock_fig2, mock_axes2)):
            # Create normalizer
            normalizer = VSNNormalizer()
            
            # Call plot_comparison
            result = normalizer.plot_comparison(self.data, self.data)
            
            # Check that create_comparison_plot was called
            mock_create_plot.assert_called_once()
            
            # Check that the result is a tuple of the mocked figures
            assert isinstance(result, tuple)
            assert result[0] == mock_fig
            assert result[1] == mock_fig2
