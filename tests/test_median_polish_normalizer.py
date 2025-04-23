import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

# Need to update __init__.py before this import works reliably outside testing context
from pronoms.normalizers.median_polish_normalizer import MedianPolishNormalizer

class TestMedianPolishNormalizer:
    """Test suite for MedianPolishNormalizer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple 3x4 matrix for testing
        self.data = np.array([
            [5, 6, 7, 8],   # Sample 1 (Row effect ~6.5)
            [8, 9, 10, 11], # Sample 2 (Row effect ~9.5)
            [2, 3, 4, 5]    # Sample 3 (Row effect ~3.5)
        ], dtype=float)
        # Column effects: ~5, 6, 7, 8
        # Overall median expected around (6.5+9.5+3.5)/3 = 6.5? Or median of medians.

        # Data with known additive structure (log scale)
        # Overall = 5, Row = [0, 2, -1], Col = [0, 1, -1, 0.5]
        # Expected: Residuals = 0
        overall = 5
        row_eff = np.array([0, 2, -1])
        col_eff = np.array([0, 1, -1, 0.5])
        self.additive_data = overall + row_eff[:, np.newaxis] + col_eff[np.newaxis, :]
        # Expected residuals should be close to 0
        # Expected normalized = overall + residuals ~ overall = 5
        self.expected_additive_normalized = np.full_like(self.additive_data, overall)

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        normalizer = MedianPolishNormalizer()
        assert normalizer.max_iterations == 10
        assert normalizer.tolerance == 0.01
        assert normalizer.epsilon == 1e-6
        assert normalizer.log_transform is True
        assert normalizer.row_effects is None
        assert normalizer.col_effects is None
        assert normalizer.overall_median is None
        assert normalizer.residuals is None
        assert normalizer.converged is None
        assert normalizer.iterations_run is None

    def test_init_custom(self):
        """Test initialization with custom parameters."""
        normalizer = MedianPolishNormalizer(max_iterations=5, tolerance=0.1, epsilon=1e-5, log_transform=False)
        assert normalizer.max_iterations == 5
        assert normalizer.tolerance == 0.1
        assert normalizer.epsilon == 1e-5
        assert normalizer.log_transform is False

    def test_init_invalid_params(self):
        """Test initialization raises ValueError for invalid parameters."""
        with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
            MedianPolishNormalizer(max_iterations=0)
        with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
            MedianPolishNormalizer(max_iterations=-5)
        with pytest.raises(ValueError, match="tolerance must be a non-negative number"):
            MedianPolishNormalizer(tolerance=-0.01)
        with pytest.raises(ValueError, match="epsilon must be a non-negative number"):
            MedianPolishNormalizer(epsilon=-1e-6)
        with pytest.raises(ValueError, match="log_transform must be a boolean"):
            MedianPolishNormalizer(log_transform="True")

    def test_normalize_shape_dtype(self):
        """Test output shape and dtype."""
        normalizer = MedianPolishNormalizer()
        normalized_data = normalizer.normalize(self.data)
        assert normalized_data.shape == self.data.shape
        assert normalized_data.dtype == np.float64 # Due to internal log/exp

    def test_normalize_nan_inf(self):
        """Test ValueError for NaN/Inf input."""
        normalizer = MedianPolishNormalizer()
        data_nan = self.data.copy()
        data_nan[0, 0] = np.nan
        # Match the error message raised *after* potential log transform
        with pytest.raises(ValueError, match="Input contains NaN or Inf values."):
            normalizer.normalize(data_nan)

        data_inf = self.data.copy()
        data_inf[1, 1] = np.inf
        # Match the error message raised *after* potential log transform
        with pytest.raises(ValueError, match="Input contains NaN or Inf values."):
            normalizer.normalize(data_inf)

    def test_normalize_non_log(self):
        """Test normalization without log transform on additive data."""
        normalizer = MedianPolishNormalizer(log_transform=False, max_iterations=20, tolerance=1e-9)
        normalized_data = normalizer.normalize(self.additive_data)

        # Should recover the overall median + residuals (which should be near 0)
        assert normalizer.converged is True
        assert normalizer.iterations_run < normalizer.max_iterations
        np.testing.assert_allclose(normalizer.residuals, np.zeros_like(self.additive_data), atol=1e-8)
        # Check normalized data is overall_median + residuals (residuals are ~0 here)
        expected_norm = np.full_like(self.additive_data, normalizer.overall_median)
        np.testing.assert_allclose(normalized_data, expected_norm, atol=1e-8)
        # Check effects (relative to their medians)
        # Calculated effects might differ slightly based on implementation details
        # but residuals and normalized data are key

    def test_normalize_log(self):
        """Test normalization with log transform on exponential data."""
        # Create data that becomes additive after log
        exp_data = np.exp(self.additive_data)
        normalizer = MedianPolishNormalizer(log_transform=True, max_iterations=20, tolerance=1e-9, epsilon=1e-12)
        normalized_data = normalizer.normalize(exp_data)

        # After log, it's the additive case. 
        # Should be overall_median + residuals -> overall_median as residuals ~0
        assert normalizer.converged is True
        assert normalizer.iterations_run < normalizer.max_iterations
        np.testing.assert_allclose(normalizer.residuals, np.zeros_like(self.additive_data), atol=1e-8)
        # Check final normalized data (now returned in log-space)
        # Should be overall_median + residuals -> overall_median as residuals ~0
        expected_normalized_log = np.full_like(self.additive_data, normalizer.overall_median)
        np.testing.assert_allclose(normalized_data, expected_normalized_log, rtol=1e-6, atol=1e-6)

    def test_normalize_convergence(self):
        """Test convergence status and iteration count."""
        # Use data likely to converge quickly
        normalizer = MedianPolishNormalizer(log_transform=False, max_iterations=5, tolerance=1e-9)
        normalizer.normalize(self.additive_data)
        assert normalizer.converged is True
        assert normalizer.iterations_run <= 5

        # Use data and params unlikely to converge
        complex_data = np.random.rand(10, 10) * 100
        normalizer_no_converge = MedianPolishNormalizer(log_transform=False, max_iterations=2, tolerance=1e-15)
        normalizer_no_converge.normalize(complex_data)
        assert normalizer_no_converge.converged is False
        assert normalizer_no_converge.iterations_run == 2

    def test_attribute_availability(self):
        """Test that effect attributes are populated after normalization."""
        normalizer = MedianPolishNormalizer(log_transform=False)
        # Attributes should be None before normalize
        assert normalizer.row_effects is None
        assert normalizer.col_effects is None
        assert normalizer.overall_median is None
        assert normalizer.residuals is None
        assert normalizer.converged is None
        assert normalizer.iterations_run is None

        normalized_data = normalizer.normalize(self.additive_data)

        # Attributes should be populated after normalize
        assert isinstance(normalizer.row_effects, np.ndarray)
        assert normalizer.row_effects.shape == (self.additive_data.shape[0],)
        assert isinstance(normalizer.col_effects, np.ndarray)
        assert normalizer.col_effects.shape == (self.additive_data.shape[1],)
        assert isinstance(normalizer.overall_median, float)
        assert isinstance(normalizer.residuals, np.ndarray)
        assert normalizer.residuals.shape == self.additive_data.shape
        assert isinstance(normalizer.converged, bool)
        assert isinstance(normalizer.iterations_run, int)

    def test_plot_comparison_log_transform(self):
        """Test plot_comparison method runs and returns Figure (log transform case)."""
        normalizer = MedianPolishNormalizer(log_transform=True, epsilon=1e-12)
        # Use exponential data for log transform test
        exp_data = np.exp(self.additive_data)
        normalized_data = normalizer.normalize(exp_data)

        # Prevent plot display during tests
        with patch('matplotlib.pyplot.show'):
            fig = normalizer.plot_comparison(exp_data, normalized_data)

        assert isinstance(fig, plt.Figure)
        # Check labels based on log_transform=True
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "Original Data (Log Scale)"
        assert ax.get_ylabel() == "Normalized Data (Log Scale)"
        assert ax.get_title() == "Median Polish Normalization Comparison"
        plt.close(fig) # Close figure to prevent display/memory leak

    def test_plot_comparison_no_log_transform(self):
        """Test plot_comparison method runs and returns Figure (no log transform case)."""
        normalizer = MedianPolishNormalizer(log_transform=False)
        # Use additive data for no log transform test (ensure positive for log x-axis)
        positive_additive_data = self.additive_data + 10 # Ensure all positive
        normalized_data = normalizer.normalize(positive_additive_data)

        # Prevent plot display during tests
        with patch('matplotlib.pyplot.show'):
            fig = normalizer.plot_comparison(positive_additive_data, normalized_data)

        assert isinstance(fig, plt.Figure)
        # Check labels based on log_transform=False
        ax = fig.get_axes()[0]
        assert ax.get_xlabel() == "Original Data (Log Scale)"
        assert ax.get_ylabel() == "Normalized Data"
        assert ax.get_title() == "Median Polish Normalization Comparison"
        plt.close(fig) # Close figure to prevent display/memory leak

    # Test case for when input data has no positive values for log scale
    def test_plot_comparison_no_positive_data(self):
        """Test plot_comparison handles case with no positive data for log x-axis."""
        normalizer = MedianPolishNormalizer(log_transform=False)
        non_positive_data = np.array([[-1, 0], [-2, -3]])
        # Normalization might fail here depending on checks, but we focus on plot
        # Let's assume normalize produced *some* output for plotting
        dummy_normalized = np.array([[0, 0], [0, 0]])

        with patch('matplotlib.pyplot.show'):
             fig = normalizer.plot_comparison(non_positive_data, dummy_normalized)

        assert isinstance(fig, plt.Figure)
        ax = fig.get_axes()[0]
        # Check that the text message is present
        assert len(ax.texts) > 0
        assert ax.texts[0].get_text() == "No positive data to plot on log scale"
        plt.close(fig)
