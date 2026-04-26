"""Tests for ``MedianPolishNormalizer``.

Median polish decomposes a 2-way table into ``overall + row + col + residual``
components by iteratively sweeping out row and column medians. The normalized
output is ``overall + residuals``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import MedianPolishNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def additive_data() -> np.ndarray:
    """Matrix with known additive structure -> residuals must be ~0."""
    overall = 5.0
    row_eff = np.array([0.0, 2.0, -1.0])
    col_eff = np.array([0.0, 1.0, -1.0, 0.5])
    return overall + row_eff[:, None] + col_eff[None, :]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_defaults():
    normalizer = MedianPolishNormalizer()
    assert normalizer.max_iterations == 10
    assert normalizer.tolerance == 0.01
    assert normalizer.epsilon == 1e-6
    assert normalizer.log_transform is True
    for attr in ("row_effects", "col_effects", "overall_median", "residuals", "converged", "iterations_run"):
        assert getattr(normalizer, attr) is None


def test_init_custom():
    normalizer = MedianPolishNormalizer(max_iterations=5, tolerance=0.1, epsilon=1e-5, log_transform=False)
    assert normalizer.max_iterations == 5
    assert normalizer.tolerance == 0.1
    assert normalizer.epsilon == 1e-5
    assert normalizer.log_transform is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 0}, r"max_iterations must be a positive integer"),
        ({"max_iterations": -5}, r"max_iterations must be a positive integer"),
        ({"tolerance": -0.01}, r"tolerance must be a non-negative number"),
        ({"epsilon": -1e-6}, r"epsilon must be a non-negative number"),
        ({"log_transform": "True"}, r"log_transform must be a boolean"),
    ],
)
def test_init_rejects_invalid_params(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MedianPolishNormalizer(**kwargs)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_additive_data_resolves_to_zero_residuals(additive_data):
    """On purely additive data, residuals must be ~0 and the output ~overall."""
    normalizer = MedianPolishNormalizer(log_transform=False, max_iterations=20, tolerance=1e-9)
    normalized = normalizer.normalize(additive_data)

    assert normalizer.converged is True
    assert normalizer.iterations_run < normalizer.max_iterations
    assert_allclose(normalizer.residuals, 0.0, atol=1e-8)
    expected = np.full_like(additive_data, normalizer.overall_median)
    assert_allclose(normalized, expected, atol=1e-8)


def test_log_mode_handles_multiplicative_structure(additive_data):
    """Exponentiating the additive matrix gives multiplicative structure;
    median polish in log space must recover near-zero residuals."""
    exp_data = np.exp(additive_data)
    normalizer = MedianPolishNormalizer(log_transform=True, max_iterations=20, tolerance=1e-9, epsilon=1e-12)
    normalized = normalizer.normalize(exp_data)

    assert normalizer.converged is True
    assert_allclose(normalizer.residuals, 0.0, atol=1e-8)
    expected_log = np.full_like(additive_data, normalizer.overall_median)
    assert_allclose(normalized, expected_log, rtol=1e-6, atol=1e-6)


def test_decomposition_is_consistent(additive_data):
    """``overall + row + col + residuals`` must reconstruct the (log-)input."""
    normalizer = MedianPolishNormalizer(log_transform=False, max_iterations=50, tolerance=1e-12)
    normalizer.normalize(additive_data)

    reconstructed = (
        normalizer.overall_median
        + normalizer.row_effects[:, None]
        + normalizer.col_effects[None, :]
        + normalizer.residuals
    )
    assert_allclose(reconstructed, additive_data, atol=1e-8)


def test_no_convergence_within_iterations():
    """Tight tolerance + tiny iter cap on noisy data must not converge."""
    rng = np.random.default_rng(42)
    noisy = rng.random((10, 10)) * 100
    normalizer = MedianPolishNormalizer(log_transform=False, max_iterations=2, tolerance=1e-15)
    normalizer.normalize(noisy)
    assert normalizer.converged is False
    assert normalizer.iterations_run == 2


# ---------------------------------------------------------------------------
# Output type / state
# ---------------------------------------------------------------------------


def test_output_shape_and_attributes_populated(additive_data):
    normalizer = MedianPolishNormalizer(log_transform=False)
    normalized = normalizer.normalize(additive_data)

    assert normalized.shape == additive_data.shape
    assert normalized.dtype == np.float64
    assert isinstance(normalizer.row_effects, np.ndarray)
    assert normalizer.row_effects.shape == (additive_data.shape[0],)
    assert isinstance(normalizer.col_effects, np.ndarray)
    assert normalizer.col_effects.shape == (additive_data.shape[1],)
    assert isinstance(normalizer.overall_median, float)
    assert isinstance(normalizer.residuals, np.ndarray)
    assert normalizer.residuals.shape == additive_data.shape
    assert isinstance(normalizer.converged, bool)
    assert isinstance(normalizer.iterations_run, int)


# ---------------------------------------------------------------------------
# Validation / error behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_input_raises(additive_data, bad_value):
    bad = additive_data.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError, match=r"NaN or Inf"):
        MedianPolishNormalizer(log_transform=False).normalize(bad)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_comparison_log_mode_labels(additive_data):
    exp_data = np.exp(additive_data)
    normalizer = MedianPolishNormalizer(log_transform=True, epsilon=1e-12)
    normalized = normalizer.normalize(exp_data)

    fig = normalizer.plot_comparison(exp_data, normalized)
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)
    assert ax.get_xlabel() == "Original Data (Log Scale)"
    assert ax.get_ylabel() == "Normalized Data (Log Scale)"
    assert ax.get_title() == "Median Polish Normalization Comparison"


def test_plot_comparison_log_mode_diagonal_endpoints(additive_data):
    """The 'y = log(x)' diagonal must be drawn against the actual coordinate
    system: x-coords are raw values (axis is ``xscale='log'``), and y-coords
    are ``log(x)`` to match the log-space y data."""
    exp_data = np.exp(additive_data)
    normalizer = MedianPolishNormalizer(log_transform=True, epsilon=1e-12)
    normalized = normalizer.normalize(exp_data)

    fig = normalizer.plot_comparison(exp_data, normalized)
    ax = fig.axes[0]

    line = next(line for line in ax.get_lines() if line.get_label() == "y = log(x)")
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    x_positive = exp_data.flatten()[exp_data.flatten() > 0]
    x_min = float(x_positive.min())
    x_max = float(x_positive.max())

    assert_allclose(xdata, [x_min, x_max], rtol=1e-12)
    assert_allclose(ydata, [np.log(x_min), np.log(x_max)], rtol=1e-12)


def test_plot_comparison_no_log_mode_labels(additive_data):
    positive = additive_data + 10.0
    normalizer = MedianPolishNormalizer(log_transform=False)
    normalized = normalizer.normalize(positive)

    fig = normalizer.plot_comparison(positive, normalized)
    ax = fig.axes[0]

    assert ax.get_xlabel() == "Original Data (Log Scale)"
    assert ax.get_ylabel() == "Normalized Data"
    assert ax.get_title() == "Median Polish Normalization Comparison"


def test_plot_comparison_no_positive_values_for_log_axis():
    """The log-scale x-axis can't render <=0 values; the helper should
    display an explanatory message instead of crashing."""
    normalizer = MedianPolishNormalizer(log_transform=False)
    non_positive = np.array([[-1.0, 0.0], [-2.0, -3.0]])
    dummy = np.zeros_like(non_positive)

    fig = normalizer.plot_comparison(non_positive, dummy)
    ax = fig.axes[0]

    assert ax.texts
    assert ax.texts[0].get_text() == "No positive data to plot on log scale"
