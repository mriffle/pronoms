"""Tests for ``MedianNormalizer``.

These tests exercise the documented contract: each row is divided by its own
median and then rescaled by the mean-of-medians so the global scale is
preserved.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import MedianNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data() -> np.ndarray:
    """3×5 matrix with row medians [30, 60, 90] (mean of medians = 60)."""
    return np.array(
        [
            [10, 20, 30, 40, 50],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected_medians"),
    [
        # odd feature count
        (
            np.array(
                [
                    [10, 20, 30, 40, 50],
                    [20, 40, 60, 80, 100],
                    [30, 60, 90, 120, 150],
                ],
                dtype=float,
            ),
            np.array([30.0, 60.0, 90.0]),
        ),
        # even feature count -> median is mean of two middle values
        (
            np.array(
                [
                    [10, 20, 30, 40],
                    [20, 40, 60, 80],
                    [30, 60, 90, 120],
                ],
                dtype=float,
            ),
            np.array([25.0, 50.0, 75.0]),
        ),
        # zeros allowed as long as the row median is positive
        (
            np.array(
                [
                    [0, 10, 20],
                    [0, 20, 40],
                    [0, 30, 60],
                ],
                dtype=float,
            ),
            np.array([10.0, 20.0, 30.0]),
        ),
    ],
)
def test_normalize_matches_closed_form(data, expected_medians):
    """Output must equal ``X / median * mean(medians)`` exactly."""
    normalizer = MedianNormalizer()
    normalized = normalizer.normalize(data)

    mean_of_medians = float(expected_medians.mean())
    expected = data / expected_medians[:, None] * mean_of_medians

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == data.shape
    assert_allclose(normalized, expected, rtol=1e-12, atol=0)
    assert_allclose(normalizer.scaling_factors, expected_medians, rtol=1e-12)
    assert normalizer.mean_of_medians == pytest.approx(mean_of_medians, rel=1e-12)


def test_scale_preservation_invariant(simple_data):
    """Mean-of-medians of the *normalized* data equals that of the input.

    This is the key property the docstring promises ("preserves overall scale").
    """
    normalizer = MedianNormalizer()
    normalized = normalizer.normalize(simple_data)

    original_mean_of_medians = float(np.median(simple_data, axis=1).mean())
    normalized_mean_of_medians = float(np.median(normalized, axis=1).mean())

    assert normalized_mean_of_medians == pytest.approx(original_mean_of_medians, rel=1e-12)


def test_normalized_rows_share_a_common_median(simple_data):
    """After scaling, every row's median equals the mean-of-medians."""
    normalizer = MedianNormalizer()
    normalized = normalizer.normalize(simple_data)

    row_medians_after = np.median(normalized, axis=1)
    assert_allclose(row_medians_after, normalizer.mean_of_medians, rtol=1e-12)


# ---------------------------------------------------------------------------
# Validation / error behaviour
# ---------------------------------------------------------------------------


def test_zero_median_row_raises():
    """A row of all zeros has median 0, which is forbidden."""
    data = np.array(
        [
            [0, 0, 0, 0, 0],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150],
        ],
        dtype=float,
    )
    with pytest.raises(ValueError, match=r"All sample medians must be > 0"):
        MedianNormalizer().normalize(data)


def test_negative_median_row_raises():
    """Negative-dominant rows (median <= 0) are rejected."""
    data = np.array(
        [
            [-10, -5, -1, 0, 1],  # median = -1
            [10, 20, 30, 40, 50],
        ],
        dtype=float,
    )
    with pytest.raises(ValueError, match=r"All sample medians must be > 0"):
        MedianNormalizer().normalize(data)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(bad_value):
    data = np.array(
        [
            [10, 20, 30],
            [20, bad_value, 60],
            [30, 60, 90],
        ],
        dtype=float,
    )
    with pytest.raises(ValueError, match=r"NaN or Inf"):
        MedianNormalizer().normalize(data)


def test_non_2d_input_raises():
    with pytest.raises(ValueError, match=r"X must be a 2D array"):
        MedianNormalizer().normalize(np.array([1.0, 2.0, 3.0]))


def test_zero_feature_input_raises():
    with pytest.raises(ValueError, match=r"X must be a 2D array"):
        MedianNormalizer().normalize(np.empty((3, 0)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_comparison_returns_figure_with_one_axis(simple_data):
    normalizer = MedianNormalizer()
    normalized = normalizer.normalize(simple_data)

    fig = normalizer.plot_comparison(simple_data, normalized)

    assert isinstance(fig, plt.Figure)
    # one axes for the hexbin + one for the colorbar
    assert len(fig.axes) >= 1
