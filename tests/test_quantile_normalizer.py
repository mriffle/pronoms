"""Tests for ``QuantileNormalizer``.

The reference distribution is the per-column mean of the row-sorted matrix.
After normalization every row, when sorted, must equal that reference.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import QuantileNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data() -> np.ndarray:
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


def test_reference_distribution_is_column_mean_of_sorted_rows(simple_data):
    normalizer = QuantileNormalizer()
    normalizer.normalize(simple_data)

    expected_reference = np.mean(np.sort(simple_data, axis=1), axis=0)
    assert_allclose(normalizer.reference_distribution, expected_reference, rtol=1e-12)


def test_normalized_rows_match_reference_distribution_when_sorted(simple_data):
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(simple_data)

    for row in normalized:
        assert_allclose(np.sort(row), normalizer.reference_distribution, rtol=1e-12)


def test_within_row_ordering_is_preserved(simple_data):
    """Quantile normalization must not permute features within a row."""
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(simple_data)

    for original_row, normalized_row in zip(simple_data, normalized):
        assert_allclose(np.argsort(np.argsort(original_row)), np.argsort(np.argsort(normalized_row)))


def test_two_sample_known_values():
    """Worked-example check from the docstring.

    For samples [1, 3, 2] and [10, 30, 20]:
      sorted rows: [1, 2, 3] and [10, 20, 30]
      reference (column mean): [5.5, 11.0, 16.5]
      output: [5.5, 16.5, 11.0] and [5.5, 16.5, 11.0]
    """
    data = np.array([[1, 3, 2], [10, 30, 20]], dtype=float)
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(data)

    expected_reference = np.array([5.5, 11.0, 16.5])
    expected = np.array([[5.5, 16.5, 11.0], [5.5, 16.5, 11.0]])

    assert_allclose(normalizer.reference_distribution, expected_reference, rtol=1e-12)
    assert_allclose(normalized, expected, rtol=1e-12)


def test_all_identical_row_still_yields_reference_distribution_when_sorted():
    """When a row is all-identical, the sorted output is still the reference
    distribution. (Note: the current rank-based mapping does *not* assign the
    same reference value to tied inputs -- callers needing tie-aware mapping
    should pre-rank the data themselves.)"""
    data = np.array(
        [
            [10, 10, 10],
            [20, 40, 60],
            [30, 60, 90],
        ],
        dtype=float,
    )
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(data)

    for row in normalized:
        assert_allclose(np.sort(row), normalizer.reference_distribution, rtol=1e-12)


def test_negative_values_are_handled(simple_data):
    """Quantile normalization is rank-based and handles negatives."""
    data = np.array(
        [
            [-10, 20, 30],
            [20, -40, 60],
            [-30, 60, -90],
        ],
        dtype=float,
    )
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(data)

    # ordering preserved per row
    for original_row, normalized_row in zip(data, normalized):
        assert_allclose(np.argsort(np.argsort(original_row)), np.argsort(np.argsort(normalized_row)))
    # reference still well-defined and produced
    assert normalizer.reference_distribution.shape == (data.shape[1],)


# ---------------------------------------------------------------------------
# Validation / error behaviour
# ---------------------------------------------------------------------------


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
        QuantileNormalizer().normalize(data)


# ---------------------------------------------------------------------------
# Different shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5, 10), (3, 20), (2, 2), (10, 1)])
def test_shape_preserved(shape):
    rng = np.random.default_rng(0)
    data = rng.random(shape)
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(data)
    assert normalized.shape == shape
    assert normalizer.reference_distribution.shape == (shape[1],)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_comparison_returns_single_figure(simple_data):
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(simple_data)

    fig = normalizer.plot_comparison(simple_data, normalized)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
