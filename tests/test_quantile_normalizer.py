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


def test_tied_inputs_within_a_row_map_to_equal_outputs():
    """Bolstad et al.: two equal input values in the same row must receive the
    same normalized value, computed as the average of the reference values at
    their tied positions."""
    data = np.array(
        [
            [5.0, 5.0, 10.0],
            [1.0, 2.0, 3.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=float,
    )
    normalizer = QuantileNormalizer()
    normalized = normalizer.normalize(data)

    # Reference is the column-mean of row-sorted data.
    expected_reference = np.array([5 + 1 + 10, 5 + 2 + 20, 10 + 3 + 30], dtype=float) / 3.0
    assert_allclose(normalizer.reference_distribution, expected_reference, rtol=1e-12)

    # Row 0: ranks via average-tie are [1.5, 1.5, 3]; outputs are
    # mean(ref[0], ref[1]) for the ties and ref[2] for the largest value.
    expected_row0 = np.array(
        [
            (expected_reference[0] + expected_reference[1]) / 2,
            (expected_reference[0] + expected_reference[1]) / 2,
            expected_reference[2],
        ]
    )
    assert_allclose(normalized[0], expected_row0, rtol=1e-12)
    assert normalized[0, 0] == normalized[0, 1]


def test_all_identical_row_collapses_to_mean_of_reference():
    """A row that is entirely tied must be mapped to a single value: the mean
    of the reference distribution (the average of all reference quantiles)."""
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

    expected = np.full(data.shape[1], normalizer.reference_distribution.mean())
    assert_allclose(normalized[0], expected, rtol=1e-12)
    # The non-tied rows (1 and 2) still satisfy the standard property: their
    # sorted output equals the reference distribution.
    for row in normalized[1:]:
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
