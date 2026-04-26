"""Tests for ``RankNormalizer``.

Each row's values are replaced with their ranks (1..N), with ``method='average'``
for ties. Optionally the ranks can be divided by N for cross-dataset
comparability.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pronoms.normalizers import RankNormalizer

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_default():
    normalizer = RankNormalizer()
    assert normalizer.normalize_by_n is False
    assert normalizer.ranks is None


def test_init_with_normalize_by_n_true():
    normalizer = RankNormalizer(normalize_by_n=True)
    assert normalizer.normalize_by_n is True
    assert normalizer.ranks is None


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # Plain ranks, no ties
        (np.array([[3.0, 1.0, 2.0]]), np.array([[3.0, 1.0, 2.0]])),
        # Ties at ranks 2,3 -> average rank 2.5
        (np.array([[1.0, 2.0, 2.0, 3.0]]), np.array([[1.0, 2.5, 2.5, 4.0]])),
        # Multiple rows with mixed ties
        (np.array([[3.0, 1.0, 2.0], [5.0, 5.0, 1.0]]), np.array([[3.0, 1.0, 2.0], [2.5, 2.5, 1.0]])),
        # 5 tied values in the middle: average rank = (2+3+4+5+6)/5 = 4
        (np.array([[1, 2, 2, 2, 2, 2, 3]], dtype=float), np.array([[1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 7.0]])),
        # All identical -> average of 1..N
        (np.array([[5.0, 5.0, 5.0, 5.0]]), np.array([[2.5, 2.5, 2.5, 2.5]])),
    ],
)
def test_ranks_match_expected(data, expected):
    normalizer = RankNormalizer()
    result = normalizer.normalize(data)
    assert_array_equal(result, expected)
    assert_array_equal(normalizer.ranks, expected)


@pytest.mark.parametrize(
    ("data", "n", "expected"),
    [
        (np.array([[3.0, 1.0, 2.0]]), 3, np.array([[1.0, 1.0 / 3, 2.0 / 3]])),
        (np.array([[1.0, 2.0, 2.0, 3.0]]), 4, np.array([[0.25, 0.625, 0.625, 1.0]])),
    ],
)
def test_normalize_by_n(data, n, expected):
    normalizer = RankNormalizer(normalize_by_n=True)
    result = normalizer.normalize(data)
    assert_allclose(result, expected, rtol=1e-12)
    assert data.shape[1] == n  # sanity check on the parametrization


def test_rank_range_property():
    """For random data with no ties, every row contains ranks 1..N exactly once."""
    rng = np.random.default_rng(0)
    data = rng.random((4, 7))
    result = RankNormalizer().normalize(data)
    for row in result:
        assert_array_equal(np.sort(row), np.arange(1, 8, dtype=float))


def test_rank_range_property_normalize_by_n():
    rng = np.random.default_rng(0)
    data = rng.random((4, 7))
    result = RankNormalizer(normalize_by_n=True).normalize(data)
    for row in result:
        assert row.min() >= 1.0 / 7
        assert row.max() <= 1.0
        assert_allclose(np.sort(row), np.arange(1, 8) / 7.0, rtol=1e-12)


def test_shape_is_preserved():
    rng = np.random.default_rng(0)
    data = rng.random((5, 10))
    assert RankNormalizer().normalize(data).shape == data.shape


# ---------------------------------------------------------------------------
# Validation / error behaviour
# ---------------------------------------------------------------------------


def test_1d_input_raises():
    with pytest.raises(ValueError, match=r"X must be a 2D array"):
        RankNormalizer().normalize(np.array([1.0, 2.0, 3.0]))


def test_zero_features_raises():
    with pytest.raises(ValueError, match=r"X must be a 2D array"):
        RankNormalizer().normalize(np.empty((2, 0)))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(bad_value):
    with pytest.raises(ValueError, match=r"NaN or Inf"):
        RankNormalizer().normalize(np.array([[1.0, bad_value, 3.0]]))


# ---------------------------------------------------------------------------
# Plotting (real call)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log_axes", [True, False])
def test_plot_comparison_axis_labels(log_axes):
    rng = np.random.default_rng(0)
    data = rng.random((3, 5)) * 100 + 1.0
    normalizer = RankNormalizer()
    normalized = normalizer.normalize(data)

    fig = normalizer.plot_comparison(data, normalized, log_axes=log_axes)
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Rank Normalization Comparison"
    expected_xlabel = "Log10(Original Value + 1)" if log_axes else "Original Value"
    assert ax.get_xlabel() == expected_xlabel
    assert ax.get_ylabel() == f"Assigned Rank (1 to {data.shape[1]})"
    # y-axis is always the rank scale, padded by 1
    assert ax.get_ylim() == (0, data.shape[1] + 1)
