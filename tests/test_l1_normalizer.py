"""Tests for ``L1Normalizer``.

Each row is divided by its L1 norm and rescaled by the mean of those norms,
so the per-row L1 norm of the *output* equals the mean of the original norms.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import L1Normalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_data() -> np.ndarray:
    """3×5 matrix with row L1 norms [150, 300, 450] (mean = 300)."""
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
    ("data", "expected_norms"),
    [
        (
            np.array(
                [
                    [10, 20, 30, 40, 50],
                    [20, 40, 60, 80, 100],
                    [30, 60, 90, 120, 150],
                ],
                dtype=float,
            ),
            np.array([150.0, 300.0, 450.0]),
        ),
        # zeros are fine; |0| contributes nothing
        (
            np.array(
                [
                    [0, 10, 20],
                    [0, 20, 40],
                    [0, 30, 60],
                ],
                dtype=float,
            ),
            np.array([30.0, 60.0, 90.0]),
        ),
        # negatives: L1 norm uses absolute values
        (
            np.array(
                [
                    [-10, 20, -30],
                    [20, -40, 60],
                    [30, 60, -90],
                ],
                dtype=float,
            ),
            np.array([60.0, 120.0, 180.0]),
        ),
    ],
)
def test_normalize_matches_closed_form(data, expected_norms):
    normalizer = L1Normalizer()
    normalized = normalizer.normalize(data)

    mean_norm = float(expected_norms.mean())
    expected = data / expected_norms[:, None] * mean_norm

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == data.shape
    assert_allclose(normalized, expected, rtol=1e-12, atol=0)
    assert_allclose(normalizer.scaling_factors, expected_norms, rtol=1e-12)
    assert normalizer.mean_of_scaling_factors == pytest.approx(mean_norm, rel=1e-12)


def test_normalized_rows_share_a_common_l1_norm(simple_data):
    """Post-normalization L1 norms equal the *mean* of the original L1 norms."""
    normalizer = L1Normalizer()
    normalized = normalizer.normalize(simple_data)

    output_norms = np.sum(np.abs(normalized), axis=1)
    assert_allclose(output_norms, normalizer.mean_of_scaling_factors, rtol=1e-12)


def test_all_zero_row_is_preserved_as_zero():
    """An all-zero row has L1 norm 0; the implementation guards against
    division by zero and the row stays all zeros."""
    data = np.array(
        [
            [0, 0, 0, 0, 0],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150],
        ],
        dtype=float,
    )
    normalizer = L1Normalizer()
    normalized = normalizer.normalize(data)

    # Stored scaling factor for the zero row is the safe-replacement value 1.0
    assert_allclose(normalizer.scaling_factors, [1.0, 300.0, 450.0])
    # The zero row remains exactly zero
    assert_allclose(normalized[0], 0.0, atol=0)


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
        L1Normalizer().normalize(data)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_comparison_returns_figure(simple_data):
    normalizer = L1Normalizer()
    normalized = normalizer.normalize(simple_data)

    fig = normalizer.plot_comparison(simple_data, normalized)

    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
