"""Tests for ``MADNormalizer``.

Each row is centered by its (log) median and scaled by its (log) MAD. The
default operates on ``log2(X + 1)``; ``log_transform=False`` operates on the
raw values.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.normalizers import MADNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data() -> np.ndarray:
    """4×5 matrix where row 3 has zero MAD on both raw and log2 scales."""
    return np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],  # median=3, MAD=1
            [6.0, 7.0, 8.0, 9.0, 10.0],  # median=8, MAD=1
            [1.0, 2.0, 3.0, 4.0, 100.0],  # outlier-tolerant
            [5.0, 5.0, 5.0, 5.0, 5.0],  # MAD=0 on every scale -> error
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log_transform", [True, False])
def test_normalize_matches_closed_form(data, log_transform):
    normalizer = MADNormalizer(log_transform=log_transform)
    subset = data[:3, :]  # exclude the zero-MAD row

    Y = np.log2(subset + 1) if log_transform else subset
    medians = np.median(Y, axis=1, keepdims=True)
    mads = np.median(np.abs(Y - medians), axis=1, keepdims=True)
    expected = (Y - medians) / mads

    normalized = normalizer.normalize(subset)

    assert normalized.shape == subset.shape
    assert_allclose(normalized, expected, rtol=1e-12, atol=0)
    assert_allclose(normalizer.row_medians, medians.flatten(), rtol=1e-12)
    assert_allclose(normalizer.row_mads, mads.flatten(), rtol=1e-12)
    assert normalizer.log_transform is log_transform


@pytest.mark.parametrize("log_transform", [True, False])
def test_per_row_median_is_zero_after_normalization(data, log_transform):
    """Docstring contract: each row has median 0 post-normalization."""
    subset = data[:3, :]
    normalized = MADNormalizer(log_transform=log_transform).normalize(subset)
    row_medians_after = np.median(normalized, axis=1)
    assert_allclose(row_medians_after, 0.0, atol=1e-12)


def test_known_values_log_false():
    """Worked example with negatives (only valid for ``log_transform=False``).

    Sample [1, 2, 3]: median=2, MAD=median(|x-2|)=median([1,0,1])=1 -> [-1, 0, 1]
    Sample [-1, -2, -3]: median=-2, MAD=1 -> [1, 0, -1]
    """
    data = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    normalizer = MADNormalizer(log_transform=False)
    normalized = normalizer.normalize(data)
    expected = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]])
    assert_allclose(normalized, expected, rtol=1e-12)
    assert_allclose(normalizer.row_medians, [2.0, -2.0], rtol=1e-12)
    assert_allclose(normalizer.row_mads, [1.0, 1.0], rtol=1e-12)


# ---------------------------------------------------------------------------
# Validation / error behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("log_transform", "scale_label"),
    [(True, r"log2\(X\+1\)"), (False, "original")],
)
def test_zero_mad_row_raises(data, log_transform, scale_label):
    """Identical-value rows have MAD=0 on both scales."""
    expected_msg = (
        rf"Cannot normalize: MAD of {scale_label} data is zero "
        r"for sample\(s\) at index/indices: \[3\]"
    )
    with pytest.raises(ValueError, match=expected_msg):
        MADNormalizer(log_transform=log_transform).normalize(data)


@pytest.mark.parametrize("log_transform", [True, False])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(data, log_transform, bad_value):
    bad = data.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError, match=r"NaN or Inf"):
        MADNormalizer(log_transform=log_transform).normalize(bad)


@pytest.mark.parametrize("log_transform", [True, False])
def test_invalid_dimensions(log_transform):
    normalizer = MADNormalizer(log_transform=log_transform)
    with pytest.raises(ValueError, match=r"must be a 2D array"):
        normalizer.normalize(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match=r"Input data cannot be empty"):
        normalizer.normalize(np.empty((2, 0)))


def test_negative_values_with_log_true_raise():
    data = np.array([[1.0, 2.0], [-1.0, 4.0]])
    with pytest.raises(ValueError, match=r"negative values"):
        MADNormalizer(log_transform=True).normalize(data)


# ---------------------------------------------------------------------------
# Plotting (real call -- no mocking, but uses the Agg backend from conftest)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("log_transform", [True, False])
def test_plot_comparison_labels(data, log_transform):
    subset = data[:3, :]
    normalizer = MADNormalizer(log_transform=log_transform)
    normalized = normalizer.normalize(subset)

    fig = normalizer.plot_comparison(subset, normalized)
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "MAD Normalization Comparison"
    expected_ylabel_kind = "Standardized Log2 Scale" if log_transform else "Standardized Original Scale"
    assert expected_ylabel_kind in ax.get_ylabel()
    # Centered output: a horizontal y=0 reference line should be present
    assert any(line.get_label() == "y = 0" for line in ax.get_lines())
