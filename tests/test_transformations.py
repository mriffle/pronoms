"""Tests for ``pronoms.utils.transformations``.

These helpers are exposed in ``pronoms.utils.__all__`` and can be used
standalone by callers, even though no normalizer currently consumes them.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pronoms.utils.transformations import log_transform, scale_data

# ---------------------------------------------------------------------------
# log_transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base", "log_fn"),
    [(2, np.log2), (10, np.log10), ("e", np.log)],
)
def test_log_transform_uses_correct_base(base, log_fn):
    data = np.array([[1.0, 3.0], [7.0, 15.0]])
    out = log_transform(data, base=base, pseudo_count=1.0)
    assert_allclose(out, log_fn(data + 1.0), rtol=1e-12)


def test_log_transform_pseudo_count_protects_against_log_zero():
    """``log(0)`` is -inf; with pseudo_count=1 the output is finite."""
    data = np.array([[0.0, 0.0], [0.0, 0.0]])
    out = log_transform(data, base=2, pseudo_count=1.0)
    assert_allclose(out, 0.0, atol=0)
    assert np.all(np.isfinite(out))


def test_log_transform_custom_pseudo_count():
    data = np.array([[0.0, 1.0]])
    out = log_transform(data, base=2, pseudo_count=3.0)
    assert_allclose(out, np.log2(data + 3.0), rtol=1e-12)


def test_log_transform_invalid_base_raises():
    with pytest.raises(ValueError, match=r"Invalid log base"):
        log_transform(np.array([[1.0]]), base=5)


# ---------------------------------------------------------------------------
# scale_data -- standard
# ---------------------------------------------------------------------------


def test_scale_standard_centers_and_scales_columns():
    data = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = scale_data(data, method="standard", axis=0)
    # Per-column mean ~ 0, std ~ 1
    assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    assert_allclose(out.std(axis=0), 1.0, rtol=1e-12)


def test_scale_standard_with_mean_only():
    data = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
    out = scale_data(data, method="standard", axis=0, with_mean=True, with_std=False)
    assert_allclose(out.mean(axis=0), 0.0, atol=1e-12)
    # Variance unchanged (only centered)
    assert_allclose(out.std(axis=0), data.std(axis=0), rtol=1e-12)


def test_scale_standard_with_std_only():
    data = np.array([[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]])
    out = scale_data(data, method="standard", axis=0, with_mean=False, with_std=True)
    assert_allclose(out.std(axis=0), 1.0, rtol=1e-12)


def test_scale_standard_handles_zero_std_column():
    data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    out = scale_data(data, method="standard", axis=0)
    # Constant column was centered to 0 and not divided by zero -> output 0
    assert_allclose(out[:, 0], 0.0, atol=0)


# ---------------------------------------------------------------------------
# scale_data -- minmax
# ---------------------------------------------------------------------------


def test_scale_minmax_columns_to_unit_range():
    data = np.array([[0.0, 10.0], [5.0, 30.0], [10.0, 50.0]])
    out = scale_data(data, method="minmax", axis=0)
    assert_allclose(out.min(axis=0), 0.0, atol=0)
    assert_allclose(out.max(axis=0), 1.0, atol=0)


def test_scale_minmax_handles_zero_range_column():
    data = np.array([[5.0, 1.0], [5.0, 5.0], [5.0, 9.0]])
    out = scale_data(data, method="minmax", axis=0)
    # Constant column: (x - min) / 1 = 0 everywhere
    assert_allclose(out[:, 0], 0.0, atol=0)


# ---------------------------------------------------------------------------
# scale_data -- robust
# ---------------------------------------------------------------------------


def test_scale_robust_centers_on_median():
    data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0], [4.0, 400.0], [5.0, 500.0]])
    out = scale_data(data, method="robust", axis=0)
    # The median of each column maps to 0 after subtraction
    assert_allclose(np.median(out, axis=0), 0.0, atol=1e-12)


def test_scale_robust_handles_zero_iqr_column():
    data = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0]])
    out = scale_data(data, method="robust", axis=0)
    assert_allclose(out[:, 0], 0.0, atol=0)


# ---------------------------------------------------------------------------
# scale_data -- l2
# ---------------------------------------------------------------------------


def test_scale_l2_each_column_has_unit_norm():
    data = np.array([[3.0, 0.0], [4.0, 1.0], [0.0, 0.0]])
    out = scale_data(data, method="l2", axis=0)
    norms = np.sqrt(np.sum(out**2, axis=0))
    assert_allclose(norms, 1.0, rtol=1e-12)


def test_scale_l2_handles_zero_column():
    data = np.array([[0.0, 1.0], [0.0, 2.0]])
    out = scale_data(data, method="l2", axis=0)
    # Zero column stays zero (no division by zero crash)
    assert_allclose(out[:, 0], 0.0, atol=0)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_scale_invalid_method_raises():
    with pytest.raises(ValueError, match=r"Invalid scaling method"):
        scale_data(np.array([[1.0]]), method="not-a-method")
