"""Tests for ``pronoms.utils.validators``.

These primitives are called at the start of every normalizer's ``normalize``,
so their behavior must be exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pronoms.utils.validators import check_nan_inf, validate_input_data

# ---------------------------------------------------------------------------
# validate_input_data
# ---------------------------------------------------------------------------


def test_accepts_valid_2d_array_and_returns_it_unchanged():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = validate_input_data(arr)
    assert out is arr  # validator does not copy


@pytest.mark.parametrize("bad_input", [[1, 2, 3], (1, 2, 3), "abc", 42, None])
def test_rejects_non_ndarray(bad_input):
    with pytest.raises(TypeError, match=r"Input data must be a numpy array"):
        validate_input_data(bad_input)


@pytest.mark.parametrize("shape", [(5,), (2, 2, 2), ()])
def test_rejects_wrong_dimensionality(shape):
    arr = np.zeros(shape)
    with pytest.raises(ValueError, match=r"must be a 2D array"):
        validate_input_data(arr)


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_rejects_empty_array(shape):
    with pytest.raises(ValueError, match=r"Input data cannot be empty"):
        validate_input_data(np.empty(shape))


# ---------------------------------------------------------------------------
# check_nan_inf
# ---------------------------------------------------------------------------


def test_check_nan_inf_clean_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    has_bad, mask = check_nan_inf(arr)
    assert has_bad is False
    assert isinstance(has_bad, bool)
    assert_array_equal(mask, np.zeros_like(arr, dtype=bool))


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_check_nan_inf_flags_single_cell(bad_value):
    arr = np.array([[1.0, 2.0], [3.0, bad_value]])
    has_bad, mask = check_nan_inf(arr)
    assert has_bad is True
    assert isinstance(has_bad, bool)
    expected_mask = np.array([[False, False], [False, True]])
    assert_array_equal(mask, expected_mask)


def test_check_nan_inf_mask_is_correct_for_mixed_input():
    arr = np.array([[1.0, np.nan, 3.0], [np.inf, 5.0, -np.inf]])
    has_bad, mask = check_nan_inf(arr)
    assert has_bad is True
    expected = np.array([[False, True, False], [True, False, True]])
    assert_array_equal(mask, expected)
    assert mask.sum() == 3
