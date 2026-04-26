"""Tests for ``pronoms.utils.plotting.create_hexbin_comparison``.

This helper backs every normalizer's ``plot_comparison``; its branches
(log/raw axes, ``transform_original='log2'``, shape mismatch, custom limits,
identity vs y=0 reference line) need direct coverage rather than being
exercised only through the normalizer wrappers.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pronoms.utils.plotting import create_hexbin_comparison, plot_comparison_hexbin

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_pair():
    rng = np.random.default_rng(0)
    before = rng.random((20, 5)) * 100 + 1.0
    after = before * 0.5 + 1.0
    return before, after


# ---------------------------------------------------------------------------
# Smoke / shape contract
# ---------------------------------------------------------------------------


def test_returns_figure_with_axes(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(before, after)
    assert isinstance(fig, plt.Figure)
    # one axes for the hexbin + one for the colorbar
    assert len(fig.axes) >= 1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match=r"Data shapes must match"):
        create_hexbin_comparison(np.zeros((3, 4)), np.zeros((3, 5)))


def test_alias_plot_comparison_hexbin_is_same_callable():
    assert plot_comparison_hexbin is create_hexbin_comparison


# ---------------------------------------------------------------------------
# log_axes branches
# ---------------------------------------------------------------------------


def test_log_axes_true_labels(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(
        before,
        after,
        log_axes=True,
        xlabel="Before",
        ylabel="After",
    )
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Log10(Before + 1)"
    assert ax.get_ylabel() == "Log10(After + 1)"


def test_log_axes_false_keeps_raw_labels(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(
        before,
        after,
        log_axes=False,
        xlabel="Before",
        ylabel="After",
    )
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Before"
    assert ax.get_ylabel() == "After"


# ---------------------------------------------------------------------------
# transform_original='log2' branch (used by VSN)
# ---------------------------------------------------------------------------


def test_transform_original_log2_relabels_x_axis(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(
        before,
        after,
        log_axes=False,
        transform_original="log2",
    )
    assert fig.axes[0].get_xlabel() == "Log2(Original Value + 1)"


# ---------------------------------------------------------------------------
# Reference-line branches
# ---------------------------------------------------------------------------


def test_add_center_line_y0_draws_zero_line(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(
        before,
        after,
        log_axes=False,
        add_center_line_y0=True,
    )
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "y = 0" in labels


def test_identity_line_drawn_when_aspect_equal_and_no_center_line(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(
        before,
        after,
        log_axes=False,
        add_identity_line=True,
        autoscale_y=False,
    )
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "y = x" in labels


def test_no_identity_line_when_autoscale_y(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(
        before,
        after,
        log_axes=False,
        add_identity_line=True,
        autoscale_y=True,
    )
    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "y = x" not in labels


# ---------------------------------------------------------------------------
# Custom axis limits
# ---------------------------------------------------------------------------


def test_custom_xlim_and_ylim_are_respected(small_pair):
    before, after = small_pair
    fig = create_hexbin_comparison(
        before,
        after,
        log_axes=False,
        autoscale_y=True,
        xlim=(-1.0, 5.0),
        ylim=(-2.0, 7.0),
    )
    ax = fig.axes[0]
    assert ax.get_xlim() == (-1.0, 5.0)
    assert ax.get_ylim() == (-2.0, 7.0)
