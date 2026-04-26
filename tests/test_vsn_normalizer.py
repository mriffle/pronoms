"""Tests for ``VSNNormalizer``.

VSN itself is implemented in R, so the bulk of the tests use mocks for the R
boundary. The R-script builder (``_create_vsn_script``) and the
``reference_sample`` branch are tested directly without R. A live integration
test is included but skipped automatically if R or the ``vsn`` Bioconductor
package is unavailable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pronoms.normalizers import VSNNormalizer
from pronoms.utils.r_interface import RInterfaceError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _r_vsn_available() -> bool:
    """Return True only if R, rpy2, and the ``vsn`` Bioconductor pkg load."""
    try:
        from pronoms.utils.r_interface import setup_r_environment

        setup_r_environment(["vsn"])
        return True
    except Exception:
        return False


_HAS_R_VSN = _r_vsn_available()


@pytest.fixture
def small_data() -> np.ndarray:
    return np.array(
        [
            [10, 20, 30, 40, 50],
            [20, 40, 60, 80, 100],
            [30, 60, 90, 120, 150],
        ],
        dtype=float,
    )


@pytest.fixture
def mock_params() -> dict:
    return {
        "coefficients": [1.0, 0.5, 0.2],
        "stdev": 0.1,
        "reference": 1,
        "h_parameters": [0.01, 0.001],
    }


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("calib", ["affine", "none", "shift", "maximum"])
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_init_stores_parameters(_mock_setup, calib):
    normalizer = VSNNormalizer(calib=calib, reference_sample=1, lts_quantile=0.5)
    assert normalizer.calib == calib
    assert normalizer.reference_sample == 1
    assert normalizer.lts_quantile == 0.5
    assert normalizer.vsn_params is None


@pytest.mark.parametrize("bad", [-0.1, 1.1])
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_init_rejects_invalid_lts_quantile(_mock_setup, bad):
    with pytest.raises(ValueError, match=r"lts_quantile must be between 0 and 1"):
        VSNNormalizer(lts_quantile=bad)


@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_init_calls_setup_r_environment_for_vsn(mock_setup):
    VSNNormalizer()
    mock_setup.assert_called_once_with(["vsn"])


@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_init_swallows_setup_failure_with_warning(mock_setup, capsys):
    """Missing R / vsn must not crash construction; only the eventual call to
    ``normalize`` should raise. This keeps ``import pronoms`` cheap on systems
    without R."""
    mock_setup.side_effect = RInterfaceError("R package not found")
    VSNNormalizer()
    captured = capsys.readouterr()
    assert "R package not found" in captured.out


# ---------------------------------------------------------------------------
# normalize() with mocked R boundary
# ---------------------------------------------------------------------------


@patch("pronoms.normalizers.vsn_normalizer.run_r_script")
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_normalize_returns_transposed_r_output(
    _mock_setup,
    mock_run,
    small_data,
    mock_params,
):
    """run_r_script returns proteins×samples; the wrapper transposes back."""
    r_output = small_data.T + 1.0  # arbitrary shape-correct payload (5×3)
    mock_run.return_value = {"normalized_data": r_output, "parameters": mock_params}

    normalizer = VSNNormalizer()
    out = normalizer.normalize(
        small_data,
        protein_ids=[f"P{i}" for i in range(small_data.shape[1])],
        sample_ids=[f"S{i}" for i in range(small_data.shape[0])],
    )

    assert out.shape == small_data.shape
    np.testing.assert_array_equal(out, r_output.T)
    assert normalizer.vsn_params == mock_params

    # The script receives the *transposed* matrix (proteins as rows).
    _, kwargs = mock_run.call_args
    np.testing.assert_array_equal(kwargs["data"], small_data.T)


@patch("pronoms.normalizers.vsn_normalizer.run_r_script")
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_normalize_generates_default_ids(_mock_setup, mock_run, small_data, mock_params):
    mock_run.return_value = {
        "normalized_data": small_data.T,
        "parameters": mock_params,
    }
    VSNNormalizer().normalize(small_data)

    kwargs = mock_run.call_args.kwargs
    assert kwargs["row_names"] == [f"Protein_{i}" for i in range(small_data.shape[1])]
    assert kwargs["col_names"] == [f"Sample_{i}" for i in range(small_data.shape[0])]


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@patch("pronoms.normalizers.vsn_normalizer.run_r_script")
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_non_finite_input_short_circuits_before_r_call(
    _mock_setup,
    mock_run,
    bad_value,
):
    bad = np.array([[10, 20, 30], [20, bad_value, 60], [30, 60, 90]], dtype=float)
    with pytest.raises(ValueError, match=r"NaN or Inf"):
        VSNNormalizer().normalize(bad)
    mock_run.assert_not_called()


@patch("pronoms.normalizers.vsn_normalizer.run_r_script")
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_run_r_script_failure_is_wrapped_as_value_error(
    _mock_setup,
    mock_run,
    small_data,
):
    mock_run.side_effect = Exception("R script error")
    with pytest.raises(ValueError, match=r"VSN normalization failed"):
        VSNNormalizer().normalize(small_data)


@patch("pronoms.normalizers.vsn_normalizer.run_r_script")
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_missing_normalized_data_in_r_response_raises(
    _mock_setup,
    mock_run,
    small_data,
    mock_params,
):
    mock_run.return_value = {"parameters": mock_params}  # no 'normalized_data'
    with pytest.raises(ValueError, match=r"failed to return normalized data"):
        VSNNormalizer().normalize(small_data)


# ---------------------------------------------------------------------------
# _create_vsn_script -- pure text builder, both branches
# ---------------------------------------------------------------------------


@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_vsn_script_without_reference_sample(_mock_setup):
    normalizer = VSNNormalizer(reference_sample=None, lts_quantile=0.6)
    script = normalizer._create_vsn_script()
    assert "library(vsn)" in script
    assert "ExpressionSet(assayData = input_data)" in script
    assert "lts.quantile = 0.6" in script
    assert "reference =" not in script  # reference omitted entirely
    assert "exprs(vsn_fit)" in script


@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_vsn_script_with_reference_sample_uses_one_indexed_value(_mock_setup):
    """Python is 0-indexed; R is 1-indexed. ``reference_sample=2`` must
    appear as ``reference = 3`` in the generated script."""
    normalizer = VSNNormalizer(reference_sample=2, lts_quantile=0.75)
    script = normalizer._create_vsn_script()
    assert "reference = 3" in script
    assert "lts.quantile = 0.75" in script


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


@patch("pronoms.normalizers.vsn_normalizer.plot_comparison_hexbin")
@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_plot_comparison_forwards_args_to_hexbin_helper(
    _mock_setup,
    mock_helper,
    small_data,
):
    mock_helper.return_value = MagicMock(spec=plt.Figure)
    normalizer = VSNNormalizer()
    fig = normalizer.plot_comparison(small_data, small_data)

    mock_helper.assert_called_once_with(
        before_data=small_data,
        after_data=small_data,
        figsize=(8, 8),
        title="VSN Normalization Comparison (glog2 vs log2)",
        gridsize=50,
        cmap="viridis",
        transform_original="log2",
        log_axes=False,
    )
    assert fig is mock_helper.return_value


# ---------------------------------------------------------------------------
# Real R-backed integration -- skipped automatically when R / vsn are missing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_R_VSN, reason="R or VSN package not available")
def test_real_r_vsn_smoke():
    """End-to-end against the real ``vsn`` package; verifies output shape and
    that fitted parameters are populated."""
    rng = np.random.default_rng(0)
    n_samples, n_features = 5, 30
    base = rng.lognormal(mean=5, sigma=1, size=(n_samples, n_features))
    out = VSNNormalizer().normalize(base)

    assert out.shape == base.shape
    assert np.all(np.isfinite(out))
