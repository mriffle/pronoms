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
    # reference_sample is deprecated; passing it emits a DeprecationWarning.
    with pytest.warns(DeprecationWarning, match=r"reference_sample"):
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
def test_vsn_script_never_emits_reference_argument(_mock_setup):
    """``reference_sample`` is soft-deprecated: passing a value emits a
    DeprecationWarning at construction and the generated R script must NOT
    include ``reference = N`` (R-VSN's reference argument expects a fitted
    vsn object, not an integer index, so the previous behavior crashed at
    runtime). The script falls back to the default vsn2 call."""
    with pytest.warns(DeprecationWarning, match=r"reference_sample"):
        normalizer = VSNNormalizer(reference_sample=2, lts_quantile=0.75)
    script = normalizer._create_vsn_script()
    assert "reference =" not in script
    assert "lts.quantile = 0.75" in script


@patch("pronoms.normalizers.vsn_normalizer.setup_r_environment")
def test_reference_sample_none_does_not_warn(_mock_setup):
    """The default reference_sample=None must remain warning-free."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        VSNNormalizer(reference_sample=None)
        VSNNormalizer()


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
def test_real_r_vsn_returns_finite_output_with_correct_shape():
    """Wrapper-level: shape and finite-ness round-trip cleanly through the
    rpy2 boundary."""
    rng = np.random.default_rng(0)
    base = rng.lognormal(mean=5, sigma=1, size=(5, 30))
    out = VSNNormalizer().normalize(base)

    assert out.shape == base.shape
    assert np.all(np.isfinite(out))


@pytest.mark.skipif(not _HAS_R_VSN, reason="R or VSN package not available")
def test_real_r_vsn_preserves_within_sample_feature_ranks():
    """Wrapper-level orientation check. VSN applies a strictly monotone
    transformation per sample, so within each sample the rank-order of
    features must be identical before and after. If the wrapper accidentally
    transposed the content (returning features × samples instead of samples
    × features), this fails — the shape check alone cannot catch it on a
    near-square or symmetric input."""
    rng = np.random.default_rng(1)
    base = rng.lognormal(mean=5, sigma=1, size=(5, 30))
    out = VSNNormalizer().normalize(base)

    np.testing.assert_array_equal(np.argsort(base, axis=1), np.argsort(out, axis=1))


@pytest.mark.skipif(not _HAS_R_VSN, reason="R or VSN package not available")
def test_real_r_vsn_lts_quantile_reaches_r():
    """Wrapper-level passthrough: ``lts_quantile`` must actually reach
    ``vsn2()`` in R, not just appear in the generated script text. Two
    markedly different quantiles on the same data — with outliers, so the
    LTS regression has something to trim differently — must produce
    different outputs."""
    rng = np.random.default_rng(2)
    base = rng.lognormal(mean=5, sigma=1, size=(5, 30))
    base[0, 0] *= 1000
    base[2, 5] *= 500

    out_low = VSNNormalizer(lts_quantile=0.5).normalize(base)
    out_high = VSNNormalizer(lts_quantile=0.99).normalize(base)

    assert not np.allclose(out_low, out_high)


@pytest.mark.skipif(not _HAS_R_VSN, reason="R or VSN package not available")
def test_real_r_vsn_coefficients_have_expected_size():
    """Wrapper-level extraction: ``vsn_params['coefficients']`` for
    ``calib='affine'`` must contain one (intercept, slope) pair per sample.
    R-VSN returns a (1, n_samples, 2) array under the default unstratified
    fit; our extraction must reach the right slot and yield 2 * n_samples
    finite values."""
    rng = np.random.default_rng(4)
    base = rng.lognormal(mean=5, sigma=1, size=(5, 30))
    normalizer = VSNNormalizer()
    normalizer.normalize(base)

    coef = normalizer.vsn_params.rx2("coefficients")
    arr = np.asarray(coef, dtype=float)
    assert arr.size == 2 * base.shape[0]
    assert np.all(np.isfinite(arr))


@pytest.mark.skipif(not _HAS_R_VSN, reason="R or VSN package not available")
def test_real_r_vsn_reference_sample_is_ignored_with_warning():
    """``reference_sample`` is soft-deprecated. Passing any value must:
    (a) emit DeprecationWarning at construction;
    (b) still produce a valid VSN result that exactly matches the
        ``reference_sample=None`` output — confirming the broken
        ``reference = N`` R-call was dropped, not silently retained."""
    rng = np.random.default_rng(5)
    base = rng.lognormal(mean=5, sigma=1, size=(5, 30))

    with pytest.warns(DeprecationWarning, match=r"reference_sample"):
        normalizer = VSNNormalizer(reference_sample=0)
    out_with_ref = normalizer.normalize(base)
    out_default = VSNNormalizer().normalize(base)

    np.testing.assert_array_equal(out_with_ref, out_default)
