"""Tests for ``SPLMNormalizer`` (Stable Protein Log-Mean Normalization).

The algorithm picks the ``num_stable_proteins`` features with the smallest
log-space coefficient of variation, then in log space subtracts each sample's
mean over those features and re-centers by the grand mean.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pronoms.normalizers import SPLMNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data() -> np.ndarray:
    """5 samples × 10 features, with features 0,1,2 the most stable."""
    return np.array(
        [
            [100, 101, 99, 50, 60, 70, 10, 200, 30, 500],
            [102, 100, 98, 55, 65, 75, 15, 250, 35, 550],
            [98, 99, 101, 45, 55, 65, 5, 150, 25, 450],
            [101, 102, 100, 52, 62, 72, 12, 220, 32, 520],
            [99, 98, 102, 48, 58, 68, 8, 180, 28, 480],
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_defaults():
    normalizer = SPLMNormalizer()
    assert normalizer.num_stable_proteins == 100
    assert normalizer.epsilon == 1e-6
    assert normalizer.stable_protein_indices is None
    assert normalizer.log_scaling_factors is None
    assert normalizer.grand_mean_log_scaling_factor is None


def test_init_custom():
    normalizer = SPLMNormalizer(num_stable_proteins=50, epsilon=1e-5)
    assert normalizer.num_stable_proteins == 50
    assert normalizer.epsilon == 1e-5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_stable_proteins": 0}, r"num_stable_proteins must be a positive integer"),
        ({"num_stable_proteins": -10}, r"num_stable_proteins must be a positive integer"),
        ({"num_stable_proteins": 10.5}, r"num_stable_proteins must be a positive integer"),
        ({"epsilon": -1e-6}, r"epsilon must be a non-negative number"),
    ],
)
def test_init_rejects_invalid_params(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SPLMNormalizer(**kwargs)


# ---------------------------------------------------------------------------
# Numerical correctness
# ---------------------------------------------------------------------------


def test_stable_protein_selection(data):
    """Indices 0,1,2 are the lowest-CV features in the fixture."""
    normalizer = SPLMNormalizer(num_stable_proteins=3)
    normalizer.normalize(data)
    assert_array_equal(np.sort(normalizer.stable_protein_indices), [0, 1, 2])


def test_log_scaling_factors_match_closed_form(data):
    n_stable = 3
    epsilon = 1e-6
    normalizer = SPLMNormalizer(num_stable_proteins=n_stable, epsilon=epsilon)
    normalizer.normalize(data)

    X_log = np.log(data + epsilon)
    expected_factors = X_log[:, [0, 1, 2]].mean(axis=1)

    assert_allclose(normalizer.log_scaling_factors, expected_factors, rtol=1e-12)
    assert normalizer.grand_mean_log_scaling_factor == pytest.approx(float(expected_factors.mean()), rel=1e-12)


def test_post_normalization_stable_log_means_are_uniform(data):
    """After normalization the stable-protein log-mean must be equal across samples
    (this is the defining property of SPLM)."""
    normalizer = SPLMNormalizer(num_stable_proteins=3)
    normalized = normalizer.normalize(data)

    norm_log = np.log(normalized + normalizer.epsilon)
    stable_means_after = norm_log[:, normalizer.stable_protein_indices].mean(axis=1)
    assert_allclose(stable_means_after, normalizer.grand_mean_log_scaling_factor, rtol=1e-6)


def test_output_is_non_negative(data):
    normalizer = SPLMNormalizer(num_stable_proteins=3)
    normalized = normalizer.normalize(data)
    assert np.all(normalized >= 0.0)


def test_output_shape_and_dtype(data):
    normalizer = SPLMNormalizer(num_stable_proteins=3)
    normalized = normalizer.normalize(data)
    assert normalized.shape == data.shape
    assert normalized.dtype == np.float64


def test_constant_data_is_near_identity(data):
    """All-constant input has CV=0 everywhere; output should equal input."""
    constant = np.full_like(data, 100.0)
    normalizer = SPLMNormalizer(num_stable_proteins=3)
    normalized = normalizer.normalize(constant)
    assert_allclose(normalized, constant, rtol=1e-5, atol=1e-5)


def test_zero_variance_protein_is_picked_as_stable(data):
    """A truly constant column should be picked first (CV=0)."""
    perturbed = data.copy()
    perturbed[:, 4] = 50.0
    normalizer = SPLMNormalizer(num_stable_proteins=1)
    normalizer.normalize(perturbed)
    assert normalizer.stable_protein_indices[0] == 4


def test_cv_is_computed_in_linear_space():
    """Selection must rank proteins by linear-space CV (std/mean), not by a
    log-space ratio. With a noisy low-mean group whose log-means are negative
    and a stable high-mean group, only the high-mean group should be picked.
    A log-space ratio std(log X)/mean(log X) flips sign and silently picks the
    noisy low-mean proteins instead."""
    rng = np.random.default_rng(42)
    n_samples = 5
    n_per_group = 50

    noisy_means = rng.uniform(0.4, 0.6, n_per_group)
    stable_means = rng.uniform(4.5, 5.5, n_per_group)
    all_means = np.concatenate([noisy_means, stable_means])
    n_features = 2 * n_per_group

    X = np.empty((n_samples, n_features))
    for j in range(n_features):
        X[:, j] = rng.normal(all_means[j], 0.05, n_samples)
    X = np.maximum(X, 1e-3)

    normalizer = SPLMNormalizer(num_stable_proteins=10)
    normalizer.normalize(X)

    # Stable proteins must come from the high-mean group (indices >= n_per_group),
    # whose linear-space CV is much smaller despite both groups having similar
    # log-space std.
    assert np.all(normalizer.stable_protein_indices >= n_per_group)


def test_all_features_used_when_n_stable_equals_features(data):
    n_stable = data.shape[1]
    normalizer = SPLMNormalizer(num_stable_proteins=n_stable)
    normalizer.normalize(data)
    assert_array_equal(np.sort(normalizer.stable_protein_indices), np.arange(n_stable))


# ---------------------------------------------------------------------------
# Validation / error behaviour
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(data, bad_value):
    bad = data.copy()
    bad[0, 0] = bad_value
    with pytest.raises(ValueError, match=r"NaN or Inf"):
        SPLMNormalizer().normalize(bad)


def test_more_stable_proteins_than_features_raises(data):
    normalizer = SPLMNormalizer(num_stable_proteins=data.shape[1] + 1)
    with pytest.raises(ValueError, match=r"cannot be greater than the number of features"):
        normalizer.normalize(data)


def test_zero_features_raises():
    """The shared validator rejects empty data before SPLM's own checks."""
    normalizer = SPLMNormalizer(num_stable_proteins=1)
    with pytest.raises(ValueError, match=r"Input data cannot be empty"):
        normalizer.normalize(np.empty((3, 0)))


# ---------------------------------------------------------------------------
# Plotting (real call)
# ---------------------------------------------------------------------------


def test_plot_comparison_returns_figure(data):
    normalizer = SPLMNormalizer(num_stable_proteins=3)
    normalized = normalizer.normalize(data)

    fig = normalizer.plot_comparison(data, normalized)
    ax = fig.axes[0]

    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "SPLM Normalization Comparison"
