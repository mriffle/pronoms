"""Tests for ``DirectLFQNormalizer``.

The directlfq pipeline itself is exercised via a small un-mocked smoke run that
verifies the wrapper's contract end-to-end (output shapes, ID parsing, NumPy
return types). Targeted mock-based tests cover input validation, custom
parameter wiring, and the optional between-sample normalization branch.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from pronoms.normalizers.directlfq_normalizer import DirectLFQNormalizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def raw_data() -> np.ndarray:
    """3 samples × 5 ions."""
    return np.array(
        [
            [10, 20, 30, 40, 50],
            [12, 22, 32, 42, 52],
            [8, 18, 28, 38, 48],
        ],
        dtype=float,
    )


@pytest.fixture
def protein_list() -> list[str]:
    return ["ProtA", "ProtA", "ProtB", "ProtC", "ProtC"]


@pytest.fixture
def peptide_list() -> list[str]:
    return ["PepA1", "PepA2", "PepB1", "PepC1", "PepC2"]


@pytest.fixture
def mock_protein_output() -> pd.DataFrame:
    return pd.DataFrame(
        {"sample_1": [100.0, 300.0], "sample_2": [120.0, 320.0], "sample_3": [80.0, 280.0]},
        index=["ProtA", "ProtC"],
    )


@pytest.fixture
def mock_ion_output() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_1": [10.0, 20.0, 40.0, 50.0],
            "sample_2": [12.0, 22.0, 42.0, 52.0],
            "sample_3": [8.0, 18.0, 38.0, 48.0],
        },
        index=["PepA1", "PepA2", "PepC1", "PepC2"],
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_init_defaults():
    normalizer = DirectLFQNormalizer()
    assert normalizer.num_cores is None
    assert normalizer.do_between_sample_norm is True
    assert normalizer.min_nonan == 1
    assert normalizer.n_quad_ions == 10
    assert normalizer.n_quad_samples == 50


def test_init_custom():
    normalizer = DirectLFQNormalizer(
        num_cores=2,
        do_between_sample_norm=False,
        min_nonan=2,
        n_quad_ions=5,
        n_quad_samples=20,
    )
    assert normalizer.num_cores == 2
    assert normalizer.do_between_sample_norm is False
    assert normalizer.min_nonan == 2
    assert normalizer.n_quad_ions == 5
    assert normalizer.n_quad_samples == 20


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_protein_list_length_mismatch_raises(raw_data, protein_list, peptide_list):
    with pytest.raises(ValueError, match=r"Lengths of 'proteins' and 'peptides'"):
        DirectLFQNormalizer().normalize(raw_data, protein_list[:-1], peptide_list)


def test_peptide_list_length_mismatch_raises(raw_data, protein_list, peptide_list):
    with pytest.raises(ValueError, match=r"Lengths of 'proteins' and 'peptides'"):
        DirectLFQNormalizer().normalize(raw_data, protein_list, peptide_list[:-1])


def test_1d_input_raises(raw_data, protein_list, peptide_list):
    with pytest.raises(ValueError, match=r"X must be a 2-D array"):
        DirectLFQNormalizer().normalize(raw_data[0, :], protein_list, peptide_list)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_input_raises(bad_value, protein_list, peptide_list):
    bad = np.array([[1.0, 2.0, bad_value, 4.0, 5.0], [4.0, 5.0, 6.0, 7.0, 8.0]])
    with pytest.raises(ValueError, match=r"DirectLFQ cannot handle NaN or Inf"):
        DirectLFQNormalizer().normalize(bad, protein_list, peptide_list)


# ---------------------------------------------------------------------------
# Mocked pipeline -- verifies the wrapper drives directlfq correctly
# ---------------------------------------------------------------------------


def _build_directlfq_outputs(mock_protein, mock_ion, df_in):
    """Mimic the (protein_df, ion_df) shape that directlfq returns."""
    prot = mock_protein.reset_index().rename(columns={"index": "protein"})
    ion = mock_ion.reset_index().rename(columns={"index": "ion"})
    protein_map = df_in.set_index("ion")["protein"]
    ion["protein"] = ion["ion"].map(protein_map)
    cols = ["protein", "ion"] + [c for c in ion.columns if c not in {"protein", "ion"}]
    return prot, ion[cols]


@pytest.mark.parametrize(
    ("do_between_sample_norm", "num_cores", "n_quad_samples", "n_quad_ions", "min_nonan"),
    [
        (True, 4, 50, 10, 1),  # default-ish
        (False, 2, 25, 7, 3),  # custom + between-sample-norm disabled
    ],
)
@patch("pronoms.normalizers.directlfq_normalizer.dlcfg.set_global_protein_and_ion_id")
@patch("pronoms.normalizers.directlfq_normalizer.dlcfg.check_wether_to_copy_numpy_arrays_derived_from_pandas")
@patch("pronoms.normalizers.directlfq_normalizer.dlu.index_and_log_transform_input_df")
@patch("pronoms.normalizers.directlfq_normalizer.dlu.sort_input_df_by_protein_and_quant_id")
@patch("pronoms.normalizers.directlfq_normalizer.dlu.remove_allnan_rows_input_df")
@patch("pronoms.normalizers.directlfq_normalizer.dlnorm.NormalizationManagerSamplesOnSelectedProteins")
@patch("pronoms.normalizers.directlfq_normalizer.dlprot.estimate_protein_intensities")
def test_normalize_drives_pipeline_with_correct_params(
    mock_estimate,
    mock_norm_manager,
    mock_remove_nan,
    mock_sort,
    mock_log,
    mock_check_copy,
    mock_set_ids,
    do_between_sample_norm,
    num_cores,
    n_quad_samples,
    n_quad_ions,
    min_nonan,
    raw_data,
    protein_list,
    peptide_list,
    mock_protein_output,
    mock_ion_output,
):
    n_samples = raw_data.shape[0]
    sample_cols = [f"sample_{i + 1}" for i in range(n_samples)]
    df_in = pd.DataFrame(
        {"protein": protein_list, "ion": peptide_list, **{s: raw_data[i, :] for i, s in enumerate(sample_cols)}}
    )

    mock_sort.return_value = df_in
    mock_log.return_value = df_in
    mock_remove_nan.return_value = df_in
    norm_instance = MagicMock()
    norm_instance.complete_dataframe = df_in
    mock_norm_manager.return_value = norm_instance

    prot_df, ion_df = _build_directlfq_outputs(mock_protein_output, mock_ion_output, df_in)
    mock_estimate.return_value = (prot_df, ion_df)

    normalizer = DirectLFQNormalizer(
        do_between_sample_norm=do_between_sample_norm,
        num_cores=num_cores,
        n_quad_samples=n_quad_samples,
        n_quad_ions=n_quad_ions,
        min_nonan=min_nonan,
    )
    norm_prot, norm_ion, prot_ids, pep_ids = normalizer.normalize(raw_data, protein_list, peptide_list)

    mock_set_ids.assert_called_once_with(protein_id="protein", quant_id="ion")
    mock_check_copy.assert_called_once()
    mock_sort.assert_called_once()
    assert_frame_equal(mock_sort.call_args[0][0], df_in)
    mock_log.assert_called_once()
    mock_remove_nan.assert_called_once()

    if do_between_sample_norm:
        mock_norm_manager.assert_called_once()
        assert mock_norm_manager.call_args.kwargs["num_samples_quadratic"] == n_quad_samples
        assert mock_norm_manager.call_args.kwargs["selected_proteins_file"] is None
    else:
        mock_norm_manager.assert_not_called()

    est_kwargs = mock_estimate.call_args.kwargs
    assert est_kwargs["min_nonan"] == min_nonan
    assert est_kwargs["num_samples_quadratic"] == n_quad_ions
    assert est_kwargs["num_cores"] == num_cores

    np.testing.assert_array_equal(norm_prot, mock_protein_output.T.to_numpy(dtype=np.float64))
    np.testing.assert_array_equal(norm_ion, mock_ion_output.T.to_numpy(dtype=np.float64))
    np.testing.assert_array_equal(prot_ids, np.array(["ProtA", "ProtC"], dtype=str))
    np.testing.assert_array_equal(pep_ids, np.array(["PepA1", "PepA2", "PepC1", "PepC2"], dtype=str))


# ---------------------------------------------------------------------------
# Real, un-mocked smoke run against directlfq
# ---------------------------------------------------------------------------


def test_real_directlfq_pipeline_smoke():
    """Tiny end-to-end run with no mocking. Verifies the wrapper produces
    reasonable, well-shaped numerical outputs against the real ``directlfq``
    library -- catches regressions the mock-based tests cannot see."""
    rng = np.random.default_rng(0)
    n_samples = 4
    proteins = ["P1"] * 3 + ["P2"] * 3 + ["P3"] * 3
    peptides = [f"pep_{i}" for i in range(len(proteins))]
    base = np.array(
        [
            1000.0,
            1100.0,
            900.0,  # P1 ions
            500.0,
            550.0,
            450.0,  # P2 ions
            200.0,
            220.0,
            180.0,
        ]
    )  # P3 ions
    sample_factors = np.array([1.0, 1.5, 0.5, 2.0])  # known loading differences
    X = sample_factors[:, None] * base[None, :] + rng.normal(0, 5, (n_samples, len(base)))
    X = np.clip(X, 1.0, None)  # ensure positive

    normalizer = DirectLFQNormalizer(num_cores=1)
    prot_matrix, ion_matrix, prot_ids, pep_ids = normalizer.normalize(X, proteins, peptides)

    # shapes / dtypes / IDs
    assert prot_matrix.shape == (n_samples, 3)
    assert ion_matrix.shape == (n_samples, len(peptides))
    assert prot_matrix.dtype == np.float64
    assert ion_matrix.dtype == np.float64
    assert set(prot_ids.tolist()) == {"P1", "P2", "P3"}
    assert set(pep_ids.tolist()) == set(peptides)

    # finite, non-negative protein intensities
    assert np.all(np.isfinite(prot_matrix))
    assert np.all(prot_matrix > 0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_comparison_calls_hexbin_helper(raw_data):
    """Plot helper is delegated to; verify args are forwarded correctly."""
    with patch("pronoms.normalizers.directlfq_normalizer.create_hexbin_comparison") as mock_helper:
        mock_helper.return_value = MagicMock(spec=plt.Figure)
        normalizer = DirectLFQNormalizer()
        before = np.random.default_rng(0).random((3, 2))
        after = np.random.default_rng(1).random((3, 2))

        fig = normalizer.plot_comparison(before, after, title="Custom", figsize=(12, 6))

        mock_helper.assert_called_once()
        args, kwargs = mock_helper.call_args
        np.testing.assert_array_equal(args[0], before)
        np.testing.assert_array_equal(args[1], after)
        assert kwargs["title"] == "Custom"
        assert kwargs["figsize"] == (12, 6)
        assert kwargs["xlabel"] == "Before DirectLFQ (Protein Intensity)"
        assert kwargs["ylabel"] == "After DirectLFQ (Protein Intensity)"
        assert fig is mock_helper.return_value


def test_plot_comparison_warns_on_shape_mismatch(capsys):
    normalizer = DirectLFQNormalizer()
    with patch("pronoms.normalizers.directlfq_normalizer.create_hexbin_comparison"):
        normalizer.plot_comparison(np.zeros((3, 5)), np.zeros((3, 4)))
    captured = capsys.readouterr()
    assert "Shape mismatch" in captured.out
