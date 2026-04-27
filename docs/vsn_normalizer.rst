VSNNormalizer
=============

The ``VSNNormalizer`` performs Variance Stabilizing Normalization (VSN). It applies a data-driven arcsinh transformation that flattens variance across the full intensity range, making downstream statistical analyses more reliable and robust to intensity-dependent noise.

As of version 0.2, ``VSNNormalizer`` is implemented in **pure Python** (NumPy + SciPy). The previous R/rpy2 backend has been removed; an R installation is no longer required to use this normalizer.

Overview
--------

Variance Stabilizing Normalization addresses a common problem in high-throughput biological data: the variance of measurements often depends on their intensity level. In proteomics and microarray data, low-intensity measurements typically have lower variance while high-intensity measurements have higher variance. This heteroscedasticity can bias statistical analyses.

VSN works by:

1. **Learning transformation parameters**: per-sample offset ``a_j`` and log-scale-factor ``beta_j`` are estimated by maximum profile likelihood.
2. **Applying transformation**: ``h_ij = arsinh(exp(beta_j) * y_ij + a_j)``, which behaves like ``log(y_ij)`` for large positive ``y_ij`` and is well-defined for zero or negative inputs.
3. **Calibrating across samples**: parameters are estimated jointly so per-sample intensities become comparable.
4. **Robust subset selection**: a least-trimmed-squares (LTS) iteration trims a configurable upper quantile of high-residual rows from each intensity slice, so the fit is not dominated by outliers.

Key Features
------------

- **Variance stabilization**: variance becomes approximately constant across intensity ranges.
- **Data-driven**: parameters are learned from the data, not predetermined.
- **Cross-sample calibration**: per-sample affine pre-transform aligns scales.
- **Robust fitting**: LTS reweighting suppresses the influence of outlier features.
- **No R dependency**: native NumPy/SciPy; no rpy2, no R install required.
- **Vectorized**: closed-form analytic gradient and fully vectorized objective; uses SciPy's L-BFGS-B.

Algorithm Details
-----------------

The model is

.. math::

   h_{ij} = \mathrm{arsinh}(b_j \cdot y_{ij} + a_j), \quad b_j = \exp(\beta_j) > 0

where :math:`y_{ij}` is the raw intensity for feature *i* in sample *j*. ``a_j`` (offset) and ``β_j`` (log-scaling-factor) are per-sample parameters; ``β`` is parametrized in log-space to keep ``b_j`` strictly positive.

Fitting minimizes the negative profile log-likelihood

.. math::

   \mathcal{L}(\theta) = \frac{n_t}{2}\,\log(2\pi\sigma^2) + \frac{1}{2}\sum_{i,j} \log(1 + Y_{ij}^2) - n_{\mathrm{features}} \sum_j \beta_j

where :math:`Y_{ij} = b_j y_{ij} + a_j`, the per-feature mean ``μ_i`` and the variance ``σ²`` are profiled out (closed-form), and ``n_t`` is the total number of (feature, sample) cells.

The robust LTS step then iterates:

1. Apply the current ``(a, β)`` to all features.
2. Compute the per-feature residual variance over samples.
3. Slice features into 5 intensity bins by rank of the per-feature mean.
4. Within each bin, keep features whose residual variance is below the ``lts_quantile`` percentile (and always keep the lowest-intensity slice in full).
5. Refit on this subset, warm-starting from the previous parameters.

Up to 7 LTS iterations are performed (matches R-VSN). The final transform is converted to a log2-comparable scale via

.. math::

   h_{ij}^{\log_2} = \frac{\mathrm{arsinh}(b_j y_{ij} + a_j)}{\ln 2} - h_{\mathrm{offset}}, \quad
   h_{\mathrm{offset}} = \log_2\!\big(2 \exp(\overline{\beta_j})\big).

Numerical agreement with R-VSN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The native engine matches the Bioconductor ``vsn`` package's ``vsn2`` output to ~1e-6 (and ``a``/``β`` parameters to ~1e-6) on realistic proteomics-shaped inputs (e.g. the canonical kidney 8704×2 dataset). On smaller, harder synthetic inputs, scipy's L-BFGS-B and R's ``lbfgsb`` may converge to slightly different local optima on the near-flat profile-likelihood surface; the resulting outputs typically agree within a few times 0.01 on the log2 scale, which is well within the noise of any reasonable downstream analysis.

Parameters
----------

.. autoclass:: pronoms.normalizers.VSNNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import VSNNormalizer

   rng = np.random.default_rng(42)
   data = np.array([
       rng.normal([100, 1000, 10000], [10, 100, 1000]),   # Sample 1
       rng.normal([120, 1200, 12000], [12, 120, 1200]),   # Sample 2
       rng.normal([80,  800,  8000],  [8,  80,  800]),    # Sample 3
   ])

   normalizer = VSNNormalizer()
   normalized = normalizer.normalize(data)

   # Inspect fitted parameters (plain Python dict)
   params = normalizer.vsn_params
   print("a:",       params["a"])
   print("beta:",    params["b_log"])
   print("sigma^2:", params["sigsq"])

Visualization:

.. code-block:: python

   fig = normalizer.plot_comparison(data, normalized)
   fig.show()


When to Use
-----------

VSNNormalizer is particularly useful when:

- **Intensity-dependent variance**: data shows increasing variance with higher intensities.
- **Proteomics applications**: mass-spectrometry data with heteroscedastic noise.
- **Microarray data**: gene expression with intensity-dependent variance.
- **Statistical analysis preparation**: before methods that assume constant variance.
- **Cross-sample comparison**: when samples must be on truly comparable scales.

Considerations
--------------

- **At least two samples** are required (rows of the input matrix). Single-sample inputs raise ``ValueError``.
- **NaN / Inf** values must be handled before calling ``normalize``; non-finite cells raise ``ValueError``.
- **Computational cost**: more expensive than simple scaling methods due to L-BFGS-B parameter estimation, but typically completes in well under a second on realistic proteomics matrices.
- **Output scale**: the result is on a log2-comparable, variance-stabilized scale; not in original intensity units.
- **Hyperparameter**: ``lts_quantile=0.75`` is the pronoms default; pass ``1.0`` to disable the LTS step (single ML fit).

See Also
--------

- :doc:`median_normalizer`: simple scaling-based normalization.
- :doc:`quantile_normalizer`: making distributions identical across samples.
- :doc:`mad_normalizer`: robust normalization via median absolute deviation.
- :doc:`rank_normalizer`: rank-based transformation.

Citation
--------
Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M. Variance stabilization applied to microarray data calibration and to the quantification of differential expression. *Bioinformatics*. 2002;18 Suppl 1:S96–104. `doi:10.1093/bioinformatics/18.suppl_1.s96 <https://doi.org/10.1093/bioinformatics/18.suppl_1.s96>`__, `PMID: 12169536 <https://pubmed.ncbi.nlm.nih.gov/12169536/>`__
