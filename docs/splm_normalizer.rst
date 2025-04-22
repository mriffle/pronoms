.. _splm_normalizer:

==================
SPLM Normalizer
==================

Stable Protein Log-Mean Normalization.

This method normalizes proteomics data by identifying a subset of proteins
(features) deemed 'stable' based on their low variability across samples
in log-transformed space. The normalization then adjusts sample intensities
so that the average log-intensity of these stable proteins is consistent
across all samples.

The core steps are:

1.  **Log Transformation**: The input intensity matrix `X` is log-transformed
    using `log(X + epsilon)`, where `epsilon` is a small constant to avoid
    issues with zero or negative intensities.
2.  **Calculate Log-CV**: The coefficient of variation (CV = standard deviation / mean)
    is calculated for each protein (feature) across all samples using the
    log-transformed data.
3.  **Identify Stable Proteins**: Proteins are ranked by their log-CV. The
    `num_stable_proteins` with the *lowest* log-CV are selected as the stable
    reference set.
4.  **Calculate Log Scaling Factors**: For each sample, a log-scaling factor is
    calculated as the mean log-intensity of only the stable proteins identified
    in the previous step.
5.  **Calculate Grand Mean**: The overall mean of these sample-specific
    log-scaling factors is computed (the grand mean log scaling factor).
6.  **Normalize in Log Space**: Each sample's log-intensities are adjusted by
    subtracting its specific log-scaling factor and adding the grand mean log
    scaling factor.
7.  **Back-transform**: The adjusted log-intensities are transformed back to the
    original intensity scale using the exponential function `exp(adjusted_log_X) - epsilon`.

This process effectively centers the distribution of stable protein intensities
across samples while preserving the overall intensity profile.

.. automodule:: pronoms.normalizers.splm_normalizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pronoms.normalizers.SPLMNormalizer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
