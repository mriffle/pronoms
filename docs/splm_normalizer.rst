.. _splm_normalizer:

==================
SPLM Normalizer
==================

Stable Protein Log-Mean Normalization (SPLM) identifies a subset of proteins (features) that are stably expressed across samples based on their low variability in log-transformed space. It then uses the mean log-intensity of these stable proteins to derive scaling factors, normalizing all samples accordingly.

SPLM normalization is ideal when you expect a subset of proteins to remain constant across conditions, allowing you to correct for technical variation using only these stable features.

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

Usage Example
-------------

.. code-block:: python

    import numpy as np
    from pronoms.normalizers import SPLMNormalizer

    # Example data (3 samples, 5 features/proteins)
    # Some proteins are more variable than others
    data = np.array([
        [100, 1000, 50, 500, 200],  # Stable: F1, F3. Variable: F2, F4, F5
        [110, 1500, 55, 600, 210],  # Stable: F1, F3. Variable: F2, F4, F5
        [ 90, 800, 45, 400, 190]   # Stable: F1, F3. Variable: F2, F4, F5
    ])

    # Initialize the normalizer
    # Identify top 2 most stable proteins (lowest log-CV)
    normalizer = SPLMNormalizer(num_stable_proteins=2)

    # Normalize the data
    normalized_data = normalizer.normalize(data)

    print("Original Data:\n", data)
    print("\nSPLM Normalized Data:\n", normalized_data)

    # You can inspect the identified stable features and scaling factors
    # print("\nIndices of Stable Features:", normalizer.stable_feature_indices_)
    # print("Log Scaling Factors:", normalizer.log_scaling_factors_)
    # print("Grand Mean Log Scaling Factor:", normalizer.grand_mean_log_scaling_factor_)

Class Documentation
-------------------

.. automodule:: pronoms.normalizers.splm_normalizer
   :members:
   :undoc-members:
   :show-inheritance:
