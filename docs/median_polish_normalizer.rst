.. _median_polish_normalizer:

=========================
Median Polish Normalizer
=========================

Tukey's Median Polish Normalization.

The MedianPolishNormalizer uses Tukey's Median Polish algorithm to iteratively remove row and column medians from a (typically log-transformed) data matrix. This decomposes the data into overall, row, column, and residual effects, helping to remove systematic biases from both samples and features.

Median Polish normalization is especially useful when both row (sample) and column (feature) effects are present, and you want to analyze the residual structure or correct for multiple sources of bias.

The `normalize` method returns the `overall_median + residuals`.
**Important:** If `log_transform=True` was used during initialization, the
returned normalized data remains in **log-space**.

The `plot_comparison` method generates a hexbin plot comparing the original
data (plotted on a log scale on the x-axis) against the normalized data
(y-axis). The y-axis scale matches the output of the `normalize` method
(i.e., it will be log-scale if `log_transform=True` was used).

Use Cases:
 - Removing systematic row (sample) and column (feature) biases.
 - Often used in microarray data analysis and applicable to proteomics.

Usage Example
-------------

.. code-block:: python

    import numpy as np
    from pronoms.normalizers import MedianPolishNormalizer

    # Example data (3 samples, 4 features)
    # Median polish is often applied to log-transformed data
    data = np.log2(np.array([
        [100, 200, 150, 250],
        [120, 220, 160, 260],
        [90, 190, 140, 240]
    ]) + 1) # Add 1 to avoid log2(0)

    # Initialize the normalizer (default is log_transform=True)
    # If data is already log-transformed, set log_transform=False
    # normalizer = MedianPolishNormalizer(log_transform=False)
    normalizer = MedianPolishNormalizer() # Assumes input is NOT log-transformed

    # If your input data is already log-transformed:
    # normalizer = MedianPolishNormalizer(log_transform=False)

    # Normalize the data
    # Note: If log_transform=True was used (or default), output is also log-transformed.
    normalized_data = normalizer.normalize(np.exp2(data)-1) # Pass original scale data if log_transform=True

    # If input was already log-transformed and log_transform=False:
    # normalized_data_log = normalizer.normalize(data)

    print("Original Data (log2 scale):\n", data)
    print("\nMedian Polished Data (log2 scale if log_transform=True):\n", normalized_data)

    # You can also inspect the components:
    # print("\nOverall Median:", normalizer.overall_)
    # print("Row Effects:", normalizer.row_effects_)
    # print("Column Effects:", normalizer.col_effects_)

Class Documentation
-------------------

.. automodule:: pronoms.normalizers.median_polish_normalizer
   :members:
   :undoc-members:
   :show-inheritance:
