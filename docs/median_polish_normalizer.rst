.. _median_polish_normalizer:

=========================
Median Polish Normalizer
=========================

Tukey's Median Polish Normalization.

This method applies Tukey's Median Polish algorithm, typically to log-transformed
data, to decompose a matrix into overall, row, column, and residual effects.
It iteratively removes the median from rows and then columns until convergence.

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

.. automodule:: pronoms.normalizers.median_polish_normalizer
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pronoms.normalizers.MedianPolishNormalizer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
