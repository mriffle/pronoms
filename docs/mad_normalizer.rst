.. _mad_normalizer:

===============
MAD Normalizer
===============

Median Absolute Deviation (MAD) Normalization.

This method standardizes each sample (row) based on its median and Median Absolute Deviation (MAD).
It's a robust method for scaling data, less sensitive to outliers than standard deviation.

The normalization formula for a sample `x` is:

.. math::

   normalized_x = \frac{x - median(x)}{MAD(x)}

where

.. math::

   MAD(x) = median(|x - median(x)|)

Use Cases:
 - Robust scaling of samples when outliers might be present.
 - Standardizing sample distributions based on robust statistics.

.. automodule:: pronoms.normalizers.mad_normalizer
   :members:
   :undoc-members:
   :show-inheritance:
