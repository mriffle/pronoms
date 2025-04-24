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

Usage Example
-------------

.. code-block:: python

    import numpy as np
    from pronoms.normalizers import MADNormalizer

    # Example data (3 samples, 4 features)
    data = np.array([
        [10, 20, 15, 25],
        [100, 120, 110, 130], # Sample with larger values
        [5, 8, 6, 9]
    ])

    # Initialize the normalizer
    normalizer = MADNormalizer()

    # Normalize the data
    normalized_data = normalizer.normalize(data)

    print("Original Data:\n", data)
    print("\nMAD Normalized Data:\n", normalized_data)

    # You can also check the calculated medians and MADs
    # print("\nMedians:", normalizer.medians_)
    # print("MADs:", normalizer.mads_)
