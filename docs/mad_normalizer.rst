MADNormalizer
=============

The ``MADNormalizer`` is a robust scaling method that centers each sample by subtracting its median and scales by the Median Absolute Deviation (MAD). This approach reduces the influence of outliers and non-normal distributions, providing a more reliable normalization than standard deviation-based methods.

Overview
--------

MAD normalization is a robust alternative to z-score normalization that uses median-based statistics instead of mean-based ones. The method transforms each sample to have a median of 0 and a MAD-based scale, making it particularly suitable for data with:

- Outliers or extreme values
- Non-normal distributions
- Skewed data where mean and standard deviation are not representative
- Need for robust statistical preprocessing

The approach works by:

1. **(Optional) Log transform**: By default the data is transformed to ``log2(X + 1)`` before computing statistics. Set ``log_transform=False`` to operate on the raw values.
2. **Centering**: Subtracting the (log) median from each value
3. **Scaling**: Dividing by the (log) Median Absolute Deviation (MAD)

This creates standardized samples that are less sensitive to outliers compared to traditional z-score normalization. Working in log-space is the default because it stabilizes variance and matches the typical multiplicative noise structure of mass-spectrometry intensity data.

Key Features
------------

- **Robust to outliers**: Uses median instead of mean, reducing outlier influence
- **Distribution-free**: Works well with non-normal and skewed distributions
- **Standardized output**: Centers data around 0 with MAD-based scaling
- **Preserves relationships**: Maintains relative ordering within samples

Algorithm Details
-----------------

For a data matrix X with shape (n_samples, n_features), let ``Y`` denote the data
on which the statistics are computed. With the default ``log_transform=True`` the
algorithm uses ``Y = log2(X + 1)``; with ``log_transform=False`` it uses ``Y = X``.

1. **Calculate median**: For each sample i, compute median_i = median(Y[i, :])
2. **Calculate MAD**: MAD_i = median(|Y[i, :] - median_i|)
3. **Apply transformation**: X_normalized[i, j] = (Y[i, j] - median_i) / MAD_i

**Mathematical representation** (with ``log_transform=True``, the default):

.. math::

   X_{normalized}[i,j] = \frac{\log_2(X[i,j] + 1) - \text{median}(\log_2(X[i,:] + 1))}{\text{MAD}(\log_2(X[i,:] + 1))}

**Mathematical representation** (with ``log_transform=False``):

.. math::

   X_{normalized}[i,j] = \frac{X[i,j] - \text{median}(X[i,:])}{\text{MAD}(X[i,:])}

where in either case:

.. math::

   \text{MAD}(Y[i,:]) = \text{median}(|Y[i,:] - \text{median}(Y[i,:])|)

**Example** (``log_transform=False``): For sample [1, 5, 10, 100]:

- Median = 7.5
- MAD = median([6.5, 2.5, 2.5, 92.5]) = 4.5
- Normalized ≈ [-1.44, -0.56, 0.56, 20.56]

Parameters
----------

.. autoclass:: pronoms.normalizers.MADNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic MAD normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import MADNormalizer
   
   # Create sample data with outliers
   data = np.array([
       [10, 20, 15, 25, 1000],    # Sample 1: with outlier
       [100, 120, 110, 130, 105], # Sample 2: normal range
       [5, 8, 6, 9, 7]            # Sample 3: low values
   ])
   
   # Create and apply normalizer.
   # By default, log_transform=True, so statistics are computed on log2(X + 1).
   # Pass log_transform=False to operate on the raw values instead.
   normalizer = MADNormalizer()  # log_transform=True (default)
   normalized_data = normalizer.normalize(data)

   print("Original data:")
   print(data)
   print("\nMAD normalized data (computed in log2 space):")
   print(normalized_data)

   # Check centering (medians should be ~0 in either mode)
   print("\nSample medians after normalization:")
   for i, sample in enumerate(normalized_data):
       print(f"Sample {i+1}: {np.median(sample):.6f}")

   # To reproduce the worked example above (raw-scale MAD):
   raw_normalizer = MADNormalizer(log_transform=False)
   raw_normalized = raw_normalizer.normalize(data)

Visualization:

.. code-block:: python

   # Visualize the normalization effect
   fig = normalizer.plot_comparison(data, normalized_data)
   fig.show()

When to Use
-----------

MADNormalizer is particularly useful when:

- **Outliers present**: Data contains extreme values that would skew mean/std-based methods
- **Non-normal distributions**: Data is skewed or has heavy tails
- **Robust preprocessing needed**: When stability against outliers is important
- **Proteomics data**: Common in mass spectrometry data with occasional extreme measurements
- **Quality control**: When some samples may have measurement artifacts

Considerations
--------------

- **Zero MAD handling**: Samples with zero MAD (all identical values) cannot be scaled and will raise a ``ValueError``
- **Negative values with log_transform=True**: The default ``log_transform=True`` requires all input values to be non-negative; negative inputs raise a ``ValueError``. Use ``log_transform=False`` for data that may contain negatives
- **Scale interpretation**: MAD-based scaling differs from standard deviation scaling, and with the default log transform the output is on a log2 scale rather than the original scale
- **Computational cost**: Slightly more expensive than mean/std-based methods due to median calculations
- **Distribution assumptions**: While robust, still assumes some variability within samples

See Also
--------

- :doc:`median_normalizer`: For median-based scaling without robust standardization
- :doc:`quantile_normalizer`: For making distributions identical across samples
- :doc:`rank_normalizer`: For rank-based transformation that handles outliers differently
- :doc:`vsn_normalizer`: For variance-stabilizing normalization
   