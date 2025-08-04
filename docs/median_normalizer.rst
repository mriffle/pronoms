MedianNormalizer
================

The ``MedianNormalizer`` scales each sample by its median value, then rescales all samples by the mean of medians to preserve overall scale. This simple yet effective normalization method corrects for systematic differences in sample loading or labeling efficiency across samples.

Overview
--------

Median normalization is based on the assumption that most proteins/features are not changing between samples, so differences in median intensity reflect technical rather than biological variation. The method works in two steps:

1. **Sample-wise scaling**: Each sample is divided by its own median
2. **Global rescaling**: All samples are multiplied by the mean of all original medians to preserve the overall data scale

This approach is particularly effective for:

- Correcting for differences in sample loading amounts
- Adjusting for labeling efficiency variations
- Normalizing systematic technical differences across samples
- Preprocessing data where most features are expected to be unchanged

Key Features
------------

- **Scale preservation**: Maintains the overall magnitude of the data through global rescaling
- **Robust to outliers**: Uses median instead of mean, making it less sensitive to extreme values
- **Simple and fast**: Computationally efficient with minimal parameters
- **Widely applicable**: Suitable for most proteomics and genomics datasets

Algorithm Details
-----------------

For a data matrix X with shape (n_samples, n_features):

1. **Calculate sample medians**: For each sample i, compute median_i = median(X[i, :])
2. **Scale samples**: X_scaled[i, :] = X[i, :] / median_i
3. **Calculate global scaling factor**: global_factor = mean(all medians)
4. **Apply global rescaling**: X_normalized[i, :] = X_scaled[i, :] * global_factor

**Mathematical representation**:

.. math::

   X_{normalized}[i,j] = \frac{X[i,j]}{\text{median}(X[i,:])} \times \frac{1}{n} \sum_{k=1}^{n} \text{median}(X[k,:])

Parameters
----------

.. autoclass:: pronoms.normalizers.MedianNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic median normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import MedianNormalizer
   
   # Create sample data with different loading amounts
   data = np.array([
       [100, 200, 300, 400],  # Sample 1: high loading
       [50, 100, 150, 200],   # Sample 2: medium loading  
       [25, 50, 75, 100]      # Sample 3: low loading
   ])
   
   # Create and apply normalizer
   normalizer = MedianNormalizer()
   normalized_data = normalizer.normalize(data)
   
   print("Original data:")
   print(data)
   print("\nNormalized data:")
   print(normalized_data)
   
   # Check that sample medians are now equal
   print("\nSample medians after normalization:")
   for i, sample in enumerate(normalized_data):
       print(f"Sample {i+1}: {np.median(sample):.2f}")

Visualization:

.. code-block:: python

   # Visualize the normalization effect
   fig = normalizer.plot_comparison(data, normalized_data)
   fig.show()

When to Use
-----------

MedianNormalizer is particularly useful when:

- **Sample loading varies**: Different amounts of sample were loaded across runs
- **Labeling efficiency differs**: Variations in chemical labeling or sample preparation
- **Most features unchanged**: The majority of proteins/features are not expected to change between conditions
- **Simple normalization needed**: When a straightforward, robust method is preferred
- **Preprocessing step**: As an initial normalization before more complex methods

Considerations
--------------

- **Assumes stable features**: Method assumes most features are not changing between samples
- **May over-normalize**: If many features are truly changing, median-based normalization may remove real biological signal
- **Not suitable for sparse data**: Performance may be poor with many zero or missing values
- **Global scaling**: The global rescaling step may not be appropriate for all applications

See Also
--------

- :doc:`quantile_normalizer`: For making entire distributions identical across samples
- :doc:`mad_normalizer`: For robust normalization using median absolute deviation
- :doc:`rank_normalizer`: For rank-based normalization that handles outliers differently
- :doc:`vsn_normalizer`: For variance-stabilizing normalization