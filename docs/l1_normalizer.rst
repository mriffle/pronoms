L1Normalizer
============

The ``L1Normalizer`` adjusts each sample so that its sum of absolute values (L1 norm) equals 1. This scaling method removes differences in total signal between samples, making them directly comparable regardless of overall intensity while preserving the relative proportions of features within each sample.

Overview
--------

L1 normalization scales each sample by dividing all values by the sum of their absolute values. This creates a probabilistic interpretation where each feature represents a proportion of the total signal. The method is particularly useful when:

- Total signal varies between samples due to technical factors
- You want to focus on relative feature proportions rather than absolute intensities
- Samples need to contribute equally to downstream analyses
- Working with compositional data or when interested in relative abundances

This approach is commonly used in:

- Proteomics for correcting sample loading differences
- Genomics for normalizing library size effects
- Machine learning for feature scaling
- Any analysis where relative proportions are more important than absolute values

Key Features
------------

- **Unit norm**: All samples have exactly the same L1 norm (sum of absolute values = 1)
- **Proportion preservation**: Relative relationships between features within samples are maintained
- **Simple and fast**: Computationally efficient with no parameters to tune
- **Interpretable**: Normalized values represent proportions of the total signal

Algorithm Details
-----------------

For a data matrix X with shape (n_samples, n_features):

1. **Calculate L1 norm**: For each sample i, compute L1_norm_i = Σ|X[i, j]|
2. **Scale sample**: X_normalized[i, :] = X[i, :] / L1_norm_i

**Mathematical representation**:

.. math::

   X_{normalized}[i,j] = \frac{X[i,j]}{\sum_{k=1}^{n} |X[i,k]|}

**Example**: For sample [10, 20, 30, 40]:
- L1 norm = |10| + |20| + |30| + |40| = 100
- Normalized = [0.1, 0.2, 0.3, 0.4]

Parameters
----------

.. autoclass:: pronoms.normalizers.L1Normalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic L1 normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import L1Normalizer
   
   # Create sample data with different total intensities
   data = np.array([
       [10, 20, 30, 40],    # Sample 1: total = 100
       [5, 10, 15, 20],     # Sample 2: total = 50
       [100, 200, 300, 400] # Sample 3: total = 1000
   ])
   
   # Create and apply normalizer
   normalizer = L1Normalizer()
   normalized_data = normalizer.normalize(data)
   
   print("Original data:")
   print(data)
   print("\nNormalized data:")
   print(normalized_data)
   
   # Verify L1 norms are 1
   print("\nL1 norms after normalization:")
   for i, sample in enumerate(normalized_data):
       l1_norm = np.sum(np.abs(sample))
       print(f"Sample {i+1}: {l1_norm:.6f}")

Visualization:

.. code-block:: python

   # Visualize the normalization effect
   fig = normalizer.plot_comparison(data, normalized_data)
   fig.show()

When to Use
-----------

L1Normalizer is particularly useful when:

- **Sample loading varies**: Different amounts of total protein/material across samples
- **Library size effects**: In genomics, when sequencing depth varies between samples
- **Compositional analysis**: When interested in relative proportions rather than absolute amounts
- **Equal contribution needed**: When samples should contribute equally to downstream analyses
- **Sparse data**: Works well with data containing many zeros

Considerations
--------------

- **Scale dependency**: Results depend on the absolute scale of the original data
- **Zero handling**: Samples with all zero values cannot be normalized (division by zero)
- **Outlier sensitivity**: Large outliers can dominate the L1 norm and compress other values
- **Information loss**: Absolute magnitude information is lost, only proportions are preserved
- **Not suitable for negative-dominant data**: Less meaningful when most values are negative

See Also
--------

- :doc:`median_normalizer`: For scaling-based normalization that preserves overall scale
- :doc:`quantile_normalizer`: For making distributions identical across samples
- :doc:`mad_normalizer`: For robust normalization using median absolute deviation
- :doc:`vsn_normalizer`: For variance-stabilizing normalization