QuantileNormalizer
==================

The ``QuantileNormalizer`` applies quantile normalization to make the empirical distributions of all samples identical. This powerful normalization method ensures that after transformation, all samples have exactly the same distribution, making them directly comparable regardless of their original distributions.

Overview
--------

Quantile normalization is based on the principle that samples should have the same distribution if technical variation is removed. The method works by:

1. **Ranking values**: Sort each sample independently to get ranks
2. **Computing reference distribution**: Average values across samples at each rank position
3. **Reassigning values**: Replace original values with the reference distribution values at corresponding ranks

This approach is particularly powerful for:

- Removing systematic differences between samples or batches
- Making samples directly comparable across different experimental conditions
- Integrating data from multiple studies or platforms
- Preprocessing for downstream analyses that assume similar distributions

Key Features
------------

- **Identical distributions**: All samples have exactly the same distribution after normalization
- **Rank preservation**: The relative ordering within each sample is maintained
- **Batch effect removal**: Effective at removing systematic batch-to-batch variation
- **Reference-based**: Uses the average distribution across all samples as the target

Algorithm Details
-----------------

For a data matrix X with shape (n_samples, n_features):

1. **Sort each sample**: For each sample i, create sorted_i = sort(X[i, :])
2. **Compute reference distribution**: ref[j] = mean(sorted_1[j], sorted_2[j], ..., sorted_n[j])
3. **Create rank mapping**: For each sample, map original ranks to reference values
4. **Reassign values**: Replace each original value with the corresponding reference value

**Example**: For samples [1, 3, 2] and [10, 30, 20]:
- Sorted: [1, 2, 3] and [10, 20, 30]
- Reference: [(1+10)/2, (2+20)/2, (3+30)/2] = [5.5, 11, 16.5]
- Result: [5.5, 16.5, 11] and [5.5, 16.5, 11]

Parameters
----------

.. autoclass:: pronoms.normalizers.QuantileNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic quantile normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import QuantileNormalizer
   
   # Create sample data with different distributions
   data = np.array([
       [1, 5, 10, 20],    # Sample 1: low values
       [100, 500, 1000, 2000],  # Sample 2: high values
       [2, 8, 15, 25]     # Sample 3: intermediate values
   ])
   
   # Create and apply normalizer
   normalizer = QuantileNormalizer()
   normalized_data = normalizer.normalize(data)
   
   print("Original data:")
   print(data)
   print("\nNormalized data:")
   print(normalized_data)
   
   # Verify identical distributions
   print("\nSample distributions after normalization:")
   for i, sample in enumerate(normalized_data):
       print(f"Sample {i+1}: {sorted(sample)}")

Visualization:

.. code-block:: python

   # Visualize the normalization effect
   fig = normalizer.plot_comparison(data, normalized_data)
   fig.show()

When to Use
-----------

QuantileNormalizer is particularly useful when:

- **Batch effects present**: Strong systematic differences between experimental batches
- **Cross-platform integration**: Combining data from different measurement platforms
- **Distribution assumptions**: Downstream methods assume samples have similar distributions
- **Direct comparability needed**: When samples must be directly comparable after normalization
- **Large-scale studies**: Multi-center studies where technical variation dominates

Considerations
--------------

- **Strong assumption**: Assumes all samples should have identical distributions
- **May remove biology**: Can eliminate real biological differences between sample groups
- **Rank-based**: Only preserves relative ordering, not absolute differences
- **Reference dependency**: Results depend on the composition of the sample set
- **Not suitable for sparse data**: Performance degrades with many missing or zero values

See Also
--------

- :doc:`median_normalizer`: For simpler scaling-based normalization
- :doc:`rank_normalizer`: For rank-based transformation without enforcing identical distributions
- :doc:`vsn_normalizer`: For variance-stabilizing normalization
- :doc:`mad_normalizer`: For robust normalization using median absolute deviation
