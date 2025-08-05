RankNormalizer
==============

The ``RankNormalizer`` transforms each sample's values to their ranks, where the smallest value receives rank 1 and the largest receives rank N (number of features). This transformation is useful for making data distributions more uniform and reducing the impact of outliers.

Overview
--------

Rank normalization replaces each value in a sample with its rank position when the values are sorted from smallest to largest. This creates a uniform distribution of ranks from 1 to N for each sample, making it particularly useful for:

- Reducing the impact of outliers
- Creating comparable scales across different measurement ranges
- Preprocessing for non-parametric statistical methods
- Making data distributions more uniform

Key Features
------------

- **Tied Value Handling**: When multiple values are identical, they receive the median rank of their group
- **Optional Normalization**: Ranks can be divided by N to create values between 1/N and 1 for comparability across datasets
- **Robust to Outliers**: Extreme values only affect the highest/lowest ranks, not the entire distribution

Algorithm Details
-----------------

For each sample (row) in the data matrix:

1. Sort the values from smallest to largest
2. Assign ranks starting from 1
3. For tied values, assign the median rank of the group
4. Optionally divide all ranks by N (number of features)

**Example with ties**: If values [1, 2, 2, 3] are encountered:
- Value 1 gets rank 1
- Both values of 2 get rank 2.5 (median of ranks 2 and 3)
- Value 3 gets rank 4

Parameters
----------

.. autoclass:: pronoms.normalizers.RankNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic rank normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import RankNormalizer
   
   # Create sample data
   data = np.array([
       [100, 50, 75, 200],  # Sample 1
       [10, 10, 30, 20]     # Sample 2 (with ties)
   ])
   
   # Create and apply normalizer
   normalizer = RankNormalizer()
   normalized_data = normalizer.normalize(data)
   
   print("Original data:")
   print(data)
   print("\nRank-transformed data:")
   print(normalized_data)
   # Output:
   # [[4. 1. 2. 3.]     # Sample 1: ranks of [100,50,75,200]
   #  [2.5 2.5 4. 1.]]  # Sample 2: ranks with ties at 10

Normalized ranks (divide by N):

.. code-block:: python

   # Normalize ranks to [1/N, 1] range
   normalizer = RankNormalizer(normalize_by_n=True)
   normalized_data = normalizer.normalize(data)
   
   print("Normalized rank data (divided by N):")
   print(normalized_data)
   # Output:
   # [[1.    0.25  0.5   0.75 ]     # Sample 1: ranks/4
   #  [0.625 0.625 1.    0.25 ]]    # Sample 2: ranks/4

Visualization:

.. code-block:: python

    # Visualize the transformation effect
    # By default, x-axis shows raw values (log_axes=False)
    fig = normalizer.plot_comparison(data, normalized_data)
    fig.show()
    
    # For data with wide dynamic ranges, use log-transformed x-axis
    fig = normalizer.plot_comparison(data, normalized_data, log_axes=True)
    fig.show()
    
    # The y-axis always shows the actual rank values from normalization
    # log_axes only affects the x-axis (original values) transformation

When to Use
-----------

RankNormalizer is particularly useful when:

- **Outliers are present**: Rank transformation limits the influence of extreme values
- **Different measurement scales**: When features have vastly different ranges
- **Non-parametric analysis**: As preprocessing for rank-based statistical tests
- **Distribution uniformity**: When you need uniform distributions across samples
- **Comparative studies**: When comparing datasets with different numbers of features (use ``normalize_by_n=True``)

Considerations
--------------

- **Information loss**: Rank transformation loses information about the magnitude of differences between values
- **Tied values**: The method for handling ties (median rank) may not be suitable for all applications
- **Discrete output**: Results are discrete ranks rather than continuous values
- **Sample independence**: Each sample is ranked independently, so cross-sample relationships may be altered

See Also
--------

- :doc:`quantile_normalizer`: For making distributions identical rather than just ranked
- :doc:`median_normalizer`: For scaling-based normalization that preserves relative differences
- :doc:`mad_normalizer`: For robust normalization that handles outliers differently
