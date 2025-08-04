MedianPolishNormalizer
======================

The ``MedianPolishNormalizer`` implements Tukey's Median Polish algorithm to iteratively remove row and column medians from a data matrix. This method decomposes the data into overall, row, column, and residual effects, effectively removing systematic biases from both samples (rows) and features (columns).

Overview
--------

Median Polish is a robust exploratory data analysis technique that decomposes a two-way table into additive components:

**Data = Overall + Row Effect + Column Effect + Residual**

The algorithm works by iteratively:

1. **Removing row medians**: Subtracting the median of each row from all values in that row
2. **Removing column medians**: Subtracting the median of each column from all values in that column
3. **Updating overall effect**: Tracking the cumulative median adjustments
4. **Repeating until convergence**: Continuing until changes become negligible

This approach is particularly effective for:

- Removing systematic biases affecting entire samples or features
- Exploratory analysis of two-way structured data
- Preprocessing for downstream analyses that assume additive effects
- Microarray and proteomics data where both sample and feature effects are present

Key Features
------------

- **Dual bias removal**: Corrects for both row (sample) and column (feature) effects simultaneously
- **Robust method**: Uses medians instead of means, making it resistant to outliers
- **Additive decomposition**: Provides interpretable components (overall, row, column, residual)
- **Iterative convergence**: Continues until stable solution is reached
- **Log-space option**: Can automatically log-transform data for multiplicative effects

Algorithm Details
-----------------

The Median Polish algorithm iteratively removes medians until convergence:

1. **Initialize**: Start with the original data matrix
2. **Row sweep**: For each row, subtract its median from all values
3. **Column sweep**: For each column, subtract its median from all values
4. **Update overall**: Add the median of medians to the overall effect
5. **Check convergence**: Repeat steps 2-4 until changes are below threshold
6. **Return residuals**: Final result is overall + residuals

**Mathematical representation**:

After convergence: X[i,j] = Overall + Row[i] + Column[j] + Residual[i,j]

The normalized output typically returns: Overall + Residual[i,j]

Parameters
----------

.. autoclass:: pronoms.normalizers.MedianPolishNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic median polish normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import MedianPolishNormalizer
   
   # Create sample data with row and column effects
   np.random.seed(42)
   base_data = np.random.normal(100, 10, (4, 5))
   
   # Add systematic row effects (sample biases)
   row_effects = np.array([0, 20, -10, 15]).reshape(-1, 1)
   
   # Add systematic column effects (feature biases)  
   col_effects = np.array([0, 50, -20, 30, 10])
   
   # Combine effects
   data = base_data + row_effects + col_effects
   
   # Create and apply normalizer
   normalizer = MedianPolishNormalizer(log_transform=False, max_iter=10)
   normalized_data = normalizer.normalize(data)
   
   print("Original data:")
   print(data)
   print("\nNormalized data (residuals + overall):")
   print(normalized_data)
   
   # Examine the decomposition
   print(f"\nOverall effect: {normalizer.overall_:.2f}")
   print(f"Row effects: {normalizer.row_effects_}")
   print(f"Column effects: {normalizer.col_effects_}")

With log transformation:

.. code-block:: python

   # For multiplicative effects, use log transformation
   normalizer_log = MedianPolishNormalizer(log_transform=True)
   normalized_log = normalizer_log.normalize(data)
   
   print("Log-transformed normalization:")
   print(normalized_log)

Visualization:

.. code-block:: python

   # Visualize the normalization effect
   fig = normalizer.plot_comparison(data, normalized_data)
   fig.show()

When to Use
-----------

MedianPolishNormalizer is particularly useful when:

- **Two-way effects present**: Both sample (row) and feature (column) biases exist
- **Exploratory analysis**: Understanding the structure of systematic effects in data
- **Microarray data**: Classic application for gene expression data
- **Proteomics preprocessing**: When both sample preparation and protein-specific effects are present
- **Robust normalization needed**: When outliers might affect mean-based methods

Considerations
--------------

- **Additive assumption**: Assumes effects are additive (or multiplicative if log-transformed)
- **Convergence**: May require multiple iterations to reach stable solution
- **Interpretation**: Results are residuals plus overall effect, not original scale
- **Missing values**: Algorithm may not handle missing data well
- **Small datasets**: May be unstable with very small sample or feature numbers

See Also
--------

- :doc:`median_normalizer`: For simple median-based scaling without two-way decomposition
- :doc:`mad_normalizer`: For robust normalization using median absolute deviation
- :doc:`quantile_normalizer`: For making distributions identical across samples
- :doc:`vsn_normalizer`: For variance-stabilizing normalization


