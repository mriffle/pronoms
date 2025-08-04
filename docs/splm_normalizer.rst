SPLMNormalizer
==============

The ``SPLMNormalizer`` implements Stable Protein Log-Mean Normalization (SPLM), which identifies a subset of stably expressed proteins based on their low variability in log-space and uses them as internal standards for normalization. This method is particularly effective when a subset of proteins can be assumed to remain constant across experimental conditions.

Overview
--------

SPLM normalization addresses the challenge of selecting appropriate reference features for normalization in proteomics data. Rather than assuming all proteins are equally suitable as references, SPLM:

1. **Identifies stable proteins**: Selects features with the lowest coefficient of variation in log-space
2. **Uses stable proteins as references**: Calculates scaling factors based only on these stable features
3. **Normalizes all features**: Applies the scaling factors derived from stable proteins to the entire dataset

This approach is particularly powerful when:

- A subset of proteins are expected to be housekeeping or constitutively expressed
- Technical variation affects all proteins proportionally
- You want to avoid bias from highly variable proteins in normalization
- Working with targeted proteomics where reference proteins can be identified

Key Features
------------

- **Automatic stable protein selection**: Identifies the most stable features based on log-space variability
- **Reference-based normalization**: Uses only stable proteins for scaling factor calculation
- **Log-space processing**: Handles multiplicative effects through log transformation
- **Robust to variable proteins**: Normalization is not affected by highly variable features
- **Preserves biological variation**: Maintains true biological differences while removing technical bias

Algorithm Details
-----------------

The SPLM algorithm works through the following steps:

1. **Log transformation**: X_log = log(X + ε) where ε prevents log(0)
2. **Calculate log-CV**: For each protein j, CV_j = std(X_log[:, j]) / mean(X_log[:, j])
3. **Select stable proteins**: Choose the `num_stable_proteins` with lowest CV
4. **Calculate scaling factors**: For each sample i, factor_i = mean(X_log[i, stable_proteins])
5. **Calculate grand mean**: grand_mean = mean(all scaling factors)
6. **Normalize in log-space**: X_norm_log[i, j] = X_log[i, j] - factor_i + grand_mean
7. **Back-transform**: X_normalized = exp(X_norm_log) - ε

**Mathematical representation**:

.. math::

   \text{CV}_j = \frac{\sigma(\log(X_{:,j} + \epsilon))}{\mu(\log(X_{:,j} + \epsilon))}

.. math::

   \text{factor}_i = \frac{1}{k} \sum_{j \in \text{stable}} \log(X_{i,j} + \epsilon)

where k is the number of stable proteins.

Parameters
----------

.. autoclass:: pronoms.normalizers.SPLMNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic SPLM normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import SPLMNormalizer
   
   # Create sample data with stable and variable proteins
   np.random.seed(42)
   
   # Stable proteins (low variability)
   stable_proteins = np.array([
       [100, 200, 150],  # Sample 1
       [105, 210, 155],  # Sample 2
       [95, 190, 145]    # Sample 3
   ])
   
   # Variable proteins (high variability)
   variable_proteins = np.array([
       [50, 1000],   # Sample 1
       [150, 500],   # Sample 2
       [25, 2000]    # Sample 3
   ])
   
   # Combine stable and variable proteins
   data = np.hstack([stable_proteins, variable_proteins])
   
   # Create and apply normalizer
   # Use 3 stable proteins (should select the first 3 columns)
   normalizer = SPLMNormalizer(num_stable_proteins=3, epsilon=1.0)
   normalized_data = normalizer.normalize(data)
   
   print("Original data:")
   print(data)
   print("\nNormalized data:")
   print(normalized_data)
   
   # Examine which proteins were selected as stable
   print(f"\nStable protein indices: {normalizer.stable_feature_indices_}")
   print(f"Log-CVs of all proteins: {normalizer.log_cvs_}")
   print(f"Scaling factors: {normalizer.log_scaling_factors_}")

Visualization:

.. code-block:: python

   # Visualize the normalization effect
   fig = normalizer.plot_comparison(data, normalized_data)
   fig.show()

When to Use
-----------

SPLMNormalizer is particularly useful when:

- **Housekeeping proteins present**: Dataset contains proteins expected to be stably expressed
- **Targeted proteomics**: Working with a curated set of proteins where some serve as references
- **Technical variation dominant**: When most variation is technical rather than biological
- **Reference protein selection**: When you want data-driven selection of reference features
- **Proportional scaling needed**: When technical effects scale all proteins proportionally

Considerations
--------------

- **Stable protein assumption**: Requires that some proteins are truly stable across conditions
- **Number of stable proteins**: Choice of `num_stable_proteins` can significantly affect results
- **Log-space processing**: Assumes multiplicative rather than additive effects
- **Minimum protein requirement**: Needs sufficient proteins to reliably identify stable ones
- **Biological interpretation**: May remove true biological signal if stable proteins are misidentified

See Also
--------

- :doc:`median_normalizer`: For simple scaling-based normalization
- :doc:`quantile_normalizer`: For making distributions identical across samples
- :doc:`mad_normalizer`: For robust normalization using median absolute deviation
- :doc:`vsn_normalizer`: For variance-stabilizing normalization


