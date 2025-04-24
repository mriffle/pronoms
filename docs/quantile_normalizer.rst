QuantileNormalizer
==================

The `QuantileNormalizer` applies quantile normalization to make the empirical distributions of all samples identical. This is achieved by sorting each sample, averaging across samples at each rank, and assigning these average values back to the original data.

Use quantile normalization when you want to directly compare samples by ensuring they have the same distribution, such as in large-scale studies or when integrating data from multiple batches.

Usage Example
-------------
.. code-block:: python

    import numpy as np
    from pronoms.normalizers.quantile_normalizer import QuantileNormalizer

    # Sample data: 3 samples and 4 features
    data = np.array([
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        [20, 30, 40, 50]
    ])

    qn = QuantileNormalizer()
    normalized = qn.normalize(data)
    print(normalized)

Class Documentation
-------------------

.. automodule:: pronoms.normalizers.quantile_normalizer
    :members:
    :undoc-members:
    :show-inheritance:
