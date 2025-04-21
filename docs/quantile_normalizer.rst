QuantileNormalizer
==================

The `QuantileNormalizer` applies quantile normalization across samples.

.. automodule:: pronoms.normalizers.quantile_normalizer
    :members:
    :undoc-members:
    :show-inheritance:

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
