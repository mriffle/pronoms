L1Normalizer
============

The `L1Normalizer` scales each sample by its L1 norm (sum of absolute values), so each sample has an L1 norm of 1.

.. automodule:: pronoms.normalizers.l1_normalizer
    :members:
    :undoc-members:
    :show-inheritance:

Usage Example
-------------
.. code-block:: python

    import numpy as np
    from pronoms.normalizers.l1_normalizer import L1Normalizer

    # Sample data: 3 samples and 4 features
    data = np.array([
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        [20, 30, 40, 50]
    ])

    l1 = L1Normalizer()
    normalized = l1.normalize(data)
    print(normalized)
