MedianNormalizer
===============

The `MedianNormalizer` scales each sample by its median value.

.. automodule:: pronoms.normalizers.median_normalizer
    :members:
    :undoc-members:
    :show-inheritance:

Usage Example
-------------
.. code-block:: python

    import numpy as np
    from pronoms.normalizers.median_normalizer import MedianNormalizer

    # Sample data: 3 samples and 4 features
    data = np.array([
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        [20, 30, 40, 50]
    ])

    mn = MedianNormalizer()
    normalized = mn.normalize(data)
    print(normalized)
