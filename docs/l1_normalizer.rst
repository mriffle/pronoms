L1Normalizer
============

The `L1Normalizer` adjusts each sample so that its sum of absolute values (L1 norm) is 1. This scaling method removes differences in total signal between samples, making them directly comparable regardless of overall intensity.

Use L1 normalization when you want to correct for differences in sample loading, instrument response, or total protein amount, ensuring that each sample contributes equally to downstream analyses.

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
