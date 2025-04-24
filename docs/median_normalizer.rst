MedianNormalizer
================

The `MedianNormalizer` scales each sample by its median value, then rescales all samples by the mean of medians to preserve overall scale. This simple normalization method corrects for systematic differences in sample loading or labeling, assuming most proteins are not changing between samples.

Use MedianNormalizer when you want to adjust for global intensity differences across samples in proteomics or other high-throughput data.

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
