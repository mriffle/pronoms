VSNNormalizer
=============

The `VSNNormalizer` implements Variance Stabilizing Normalization using the R vsn package (Bioconductor).
Requires R and the `vsn` package installed.

.. automodule:: pronoms.normalizers.vsn_normalizer
    :members:
    :undoc-members:
    :show-inheritance:

Usage Example
-------------
.. code-block:: python

    import numpy as np
    from pronoms.normalizers.vsn_normalizer import VSNNormalizer

    # Sample data: 3 samples and 4 features
    data = np.array([
        [10, 20, 30, 40],
        [15, 25, 35, 45],
        [20, 30, 40, 50]
    ])

    vsn = VSNNormalizer()
    vsn_data = vsn.normalize(data)
    print(vsn_data)
