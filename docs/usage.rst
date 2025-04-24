Usage
=====

Installation
------------

You can install Pronoms directly from PyPI using pip:

.. code-block:: bash

   pip install pronoms

Basic Usage
-----------

.. code-block:: python

    import numpy as np
    from pronoms.normalizers import (
        MedianNormalizer,
        QuantileNormalizer,
        L1Normalizer,
        VSNNormalizer
    )

    # Example data: 5 samples (rows) and 3 proteins (columns)
    data = np.array([
        [10, 20, 30],
        [15, 25, 35],
        [20, 30, 40],
        [25, 35, 45],
        [30, 40, 50]
    ])

    # Median normalization
    med_norm = MedianNormalizer()
    med_data = med_norm.normalize(data)

    # Quantile normalization
    qnorm = QuantileNormalizer()
    q_data = qnorm.normalize(data)

    # L1 normalization
    l1norm = L1Normalizer()
    l1_data = l1norm.normalize(data)

    # VSN normalization (requires R and Bioconductor 'vsn' package)
    vsn_norm = VSNNormalizer()
    vsn_data = vsn_norm.normalize(data)


See each normalizer's documentation for detailed API and parameters.
