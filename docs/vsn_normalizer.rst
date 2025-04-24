VSNNormalizer
=============

The `VSNNormalizer` performs Variance Stabilizing Normalization (VSN) using the R `vsn` package from Bioconductor. This method applies a transformation to the data that stabilizes the variance across the full range of intensities, making downstream statistical analyses more reliable and robust to intensity-dependent noise.

VSN normalization is particularly valuable for high-throughput proteomics or microarray data, where variance often increases with signal intensity. Use it when you need to remove intensity-dependent variance and improve comparability across samples. Requires R and the `vsn` package installed.

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


Installation Requirements
-------------------------

The ``VSNNormalizer`` requires a working R installation and the ``vsn`` package from Bioconductor.

Installing R
~~~~~~~~~~~~

Follow the instructions for your operating system:

*   **Windows:** Download the installer from the `Comprehensive R Archive Network (CRAN) <https://cran.r-project.org/bin/windows/base/>`_ and follow the installation wizard.
*   **macOS:** Download the appropriate ``.pkg`` file for your system from `CRAN <https://cran.r-project.org/bin/macosx/>`_ and run the installer.
*   **Linux:** Use your distribution's package manager. For example:
    *   Debian/Ubuntu: ``sudo apt update && sudo apt install r-base r-base-dev``
    *   Fedora/CentOS/RHEL: ``sudo dnf install R``
    *   Consult the `CRAN instructions for Linux <https://cran.r-project.org/bin/linux/>`_ for specific distributions and potential repository configurations.

Installing the 'vsn' R package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once R is installed, open an R console or terminal and run the following commands to install the ``vsn`` package using the Bioconductor package manager:

.. code-block:: r

    if (!requireNamespace("BiocManager", quietly = TRUE))
        install.packages("BiocManager")

    BiocManager::install("vsn")

This will install ``vsn`` and any necessary dependencies from Bioconductor.


Citation
--------
Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M. Variance stabilization applied to microarray data calibration and to the quantification of differential expression. *Bioinformatics*. 2002;18 Suppl 1:S96–104. `doi:10.1093/bioinformatics/18.suppl_1.s96 <https://doi.org/10.1093/bioinformatics/18.suppl_1.s96>`__, `PMID: 12169536 <https://pubmed.ncbi.nlm.nih.gov/12169536/>`__

Class Documentation
-------------------

.. automodule:: pronoms.normalizers.vsn_normalizer
    :members:
    :undoc-members:
    :show-inheritance:
