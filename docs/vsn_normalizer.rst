VSNNormalizer
=============

The ``VSNNormalizer`` performs Variance Stabilizing Normalization (VSN) using the R ``vsn`` package from Bioconductor. This method applies a data-driven transformation that stabilizes the variance across the full range of intensities, making downstream statistical analyses more reliable and robust to intensity-dependent noise.

Overview
--------

Variance Stabilizing Normalization addresses a common problem in high-throughput biological data: the variance of measurements often depends on their intensity level. In proteomics and microarray data, low-intensity measurements typically have lower variance while high-intensity measurements have higher variance. This heteroscedasticity can bias statistical analyses.

VSN works by:

1. **Learning transformation parameters**: Fitting an arcsinh transformation with data-driven parameters
2. **Applying transformation**: Converting data to a scale where variance is approximately constant
3. **Calibrating across samples**: Ensuring comparable scales between samples

This approach is particularly effective for:

- Proteomics data with intensity-dependent variance
- Microarray expression data
- Any high-throughput data where variance scales with intensity
- Preprocessing for statistical methods that assume homoscedasticity

Key Features
------------

- **Variance stabilization**: Makes variance approximately constant across intensity ranges
- **Data-driven**: Parameters are learned from the data, not predetermined
- **Cross-sample calibration**: Ensures samples are on comparable scales
- **Robust transformation**: Uses arcsinh-based transformation that handles zero and negative values
- **R integration**: Leverages the mature ``vsn`` Bioconductor package

Algorithm Details
-----------------

VSN applies an arcsinh-based transformation with the form:

.. math::

   h(x) = \text{arcsinh}(a + b \cdot x)

where parameters ``a`` and ``b`` are estimated from the data to achieve variance stabilization.

The algorithm:

1. **Parameter estimation**: Uses maximum likelihood to estimate transformation parameters
2. **Transformation**: Applies the learned transformation to stabilize variance
3. **Calibration**: Adjusts samples to have comparable scales

The ``vsn`` package uses robust methods to handle outliers and missing values during parameter estimation.

Parameters
----------

.. autoclass:: pronoms.normalizers.VSNNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic VSN normalization:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import VSNNormalizer
   
   # Create sample data with intensity-dependent variance
   np.random.seed(42)
   data = np.array([
       np.random.normal([100, 1000, 10000], [10, 100, 1000]),  # Sample 1
       np.random.normal([120, 1200, 12000], [12, 120, 1200]),  # Sample 2
       np.random.normal([80, 800, 8000], [8, 80, 800])         # Sample 3
   ])
   
   # Create and apply normalizer
   normalizer = VSNNormalizer()
   normalized_data = normalizer.normalize(data)
   
   print("Original data:")
   print(data)
   print("\nVSN normalized data:")
   print(normalized_data)
   
   # Access fitted parameters
   print("\nVSN parameters:")
   print(normalizer.vsn_params)

Visualization:

.. code-block:: python

   # Visualize the normalization effect
   fig = normalizer.plot_comparison(data, normalized_data)
   fig.show()


When to Use
-----------

VSNNormalizer is particularly useful when:

- **Intensity-dependent variance**: Data shows increasing variance with higher intensities
- **Proteomics applications**: Mass spectrometry data with heteroscedastic noise
- **Microarray data**: Gene expression data with intensity-dependent variance
- **Statistical analysis preparation**: Before applying methods that assume constant variance
- **Cross-sample comparison**: When samples need to be on truly comparable scales

Considerations
--------------

- **R dependency**: Requires R installation and the ``vsn`` Bioconductor package
- **Computational cost**: More expensive than simple scaling methods due to parameter estimation
- **Data requirements**: Works best with sufficient data points for robust parameter estimation
- **Transformation interpretation**: Results are on a transformed scale, not original units
- **Missing values**: Handles missing values but may affect parameter estimation quality

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


See Also
--------

- :doc:`median_normalizer`: For simple scaling-based normalization
- :doc:`quantile_normalizer`: For making distributions identical across samples
- :doc:`mad_normalizer`: For robust normalization using median absolute deviation
- :doc:`rank_normalizer`: For rank-based transformation

Citation
--------
Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M. Variance stabilization applied to microarray data calibration and to the quantification of differential expression. *Bioinformatics*. 2002;18 Suppl 1:S96–104. `doi:10.1093/bioinformatics/18.suppl_1.s96 <https://doi.org/10.1093/bioinformatics/18.suppl_1.s96>`__, `PMID: 12169536 <https://pubmed.ncbi.nlm.nih.gov/12169536/>`__
