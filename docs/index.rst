Welcome to Pronoms documentation!
=================================

Pronoms is a Python library providing a suite of normalization methods for quantitative proteomics data. It is designed to help researchers and bioinformaticians preprocess and normalize large-scale proteomics datasets, enabling fair comparison across samples and reducing technical variation. Pronoms includes both classic and modern normalization techniques, some of which leverage R for advanced statistical methods. The library is modular, extensible, and comes with visualization tools to help users assess the effects of normalization.

Use Pronoms if you need robust, reproducible normalization of proteomics data for downstream statistical analysis, biomarker discovery, or comparative proteomics studies.

Installation
------------

You can install Pronoms directly from PyPI using pip:

.. code-block:: bash

   pip install pronoms

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   directlfq_normalizer
   l1_normalizer
   mad_normalizer
   median_normalizer
   median_polish_normalizer
   quantile_normalizer
   rank_normalizer
   splm_normalizer
   vsn_normalizer

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
