DirectLFQNormalizer
===================

The ``DirectLFQNormalizer`` implements the DirectLFQ algorithm for protein quantification directly from peptide or ion-level intensity data. This method directly infers protein abundances by modeling the relationship between peptides and their parent proteins, enabling accurate label-free quantification across many samples without the biases of traditional summary-based approaches.

Overview
--------

DirectLFQ addresses fundamental limitations in traditional label-free quantification approaches that typically summarize peptide intensities (e.g., by taking the top 3 peptides or using all peptides). Instead, DirectLFQ:

1. **Models peptide-protein relationships**: Directly accounts for the contribution of each peptide to its parent protein
2. **Handles missing values**: Uses all available peptide information without requiring complete data
3. **Scales to large datasets**: Efficiently processes hundreds or thousands of samples
4. **Provides dual output**: Returns both protein-level and peptide-level quantification

This approach is particularly powerful for:

- Large-scale proteomics studies with many samples
- Datasets with significant missing values
- Comparative studies requiring accurate protein quantification
- Clinical proteomics where precision is critical

Key Features
------------

- **Direct quantification**: Bypasses traditional peptide summarization steps
- **Missing value robust**: Utilizes all available peptide evidence
- **Dual-level output**: Provides both protein and peptide quantification
- **Scalable**: Handles large sample numbers efficiently
- **Normalization integrated**: Combines quantification with normalization in one step


Algorithm Details
-----------------

DirectLFQ uses a sophisticated statistical model to infer protein abundances from peptide intensities. The algorithm:

1. **Constructs design matrix**: Maps peptides to their parent proteins
2. **Applies statistical model**: Uses robust regression to estimate protein abundances
3. **Handles missing values**: Incorporates all available evidence without imputation
4. **Normalizes across samples**: Ensures comparable scales between samples
5. **Returns dual quantification**: Provides both protein and peptide-level results

The method avoids common pitfalls of peptide summarization approaches by directly modeling the underlying biological relationships.

Parameters
----------

.. autoclass:: pronoms.normalizers.DirectLFQNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
-------------

Basic DirectLFQ quantification:

.. code-block:: python

   import numpy as np
   from pronoms.normalizers import DirectLFQNormalizer
   
   # Example peptide-level data (samples x peptides)
   # In practice, load from MaxQuant or similar output
   peptide_data = np.array([
       [1000, 1100, 500, 600, 0],     # Sample 1
       [1200, 1300, 550, 650, 200],   # Sample 2  
       [900, 1000, 450, 550, 0]       # Sample 3
   ])
   
   # Protein and peptide identifiers
   protein_ids = ['ProtA', 'ProtA', 'ProtB', 'ProtB', 'ProtC']
   peptide_ids = ['Pep1', 'Pep2', 'Pep3', 'Pep4', 'Pep5']
   
   # Create and apply normalizer
   normalizer = DirectLFQNormalizer(num_cores=2)
   
   protein_matrix, peptide_matrix, protein_names, peptide_names = normalizer.normalize(
       peptide_data,
       proteins=protein_ids,
       peptides=peptide_ids
   )
   
   print("Protein quantification:")
   print(f"Shape: {protein_matrix.shape}")
   print(f"Proteins: {protein_names}")
   print(protein_matrix)
   
   print("\nPeptide quantification:")
   print(f"Shape: {peptide_matrix.shape}")
   print(f"Peptides: {peptide_names}")
   print(peptide_matrix)

Visualization:

.. code-block:: python

   # Visualize protein-level normalization
   fig = normalizer.plot_comparison(peptide_data, protein_matrix)
   fig.show()

When to Use
-----------

DirectLFQNormalizer is particularly useful when:

- **Large-scale studies**: Processing hundreds or thousands of samples
- **Missing value issues**: Datasets with substantial missing peptide measurements
- **Accurate quantification needed**: Clinical or biomarker studies requiring precision
- **Peptide-level data available**: Starting from MaxQuant, Proteome Discoverer, or similar outputs
- **Comparative proteomics**: Studies comparing protein abundances across conditions

Considerations
--------------

- **Computational requirements**: More intensive than simple summarization methods
- **Python dependency**: Requires the ``directlfq`` Python package
- **Data format**: Needs peptide-to-protein mapping information
- **Memory usage**: Large datasets may require substantial memory
- **Parameter tuning**: May benefit from adjusting algorithm parameters for specific datasets


See Also
--------

- :doc:`median_normalizer`: For simple scaling-based normalization at the protein level
- :doc:`quantile_normalizer`: For making distributions identical across samples
- :doc:`vsn_normalizer`: For variance-stabilizing normalization
- :doc:`rank_normalizer`: For rank-based transformation

Citation
--------
Ammar C, Schessner JP, Willems S, Michaelis AC, Mann M. Accurate Label-Free Quantification by directLFQ to Compare Unlimited Numbers of Proteomes. *Mol Cell Proteomics*. 2023 Jul;22(7):100581. `doi:10.1016/j.mcpro.2023.100581 <https://doi.org/10.1016/j.mcpro.2023.100581>`__, `PMID: 37225017 <https://pubmed.ncbi.nlm.nih.gov/37225017/>`__
   