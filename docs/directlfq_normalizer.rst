DirectLFQ Normalizer
=====================

.. automodule:: pronoms.normalizers.directlfq_normalizer

.. autoclass:: pronoms.normalizers.directlfq_normalizer.DirectLFQNormalizer
   :members:
   :undoc-members:
   :show-inheritance:

Usage Example
+++++++++++++

.. code-block:: python

    import numpy as np
    from pronoms.normalizers import DirectLFQNormalizer

    # Example peptide-level data (samples x features)
    # Typically, you'd load this from a file
    X_peptides = np.array([
        [100, 110, 50, 60],
        [120, 130, 55, 65],
        [ 90, 100, 45, 55]
    ])

    # Corresponding protein and peptide IDs (one per feature)
    protein_ids_input = ['ProtA', 'ProtA', 'ProtB', 'ProtB']
    peptide_ids_input = ['Peptide1', 'Peptide2', 'Peptide3', 'Peptide4']

    # Initialize the normalizer
    normalizer = DirectLFQNormalizer(num_cores=2)

    # Perform normalization
    norm_prot_matrix, norm_ion_matrix, protein_ids_output, peptide_ids_output = normalizer.normalize(
        X_peptides,
        proteins=protein_ids_input,
        peptides=peptide_ids_input
    )

    # norm_prot_matrix contains normalized protein intensities (samples x proteins)
    # norm_ion_matrix contains normalized peptide/ion intensities (samples x peptides)
    # protein_ids_output contains the unique protein IDs for norm_prot_matrix columns
    # peptide_ids_output contains the unique peptide IDs for norm_ion_matrix columns

    print("Normalized Protein Matrix Shape:", norm_prot_matrix.shape)
    print("Output Protein IDs:", protein_ids_output)
    print("Normalized Ion Matrix Shape:", norm_ion_matrix.shape)
    print("Output Peptide IDs:", peptide_ids_output)
