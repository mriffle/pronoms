# Pronoms: Proteomics Normalization Python Library

## Overview
Pronoms is a Python library implementing multiple normalization methods for quantitative proteomics data. Each normalization method is encapsulated within modular, reusable classes. The library includes visualization capabilities that allow users to easily observe the effects of normalization. Some normalization methods, such as VSN normalization, leverage R on the backend for computation.

## Installation

### Prerequisites
- Python 3.9 or higher
- For R-based normalizers (VSN):
  - R installed on your system
  - Required R packages: `vsn`

### Installing from PyPI
```bash
pip install pronoms
```

### Installing for Development
```bash
# Clone the repository
git clone https://github.com/yourusername/pronoms.git
cd pronoms

# Install in development mode with dev dependencies
pip install -e .[dev]
```

## Usage

### Basic Example
```python
import numpy as np
from pronoms.normalizers import MedianNormalizer

# Create sample data
data = np.random.rand(5, 100)  # 5 samples, 100 proteins/features

# Create normalizer and apply normalization
normalizer = MedianNormalizer()
normalized_data = normalizer.normalize(data)

# Visualize the effect of normalization
normalizer.plot_comparison(data, normalized_data)
```

### Available Normalizers
*   **DirectLFQNormalizer**: Performs protein quantification directly from peptide/ion intensity data using the DirectLFQ algorithm. **Ammar C, Schessner JP, Willems S, Michaelis AC, Mann M.** Accurate Label-Free Quantification by directLFQ to Compare Unlimited Numbers of Proteomes. *Mol Cell Proteomics*. 2023 Jul;22(7):100581. [doi:10.1016/j.mcpro.2023.100581](https://doi.org/10.1016/j.mcpro.2023.100581). [PMID: 37225017](https://pubmed.ncbi.nlm.nih.gov/37225017/)
*   **L1Normalizer**: Scales samples to have a unit L1 norm (sum of absolute values).
*   **MADNormalizer**: Median Absolute Deviation Normalization. Robustly scales samples by subtracting the median and dividing by the Median Absolute Deviation (MAD).
*   **MedianNormalizer**: Scales each sample (row) by its median, then rescales by the mean of medians to preserve overall scale.
*   **MedianPolishNormalizer**: Tukey's Median Polish. Decomposes data (often log-transformed) into overall, row, column, and residual effects by iterative median removal.
*   **QuantileNormalizer**: Normalizes samples to have the same distribution using quantile mapping.
*   **SPLMNormalizer**: Stable Protein Log-Mean Normalization. Uses stably expressed proteins (low log-space CV) to derive scaling factors for normalization in log-space, then transforms back.
*   **VSNNormalizer**: Variance Stabilizing Normalization (via R's `vsn` package). Stabilizes variance across the intensity range. **Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M.** Variance stabilization applied to microarray data calibration and to the quantification of differential expression. *Bioinformatics*. 2002;18 Suppl 1:S96–104. [doi:10.1093/bioinformatics/18.suppl_1.s96](https://doi.org/10.1093/bioinformatics/18.suppl_1.s96). [PMID: 12169536](https://pubmed.ncbi.nlm.nih.gov/12169536/)

### Data Format
All normalizers expect data in the format of a 2D numpy array or pandas DataFrame with shape `(n_samples, n_features)` where:
- Each **row** represents a sample
- Each **column** represents a protein/feature

This follows the standard convention used in scikit-learn and other Python data science libraries.

## R Integration
For normalizers that use R (VSN), ensure R is properly installed and accessible. The library uses `rpy2` to interface with R.

### Installing Required R Packages
The VSN package is part of Bioconductor. In R, run the following commands:

```R
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("vsn")
```

## Development
- Run tests: `pytest`

## License
This project is licensed under the Apache License License - see the LICENSE file for details.
