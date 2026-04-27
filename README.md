<p align="center">
  <img src="docs/_static/logo.png" alt="Pronoms Logo" width="180"/>
</p>

# Pronoms: Proteomics Normalization Python Library

## Overview
Pronoms is a Python library implementing multiple normalization methods for quantitative proteomics data. Each normalization method is encapsulated within modular, reusable classes. The library includes visualization capabilities that allow users to easily observe the effects of normalization. All normalization methods are implemented in pure Python (NumPy/SciPy/pandas); pronoms has no R or rpy2 dependency.

## Documentation
See https://pronoms.readthedocs.io/ for complete documentation.

## Installation

You can install Pronoms directly from PyPI using pip:

```bash
pip install pronoms
```

### Prerequisites
- Python 3.10 or higher
- No R / rpy2 install required.

### Installing for Development
```bash
# Clone the repository
git clone https://github.com/mriffle/pronoms.git
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
*   **MADNormalizer**: Median Absolute Deviation Normalization. Robustly scales samples by subtracting the median and dividing by the Median Absolute Deviation (MAD). Pass ``scale_to_sigma=True`` to multiply MAD by 1.4826 so the output is a robust z-score (matches R's ``mad()``).
*   **MedianNormalizer**: Scales each sample (row) by its median, then rescales by the mean of medians to preserve overall scale.
*   **MedianPolishNormalizer**: Tukey's Median Polish. Decomposes data (often log-transformed) into overall, row, column, and residual effects by iterative median removal.
*   **QuantileNormalizer**: Normalizes samples to have the same distribution using quantile mapping.
*   **RankNormalizer**: Transforms each sample's values to their ranks (1 to N), with tied values receiving the median rank. Optionally normalizes ranks by dividing by N for cross-dataset comparability.
*   **SPLMNormalizer**: Stable Protein Log-Mean Normalization. Uses stably expressed proteins (lowest linear-space CV, ``std/mean``) to derive scaling factors for normalization in log-space, then transforms back.
*   **VSNNormalizer**: Variance Stabilizing Normalization. Native NumPy/SciPy implementation of Huber et al.'s arcsinh-based variance-stabilizing transform with LTS-robust parameter estimation; matches Bioconductor's `vsn` package output to ~1e-6 on realistic proteomics data. **Huber W, von Heydebreck A, Sültmann H, Poustka A, Vingron M.** Variance stabilization applied to microarray data calibration and to the quantification of differential expression. *Bioinformatics*. 2002;18 Suppl 1:S96–104. [doi:10.1093/bioinformatics/18.suppl_1.s96](https://doi.org/10.1093/bioinformatics/18.suppl_1.s96). [PMID: 12169536](https://pubmed.ncbi.nlm.nih.gov/12169536/)

### Data Format
All normalizers expect data in the format of a 2D numpy array or pandas DataFrame with shape `(n_samples, n_features)` where:
- Each **row** represents a sample
- Each **column** represents a protein/feature

This follows the standard convention used in scikit-learn and other Python data science libraries.

## Development

Set up a virtual environment and install the dev extras:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The pre-flight gate before any commit is:

```bash
pytest                          # full test suite (warnings -> errors)
ruff check src tests            # lint
ruff format --check src tests   # formatting (use `ruff format` to apply)
mypy                            # static type check
```

Coverage:

```bash
pytest --cov=src/pronoms --cov-report=term-missing
```

## License
This project is licensed under the Apache License License - see the LICENSE file for details.
