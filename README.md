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
*   **MedianNormalizer**: Scales each sample (row) by its median, then rescales by the mean of medians to preserve overall scale.
*   **QuantileNormalizer**: Normalizes samples to have the same distribution using quantile mapping.
*   **L1Normalizer**: Scales samples to have a unit L1 norm (sum of absolute values).
*   **VSNNormalizer**: Variance Stabilizing Normalization (via R's `vsn` package). Stabilizes variance across the intensity range.
*   **SPLMNormalizer**: Stable Protein Log-Mean Normalization. Uses stably expressed proteins (low log-space CV) to derive scaling factors for normalization in log-space, then transforms back.
*   **MedianPolishNormalizer**: Tukey's Median Polish. Decomposes data (often log-transformed) into overall, row, column, and residual effects by iterative median removal.
*   **MADNormalizer**: Median Absolute Deviation Normalization. Robustly scales samples by subtracting the median and dividing by the Median Absolute Deviation (MAD).

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
- Format code: `black src tests`
- Check linting: `flake8 src tests`

## License
This project is licensed under the Apache License License - see the LICENSE file for details.
