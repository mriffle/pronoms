# Pronoms: Proteomics Normalization Python Library

## Overview
Pronoms is a Python library implementing multiple normalization methods for quantitative proteomics data. Each normalization method is encapsulated within modular, reusable classes. The library includes visualization capabilities that allow users to easily observe the effects of normalization. Some normalization methods, such as DirectLFQ and VSN normalization, leverage R on the backend for computation.

## Installation

### Prerequisites
- Python 3.9 or higher
- For R-based normalizers (DirectLFQ, VSN):
  - R installed on your system
  - Required R packages: `vsn`, `DirectLFQ`

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
data = np.random.rand(100, 5)  # 100 proteins, 5 samples

# Create normalizer and apply normalization
normalizer = MedianNormalizer()
normalized_data = normalizer.normalize(data)

# Visualize the effect of normalization
normalizer.plot_comparison(data, normalized_data)
```

### Available Normalizers
- `MedianNormalizer`: Normalizes data by scaling each sample by its median
- `QuantileNormalizer`: Applies quantile normalization across samples
- `L1Normalizer`: Normalizes data by scaling each sample to have an L1 norm of 1
- `DirectLFQNormalizer`: Implements DirectLFQ normalization (requires R)
- `VSNNormalizer`: Implements Variance Stabilizing Normalization (requires R)

## R Integration
For normalizers that use R (DirectLFQ, VSN), ensure R is properly installed and accessible. The library uses `rpy2` to interface with R.

### Installing Required R Packages
In R:
```R
install.packages("vsn")
# For DirectLFQ, which may be on Bioconductor:
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("DirectLFQ")
```

## Development
- Run tests: `pytest`
- Format code: `black src tests`
- Check linting: `flake8 src tests`

## License
This project is licensed under the MIT License - see the LICENSE file for details.
