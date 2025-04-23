"""
Normalizers module for Pronoms.

This module contains various normalization methods for proteomics data.
"""

import importlib
from .median_normalizer import MedianNormalizer
from .quantile_normalizer import QuantileNormalizer
from .l1_normalizer import L1Normalizer
from .median_polish_normalizer import MedianPolishNormalizer
from .mad_normalizer import MADNormalizer
from .splm_normalizer import SPLMNormalizer
from .vsn_normalizer import VSNNormalizer
from .directlfq_normalizer import DirectLFQNormalizer

__all__ = [
    "MedianNormalizer",
    "QuantileNormalizer",
    "L1Normalizer",
    "MADNormalizer",
    "MedianPolishNormalizer",
    "SPLMNormalizer",
    "VSNNormalizer",
    "DirectLFQNormalizer"
]

# Lazy-load VSNNormalizer to avoid rpy2 import on package import
_lazy_imports = {
    "VSNNormalizer": ".vsn_normalizer",
    "MedianNormalizer": ".median_normalizer",
    "QuantileNormalizer": ".quantile_normalizer",
    "L1Normalizer": ".l1_normalizer",
    "SPLMNormalizer": ".splm_normalizer",
    "MedianPolishNormalizer": ".median_polish_normalizer",
    "MADNormalizer": ".mad_normalizer",
    "DirectLFQNormalizer": ".directlfq_normalizer",
}

def __getattr__(name):
    if name in _lazy_imports:
        module = importlib.import_module(_lazy_imports[name], __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    return __all__
