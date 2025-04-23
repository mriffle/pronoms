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

__all__ = [
    "MedianNormalizer",
    "QuantileNormalizer",
    "L1Normalizer",
    "VSNNormalizer",
    "SPLMNormalizer",
    "MedianPolishNormalizer",
    "MADNormalizer",
]

# Lazy-load VSNNormalizer to avoid rpy2 import on package import
_LAZY_LOADABLE = {
    "VSNNormalizer": ".vsn_normalizer",
}

# Directly available
_AVAILABLE = {
    "MedianNormalizer": ".median_normalizer",
    "QuantileNormalizer": ".quantile_normalizer",
    "L1Normalizer": ".l1_normalizer",
    "SPLMNormalizer": ".splm_normalizer",
    "MedianPolishNormalizer": ".median_polish_normalizer",
    "MADNormalizer": ".mad_normalizer",
}

def __getattr__(name):
    if name in _LAZY_LOADABLE:
        module = importlib.import_module(_LAZY_LOADABLE[name], __name__)
        return getattr(module, name)
    elif name in _AVAILABLE:
        module = importlib.import_module(_AVAILABLE[name], __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    return __all__
