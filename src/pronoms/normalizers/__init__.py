"""
Normalizers module for Pronoms.

This module contains various normalization methods for proteomics data.
"""

from .median_normalizer import MedianNormalizer
from .quantile_normalizer import QuantileNormalizer
from .l1_normalizer import L1Normalizer

__all__ = [
    "MedianNormalizer",
    "QuantileNormalizer",
    "L1Normalizer",
    "VSNNormalizer",
    "SPLMNormalizer",
]

# Lazy-load VSNNormalizer to avoid rpy2 import on package import
def __getattr__(name: str):
    if name == "VSNNormalizer":
        from .vsn_normalizer import VSNNormalizer
        return VSNNormalizer
    elif name == "SPLMNormalizer":
        from .splm_normalizer import SPLMNormalizer
        return SPLMNormalizer
    raise AttributeError(f"module {__name__} has no attribute {name}")

def __dir__():
    return __all__
