"""Tests for the lazy-loading mechanism in ``pronoms.normalizers``.

The package uses ``__getattr__`` to defer loading heavy R / directlfq modules
until first access. This module verifies that:

- every name in ``__all__`` resolves to a concrete class via ``__getattr__``;
- unknown names raise ``AttributeError``;
- ``dir(pronoms.normalizers)`` returns the public ``__all__``.
"""

from __future__ import annotations

import importlib

import pytest

import pronoms.normalizers as norm_pkg

PUBLIC_NAMES = [
    "MedianNormalizer",
    "QuantileNormalizer",
    "L1Normalizer",
    "MADNormalizer",
    "MedianPolishNormalizer",
    "SPLMNormalizer",
    "VSNNormalizer",
    "DirectLFQNormalizer",
    "RankNormalizer",
]


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_lazy_getattr_returns_a_class(name):
    cls = getattr(norm_pkg, name)
    assert isinstance(cls, type)
    assert cls.__name__ == name


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError, match=r"has no attribute NopeNormalizer"):
        norm_pkg.NopeNormalizer  # noqa: B018


def test_dir_returns_public_all():
    assert sorted(dir(norm_pkg)) == sorted(PUBLIC_NAMES)


def test_public_all_matches_known_list():
    assert sorted(norm_pkg.__all__) == sorted(PUBLIC_NAMES)


def test_module_can_be_reimported_cleanly():
    """A fresh import should not error and should still expose every public
    class through the lazy-load path."""
    mod = importlib.reload(norm_pkg)
    for name in PUBLIC_NAMES:
        assert isinstance(getattr(mod, name), type)
