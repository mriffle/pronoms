"""Shared pytest configuration for the pronoms test suite.

- Forces matplotlib to the headless ``Agg`` backend so plotting tests never
  attempt to open an interactive window in CI or in a venv without a display.
- Closes every matplotlib figure after each test to prevent figure-handle leaks
  when the suite grows.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Ensure no matplotlib figure outlives the test that created it."""
    yield
    plt.close("all")
