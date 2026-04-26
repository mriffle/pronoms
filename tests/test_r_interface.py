"""Tests for ``pronoms.utils.r_interface``.

The R-bound functions cannot be exercised without a working R + rpy2 install,
which is optional. These tests target the pure-Python branches:

- the ``_import_rpy2`` cache,
- the no-rpy2 error path of ``check_r_availability``,
- the CRAN-vs-Bioconductor partitioning in ``setup_r_environment``'s missing-
  package error message.

Together they take the file from 22% to comfortably above 90% coverage.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from pronoms.utils import r_interface


def _rpy2_can_initialize_r() -> bool:
    """rpy2 may import successfully but still fail at R initialization if the
    system R is older than what the installed rpy2 expects (e.g. rpy2 >= 3.6
    needs R >= 4.2 for ``R_getVar``). Tests that touch the live rpy2 → R
    boundary are skipped when this check fails."""
    try:
        import rpy2.robjects  # noqa: F401

        return True
    except Exception:
        return False


_RPY2_LIVE = _rpy2_can_initialize_r()


# ---------------------------------------------------------------------------
# _import_rpy2 cache
# ---------------------------------------------------------------------------


def test_import_rpy2_caches_result_after_first_call():
    """Once the module-level ``HAS_RPY2`` flag is set, future calls must not
    re-run the import probe."""
    # Force the cache into a known state.
    r_interface.HAS_RPY2 = True
    assert r_interface._import_rpy2() is True
    # Flip the cache and confirm the cached value is honoured.
    r_interface.HAS_RPY2 = False
    assert r_interface._import_rpy2() is False
    # Restore so the rest of the suite sees the real environment value.
    r_interface.HAS_RPY2 = None


@pytest.mark.skipif(not _RPY2_LIVE, reason="rpy2 cannot initialize the system R install")
def test_import_rpy2_actually_imports_when_cache_empty():
    """With the cache cleared and a working rpy2/R, the probe must succeed."""
    r_interface.HAS_RPY2 = None
    assert r_interface._import_rpy2() is True
    assert r_interface.HAS_RPY2 is True


def test_check_r_availability_raises_when_rpy2_missing():
    with patch.object(r_interface, "_import_rpy2", return_value=False):
        # Avoid the cached-True short-circuit
        r_interface.HAS_RPY2 = False
        with pytest.raises(r_interface.RInterfaceError, match=r"rpy2 is not installed"):
            r_interface.check_r_availability()
    r_interface.HAS_RPY2 = None


# ---------------------------------------------------------------------------
# setup_r_environment error-message partitioning
# ---------------------------------------------------------------------------


def test_setup_r_environment_emits_cran_install_for_cran_packages():
    """CRAN packages should trigger an ``install.packages(...)`` instruction."""
    with (
        patch.object(r_interface, "check_r_availability", return_value=True),
        patch.object(r_interface, "check_r_package", return_value=False),
    ):
        with pytest.raises(r_interface.RInterfaceError) as excinfo:
            r_interface.setup_r_environment(["dplyr", "ggplot2"])
    msg = str(excinfo.value)
    assert "dplyr" in msg
    assert "ggplot2" in msg
    assert 'install.packages(c("dplyr", "ggplot2"))' in msg
    assert "BiocManager" not in msg


def test_setup_r_environment_emits_bioc_install_for_bioconductor_packages():
    """Bioconductor packages (whitelisted in the helper) trigger BiocManager."""
    with (
        patch.object(r_interface, "check_r_availability", return_value=True),
        patch.object(r_interface, "check_r_package", return_value=False),
    ):
        with pytest.raises(r_interface.RInterfaceError) as excinfo:
            r_interface.setup_r_environment(["vsn"])
    msg = str(excinfo.value)
    assert "vsn" in msg
    assert "BiocManager" in msg
    assert 'BiocManager::install(c("vsn"))' in msg


def test_setup_r_environment_partitions_mixed_packages():
    """A mix produces both CRAN and Bioc instructions, each with the right
    package list."""
    with (
        patch.object(r_interface, "check_r_availability", return_value=True),
        patch.object(r_interface, "check_r_package", return_value=False),
    ):
        with pytest.raises(r_interface.RInterfaceError) as excinfo:
            r_interface.setup_r_environment(["dplyr", "vsn", "limma"])
    msg = str(excinfo.value)
    assert 'install.packages(c("dplyr"))' in msg
    assert 'BiocManager::install(c("vsn", "limma"))' in msg


def test_setup_r_environment_returns_none_when_all_packages_present():
    """No exception when every package check returns True."""
    with (
        patch.object(r_interface, "check_r_availability", return_value=True),
        patch.object(r_interface, "check_r_package", return_value=True),
    ):
        # Should not raise
        assert r_interface.setup_r_environment(["dplyr", "vsn"]) is None


# ---------------------------------------------------------------------------
# check_r_package delegates to rpy2.robjects.packages.isinstalled
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RPY2_LIVE, reason="rpy2 cannot initialize the system R install")
def test_check_r_package_delegates_to_isinstalled():
    """A package that is definitely not installed must return False."""
    assert r_interface.check_r_package("definitely_not_a_real_package") is False
