"""
R interface utilities for Pronoms.

This module provides functions for interfacing with R using rpy2,
which is necessary for normalizers that rely on R packages.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# rpy2 is imported lazily to avoid import-time crashes and to allow environments
# without rpy2 to import this module for non-R functionality.
if TYPE_CHECKING:  # type-only imports; never executed at runtime
    import rpy2.robjects as robjects

_R_INITIALIZED = False
HAS_RPY2: bool | None = None  # None = not checked yet


class RInterfaceError(Exception):
    """Exception raised for errors in the R interface."""

    pass


def _import_rpy2() -> bool:
    """
    Lazily check that rpy2 can be imported. This does NOT try to activate
    any converters (e.g., pandas2ri), so converter deprecations won't cause
    a false negative.
    """
    global HAS_RPY2
    if HAS_RPY2 is not None:
        return HAS_RPY2

    try:
        import rpy2
        import rpy2.robjects  # noqa: F401

        HAS_RPY2 = True
    except ImportError:
        HAS_RPY2 = False

    return HAS_RPY2


def check_r_availability() -> bool:
    """
    Check if R and rpy2 are available and (if necessary) initialize R.

    Returns
    -------
    bool
        True if R and rpy2 are available.

    Raises
    ------
    RInterfaceError
        If rpy2 is not installed, or if R initialization fails.
    """
    global _R_INITIALIZED

    if not _import_rpy2():
        raise RInterfaceError("rpy2 is not installed in this Python environment.")

    try:
        # Import embedded lazily
        import rpy2.rinterface_lib.embedded as embedded

        # Newer rpy2 often initializes R automatically. If the attribute exists,
        # we can query it; otherwise, fall back to explicit initialize() on older versions.
        if hasattr(embedded, "isinitialized"):
            # If it's already initialized, great; if not, many rpy2 builds will initialize
            # on first robjects use. We don't force it here unless needed.
            pass
        elif hasattr(embedded, "initialize") and not _R_INITIALIZED:
            # rpy2's stubs declare a 1-tuple but the implementation accepts any
            # tuple of CLI flags; cast to keep mypy happy without a runtime cast.
            embedded.set_initoptions(("--vanilla", "--quiet", "--no-save"))  # type: ignore[arg-type, unused-ignore]
            embedded.initialize()

        _R_INITIALIZED = True
        return True

    except ImportError as e:
        # Extremely rare: embedded missing though rpy2 imported
        raise RInterfaceError(f"rpy2 embedded interface unavailable: {e}") from e
    except Exception as e:
        # Surface the real reason
        raise RInterfaceError(f"rpy2 imported, but R failed to initialize: {type(e).__name__}: {e}") from e


def check_r_package(package_name: str) -> bool:
    """
    Check if an R package is installed.

    Parameters
    ----------
    package_name : str
        Name of the R package to check.

    Returns
    -------
    bool
        True if the package is installed, False otherwise.
    """
    check_r_availability()

    from rpy2.robjects.packages import isinstalled

    return isinstalled(package_name)


def setup_r_environment(required_packages: list[str]) -> None:
    """
    Set up the R environment by checking required packages.

    Parameters
    ----------
    required_packages : List[str]
        List of R packages required for the normalizer.

    Raises
    ------
    RInterfaceError
        If any required R package is not installed.
    """
    check_r_availability()

    missing_packages = [pkg for pkg in required_packages if not check_r_package(pkg)]

    if missing_packages:
        bioc_packages = {"vsn", "limma", "DirectLFQ"}
        cran_packages = [p for p in missing_packages if p not in bioc_packages]
        bioc_missing = [p for p in missing_packages if p in bioc_packages]

        instructions: list[str] = []
        if cran_packages:
            pkg_str = ", ".join(f'"{p}"' for p in cran_packages)
            instructions.append(f"install.packages(c({pkg_str}))")
        if bioc_missing:
            instructions.append('if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")')
            pkg_str = ", ".join(f'"{p}"' for p in bioc_missing)
            instructions.append(f"BiocManager::install(c({pkg_str}))")

        install_cmd = "\n".join(instructions) if instructions else "(no instructions generated)"

        raise RInterfaceError(
            "The following R packages are required but not installed: "
            f"{', '.join(missing_packages)}.\n"
            "Please install them in R using:\n\n"
            f"{install_cmd}"
        )


def convert_to_r_matrix(
    data: np.ndarray,
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
) -> robjects.Matrix:
    """
    Convert a numpy array to an R matrix.

    Parameters
    ----------
    data : np.ndarray
        Numpy array to convert.
    row_names : Optional[List[str]]
        Row names for the R matrix.
    col_names : Optional[List[str]]
        Column names for the R matrix.

    Returns
    -------
    robjects.Matrix
        R matrix object.
    """
    check_r_availability()

    import rpy2.robjects as robjects
    from rpy2.robjects.vectors import StrVector

    # Ensure data is float (R numeric) and C-contiguous
    arr = np.ascontiguousarray(data, dtype=float)

    r_matrix = robjects.r.matrix(
        robjects.FloatVector(arr.ravel(order="C")),
        nrow=arr.shape[0],
        ncol=arr.shape[1],
        byrow=True,
    )

    if row_names is not None:
        try:
            r_matrix.rownames = StrVector(row_names)
        except Exception:
            try:
                robjects.r["rownames<-"](r_matrix, StrVector(row_names))
            except Exception as e:
                warnings.warn(f"Could not set row names: {e}", stacklevel=2)

    if col_names is not None:
        try:
            r_matrix.colnames = StrVector(col_names)
        except Exception:
            try:
                robjects.r["colnames<-"](r_matrix, StrVector(col_names))
            except Exception as e:
                warnings.warn(f"Could not set column names: {e}", stacklevel=2)

    return r_matrix


def convert_from_r_matrix(r_matrix: robjects.Matrix) -> np.ndarray:
    """
    Convert an R matrix to a numpy array.

    Parameters
    ----------
    r_matrix : robjects.Matrix
        R matrix to convert.

    Returns
    -------
    np.ndarray
        Numpy array with shape (nrow, ncol).
    """
    check_r_availability()

    import rpy2.robjects as robjects

    nrow = int(robjects.r.nrow(r_matrix)[0])
    ncol = int(robjects.r.ncol(r_matrix)[0])

    # rpy2 can expose the underlying vector; ensure we reshape correctly.
    np_array = np.asarray(r_matrix, dtype=float).reshape(nrow, ncol, order="C")
    return np_array


def run_r_script(
    script: str,
    data: np.ndarray | None = None,
    row_names: list[str] | None = None,
    col_names: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run an R script with optional input matrix and return selected results.

    Parameters
    ----------
    script : str
        R script to run.
    data : Optional[np.ndarray]
        Input data matrix made available in R as `input_data`.
    row_names : Optional[List[str]]
        Row names for the input data.
    col_names : Optional[List[str]]
        Column names for the input data.

    Returns
    -------
    Dict[str, Any]
        Dictionary of results extracted from the R global environment.

    Raises
    ------
    RInterfaceError
        If the R script execution fails.
    """
    check_r_availability()

    import rpy2.robjects as robjects

    # Create a temporary file for the R script
    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        if data is not None:
            r_data = convert_to_r_matrix(data, row_names, col_names)
            robjects.globalenv["input_data"] = r_data

        # Execute the R script
        robjects.r.source(script_path)

        # Collect results if present
        results: dict[str, Any] = {}

        if bool(robjects.r("exists('normalized_data')")[0]):
            results["normalized_data"] = convert_from_r_matrix(robjects.r("normalized_data"))

        for var_name in ("stats", "parameters", "diagnostics"):
            if bool(robjects.r(f"exists('{var_name}')")[0]):
                results[var_name] = robjects.r(var_name)

        return results

    except Exception as e:
        raise RInterfaceError(f"Error executing R script: {type(e).__name__}: {e}") from e
    finally:
        try:
            if os.path.exists(script_path):
                os.remove(script_path)
        except Exception:
            # Best effort cleanup
            pass


# ---------- OPTIONAL: Pandas <-> R data.frame helpers (no global activation) ----------


def pandas_to_r_df(df: pd.DataFrame) -> Any:
    """
    Convert a pandas DataFrame to an R data.frame using a local conversion context.
    """
    check_r_availability()
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter):
        return pandas2ri.py2rpy(df)


def r_df_to_pandas(r_obj: Any) -> pd.DataFrame:
    """
    Convert an R data.frame to a pandas DataFrame using a local conversion context.
    """
    check_r_availability()
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    with localconverter(pandas2ri.converter):
        return pandas2ri.rpy2py(r_obj)


# ---------- OPTIONAL: Quiet rpy2’s console noise (uncomment if desired) ----------
# import logging
# logging.getLogger("rpy2").setLevel(logging.ERROR)
# from rpy2.rinterface_lib import callbacks
# def _quiet_console(_msg: str) -> None:
#     pass
# callbacks.consolewrite_print = _quiet_console
# callbacks.consolewrite_warnerror = _quiet_console
