"""
R interface utilities for Pronoms.

This module provides functions for interfacing with R using rpy2,
which is necessary for normalizers that rely on R packages.
"""

import os
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Union

# Import rpy2 with error handling
try:
    import rpy2
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr, isinstalled
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
    import rpy2.rinterface_lib.embedded as embedded
    
    # Activate pandas conversion
    pandas2ri.activate()
    
    # Flag to track if R is initialized
    _R_INITIALIZED = False
    
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    _R_INITIALIZED = False


class RInterfaceError(Exception):
    """Exception raised for errors in the R interface."""
    pass


def check_r_availability():
    """
    Check if R and rpy2 are available.
    
    Returns
    -------
    bool
        True if R and rpy2 are available, False otherwise.
        
    Raises
    ------
    RInterfaceError
        If rpy2 is not installed.
    """
    global _R_INITIALIZED
    
    if not HAS_RPY2:
        raise RInterfaceError(
            "rpy2 is not installed. Install it with 'pip install rpy2>=3.5'"
        )
    
    try:
        # Set R options before initialization if not already initialized
        if not _R_INITIALIZED:
            try:
                # Suppress R warnings
                embedded.set_initoptions(('--vanilla', '--quiet', '--no-save'))
            except RuntimeError:
                # If we get here, R is already initialized
                pass
            
            # Try to initialize R
            embedded.initialize()
            _R_INITIALIZED = True
            
        return True
    except Exception as e:
        raise RInterfaceError(f"Failed to initialize R: {str(e)}")


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
    return isinstalled(package_name)


def setup_r_environment(required_packages: List[str]) -> None:
    """
    Set up the R environment by checking and suggesting installation of required packages.
    
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
    
    missing_packages = []
    for package in required_packages:
        if not check_r_package(package):
            missing_packages.append(package)
    
    if missing_packages:
        # Construct installation instructions
        bioc_packages = ["vsn", "limma", "DirectLFQ"]
        cran_packages = [pkg for pkg in missing_packages if pkg not in bioc_packages]
        bioc_missing = [pkg for pkg in missing_packages if pkg in bioc_packages]
        
        instructions = []
        
        if cran_packages:
            pkg_str = ', '.join([f'"{pkg}"' for pkg in cran_packages])
            instructions.append(f'install.packages(c({pkg_str}))')
        
        if bioc_missing:
            instructions.append('if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")')
            pkg_str = ', '.join([f'"{pkg}"' for pkg in bioc_missing])
            instructions.append(f'BiocManager::install(c({pkg_str}))')
        
        install_cmd = '\n'.join(instructions)
        
        raise RInterfaceError(
            f"The following R packages are required but not installed: {', '.join(missing_packages)}.\n"
            f"Please install them in R using:\n\n{install_cmd}"
        )


def convert_to_r_matrix(data: np.ndarray, row_names: Optional[List[str]] = None, 
                       col_names: Optional[List[str]] = None) -> 'robjects.Matrix':
    """
    Convert a numpy array to an R matrix.
    
    Parameters
    ----------
    data : np.ndarray
        Data to convert.
    row_names : Optional[List[str]], optional
        Row names for the R matrix, by default None.
    col_names : Optional[List[str]], optional
        Column names for the R matrix, by default None.
        
    Returns
    -------
    robjects.Matrix
        R matrix object.
    """
    check_r_availability()
    
    # Convert to R matrix
    r_matrix = robjects.r.matrix(
        robjects.FloatVector(data.flatten()),
        nrow=data.shape[0],
        ncol=data.shape[1],
        byrow=True
    )
    
    # Set row and column names if provided
    if row_names is not None:
        robjects.r.rownames(r_matrix, robjects.StrVector(row_names))
    
    if col_names is not None:
        robjects.r.colnames(r_matrix, robjects.StrVector(col_names))
    
    return r_matrix


def convert_from_r_matrix(r_matrix: 'robjects.Matrix') -> np.ndarray:
    """
    Convert an R matrix to a numpy array.
    
    Parameters
    ----------
    r_matrix : robjects.Matrix
        R matrix to convert.
        
    Returns
    -------
    np.ndarray
        Numpy array.
    """
    check_r_availability()
    
    # Get dimensions
    nrow = robjects.r.nrow(r_matrix)[0]
    ncol = robjects.r.ncol(r_matrix)[0]
    
    # Convert to numpy array
    np_array = np.array(r_matrix).reshape(nrow, ncol)
    
    return np_array


def run_r_script(script: str, data: Optional[np.ndarray] = None, 
                row_names: Optional[List[str]] = None, 
                col_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Run an R script with input data and return the results.
    
    Parameters
    ----------
    script : str
        R script to run.
    data : Optional[np.ndarray], optional
        Input data for the script, by default None.
    row_names : Optional[List[str]], optional
        Row names for the input data, by default None.
    col_names : Optional[List[str]], optional
        Column names for the input data, by default None.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary of results from the R script.
        
    Raises
    ------
    RInterfaceError
        If the R script execution fails.
    """
    check_r_availability()
    
    # Create a temporary file for the R script
    with tempfile.NamedTemporaryFile(suffix='.R', mode='w', delete=False) as f:
        f.write(script)
        script_path = f.name
    
    try:
        # If data is provided, convert it to an R matrix and assign it
        if data is not None:
            r_data = convert_to_r_matrix(data, row_names, col_names)
            robjects.r.assign("input_data", r_data)
        
        # Execute the R script
        robjects.r.source(script_path)
        
        # Get the results
        results = {}
        
        # Check if normalized_data exists in R environment
        if robjects.r("exists('normalized_data')")[0]:
            results['normalized_data'] = convert_from_r_matrix(robjects.r('normalized_data'))
        
        # Check for additional outputs
        for var_name in ['stats', 'parameters', 'diagnostics']:
            if robjects.r(f"exists('{var_name}')")[0]:
                # Convert to Python object (this handles lists, vectors, etc.)
                results[var_name] = robjects.r(var_name)
        
        return results
    
    except Exception as e:
        raise RInterfaceError(f"Error executing R script: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(script_path):
            os.remove(script_path)
