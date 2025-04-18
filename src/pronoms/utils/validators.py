"""
Validation utilities for Pronoms.

This module provides functions for validating input data for normalization.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def validate_input_data(data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Validate input data for normalization.
    
    Parameters
    ----------
    data : Union[np.ndarray, pd.DataFrame]
        Input data for normalization. If DataFrame, it will be converted to numpy array.
        
    Returns
    -------
    np.ndarray
        Validated numpy array.
        
    Raises
    ------
    TypeError
        If data is not a numpy array or pandas DataFrame.
    ValueError
        If data is empty or has incorrect dimensions.
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    elif not isinstance(data, np.ndarray):
        raise TypeError(
            f"Input data must be a numpy array or pandas DataFrame, got {type(data)}"
        )
    
    if data.ndim != 2:
        raise ValueError(
            f"Input data must be a 2D array with shape (n_proteins, n_samples), got shape {data.shape}"
        )
    
    if data.size == 0:
        raise ValueError("Input data cannot be empty")
    
    return data


def check_nan_inf(data: np.ndarray) -> Tuple[bool, np.ndarray]:
    """
    Check if input data contains NaN or Inf values.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to check.
        
    Returns
    -------
    Tuple[bool, np.ndarray]
        A tuple containing:
        - bool: True if data contains NaN or Inf values, False otherwise.
        - np.ndarray: Boolean mask of NaN or Inf values in data.
    """
    nan_inf_mask = np.isnan(data) | np.isinf(data)
    has_nan_inf = np.any(nan_inf_mask)
    
    return has_nan_inf, nan_inf_mask
