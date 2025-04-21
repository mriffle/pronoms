"""
Validation utilities for Pronoms.

This module provides functions for validating input data for normalization.
"""

import numpy as np
from typing import Tuple


def validate_input_data(data: np.ndarray) -> np.ndarray:
    """
    Validate input data for normalization.
    
    Parameters
    ----------
    data : np.ndarray
        Input data for normalization. Must be a 2D numpy array.
        
    Returns
    -------
    np.ndarray
        Validated numpy array.
        
    Raises
    ------
    TypeError
        If data is not a numpy array.
    ValueError
        If data is empty or has incorrect dimensions.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            f"Input data must be a numpy array, got {type(data)}"
        )
    
    if data.ndim != 2:
        raise ValueError(
            f"Input data must be a 2D array with shape (n_samples, n_features), got shape {data.shape}"
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
