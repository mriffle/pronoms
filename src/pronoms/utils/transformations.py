"""
Transformation utilities for Pronoms.

This module provides functions for transforming proteomics data.
"""

import numpy as np
from typing import Union, Optional


def log_transform(
    data: np.ndarray, 
    base: Union[int, float, str] = 2, 
    pseudo_count: float = 1.0
) -> np.ndarray:
    """
    Apply logarithmic transformation to data.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to transform.
    base : Union[int, float, str], optional
        Base of the logarithm, by default 2.
        Options: 2, 10, 'e' (natural logarithm).
    pseudo_count : float, optional
        Small value to add to data before log transform to avoid log(0), by default 1.0.
        
    Returns
    -------
    np.ndarray
        Log-transformed data.
        
    Raises
    ------
    ValueError
        If an invalid base is provided.
    """
    data_offset = data + pseudo_count
    
    if base == 2:
        return np.log2(data_offset)
    elif base == 10:
        return np.log10(data_offset)
    elif base == 'e':
        return np.log(data_offset)
    else:
        raise ValueError(f"Invalid log base: {base}. Use 2, 10, or 'e'.")


def scale_data(
    data: np.ndarray, 
    method: str = 'standard', 
    axis: int = 0,
    with_mean: bool = True,
    with_std: bool = True
) -> np.ndarray:
    """
    Scale data using various methods.
    
    Parameters
    ----------
    data : np.ndarray
        Input data to scale.
    method : str, optional
        Scaling method, by default 'standard'.
        Options: 'standard', 'minmax', 'robust', 'l2'.
    axis : int, optional
        Axis along which to scale, by default 0 (scale each feature/column).
        0 = scale columns, 1 = scale rows.
    with_mean : bool, optional
        If True, center the data before scaling, by default True.
    with_std : bool, optional
        If True, scale the data to unit variance, by default True.
        
    Returns
    -------
    np.ndarray
        Scaled data.
        
    Raises
    ------
    ValueError
        If an invalid scaling method is provided.
    """
    if method == 'standard':
        if with_mean:
            mean = np.mean(data, axis=axis, keepdims=True)
            data = data - mean
        
        if with_std:
            std = np.std(data, axis=axis, keepdims=True)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            data = data / std
            
    elif method == 'minmax':
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        # Avoid division by zero
        denominator = np.where((max_val - min_val) == 0, 1.0, max_val - min_val)
        data = (data - min_val) / denominator
        
    elif method == 'robust':
        # Use percentiles for robust scaling
        q25 = np.percentile(data, 25, axis=axis, keepdims=True)
        q75 = np.percentile(data, 75, axis=axis, keepdims=True)
        median = np.median(data, axis=axis, keepdims=True)
        
        iqr = q75 - q25
        # Avoid division by zero
        iqr = np.where(iqr == 0, 1.0, iqr)
        
        data = (data - median) / iqr
        
    elif method == 'l2':
        # L2 normalization (unit norm)
        norm = np.sqrt(np.sum(data**2, axis=axis, keepdims=True))
        # Avoid division by zero
        norm = np.where(norm == 0, 1.0, norm)
        data = data / norm
        
    else:
        raise ValueError(
            f"Invalid scaling method: {method}. "
            "Use 'standard', 'minmax', 'robust', or 'l2'."
        )
        
    return data
