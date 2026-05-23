"""
Preprocessing Module
Provides functionality for frame preprocessing: resize, grayscale, normalize
"""

import cv2
import numpy as np
from typing import Union, Tuple, List, Optional


def resize_frame(frame: np.ndarray, 
                width: int, 
                height: int,
                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Resize a frame to specified dimensions
    
    Args:
        frame (np.ndarray): Input frame
        width (int): Target width
        height (int): Target height
        interpolation (int): OpenCV interpolation method (default: cv2.INTER_LINEAR)
        
    Returns:
        np.ndarray: Resized frame
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Frame must be numpy array, got {type(frame)}")
    
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be positive, got {width}x{height}")
    
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame to grayscale
    
    Args:
        frame (np.ndarray): Input frame (BGR or RGB format)
        
    Returns:
        np.ndarray: Grayscale frame (single channel)
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Frame must be numpy array, got {type(frame)}")
    
    if frame.ndim < 2:
        raise ValueError(f"Frame must be 2D or 3D array, got {frame.ndim}D")
    
    # Check if already grayscale
    if frame.ndim == 2:
        return frame
    
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def normalize(frame: np.ndarray, 
             method: str = "minmax",
             mean: Optional[float] = None,
             std: Optional[float] = None) -> np.ndarray:
    """
    Normalize frame values
    
    Args:
        frame (np.ndarray): Input frame
        method (str): Normalization method - "minmax" or "zscore" (default: "minmax")
        mean (Optional[float]): Mean value for z-score normalization
        std (Optional[float]): Standard deviation for z-score normalization
        
    Returns:
        np.ndarray: Normalized frame (as float32 with values in [0, 1] or standardized)
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Frame must be numpy array, got {type(frame)}")
    
    frame = frame.astype(np.float32)
    
    if method == "minmax":
        # Min-max normalization: (x - min) / (max - min)
        frame_min = frame.min()
        frame_max = frame.max()
        
        if frame_max == frame_min:
            return np.zeros_like(frame)
        
        return (frame - frame_min) / (frame_max - frame_min)
    
    elif method == "zscore":
        # Z-score normalization: (x - mean) / std
        if mean is None:
            mean = frame.mean()
        if std is None:
            std = frame.std()
        
        if std == 0:
            return np.zeros_like(frame)
        
        return (frame - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess(frame: np.ndarray,
              target_width: int,
              target_height: int,
              to_grayscale: bool = True,
              normalize_method: str = "minmax",
              normalize_mean: Optional[float] = None,
              normalize_std: Optional[float] = None) -> np.ndarray:
    """
    Complete preprocessing pipeline: resize, grayscale (optional), normalize
    
    Args:
        frame (np.ndarray): Input frame
        target_width (int): Target width after resizing
        target_height (int): Target height after resizing
        to_grayscale (bool): Convert to grayscale (default: True)
        normalize_method (str): Normalization method (default: "minmax")
        normalize_mean (Optional[float]): Mean for z-score normalization
        normalize_std (Optional[float]): Std for z-score normalization
        
    Returns:
        np.ndarray: Preprocessed frame
    """
    # Step 1: Resize
    processed = resize_frame(frame, target_width, target_height)
    
    # Step 2: Convert to grayscale (optional)
    if to_grayscale:
        processed = grayscale(processed)
    
    # Step 3: Normalize
    processed = normalize(processed, 
                         method=normalize_method,
                         mean=normalize_mean,
                         std=normalize_std)
    
    return processed


def preprocess_batch(frames: List[np.ndarray],
                    target_width: int,
                    target_height: int,
                    to_grayscale: bool = True,
                    normalize_method: str = "minmax") -> np.ndarray:
    """
    Preprocess a batch of frames
    
    Args:
        frames (List[np.ndarray]): List of input frames
        target_width (int): Target width
        target_height (int): Target height
        to_grayscale (bool): Convert to grayscale (default: True)
        normalize_method (str): Normalization method (default: "minmax")
        
    Returns:
        np.ndarray: Preprocessed frames array (frames, height, width) or (frames, height, width, channels)
    """
    if not frames:
        raise ValueError("Frames list is empty")
    
    processed_frames = []
    for frame in frames:
        processed = preprocess(frame, target_width, target_height, 
                              to_grayscale, normalize_method)
        processed_frames.append(processed)
    
    return np.stack(processed_frames, axis=0)
