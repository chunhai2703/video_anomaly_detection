"""
Video Loader Module
Provides functionality to load and read video files
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional


def load_video(path: str) -> cv2.VideoCapture:
    """
    Load a video from file path
    
    Args:
        path (str): Path to the video file
        
    Returns:
        cv2.VideoCapture: Video capture object
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file cannot be opened
    """
    if not isinstance(path, str):
        raise TypeError(f"Path must be string, got {type(path)}")
    
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """
    Extract video information
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        
    Returns:
        dict: Video information containing fps, frame_count, width, height
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": frame_count / fps if fps > 0 else 0
    }


def read_frames(cap: cv2.VideoCapture, 
                start_frame: int = 0, 
                end_frame: Optional[int] = None) -> Generator[np.ndarray, None, None]:
    """
    Read frames from video sequentially
    
    Args:
        cap (cv2.VideoCapture): Video capture object
        start_frame (int): Starting frame index (default: 0)
        end_frame (Optional[int]): Ending frame index (default: None - read all)
        
    Yields:
        np.ndarray: Frame as numpy array (BGR format)
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
        frame_idx += 1
