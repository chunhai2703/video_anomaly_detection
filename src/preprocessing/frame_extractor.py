"""
Frame Extractor Module
Provides functionality to extract frames from videos
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from .video_loader import load_video, get_video_info


def extract_frames(video_path: str, 
                   output_dir: Optional[str] = None,
                   frame_interval: int = 1,
                   start_frame: int = 0,
                   end_frame: Optional[int] = None) -> List[np.ndarray]:
    """
    Extract frames from a video file
    
    Args:
        video_path (str): Path to the video file
        output_dir (Optional[str]): Directory to save extracted frames (default: None - only return in memory)
        frame_interval (int): Extract every nth frame (default: 1 - extract all frames)
        start_frame (int): Starting frame index (default: 0)
        end_frame (Optional[int]): Ending frame index (default: None - extract all)
        
    Returns:
        List[np.ndarray]: List of extracted frames
    """
    cap = load_video(video_path)
    video_info = get_video_info(cap)
    
    if end_frame is None:
        end_frame = video_info["frame_count"]
    
    frames = []
    frame_count = 0
    frame_idx = start_frame
    
    # Create output directory if specified
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frames.append(frame)
            
            # Save frame if output directory is specified
            if output_dir is not None:
                frame_filename = f"frame_{frame_count:06d}.png"
                cv2.imwrite(str(output_path / frame_filename), frame)
            
            frame_count += 1
        
        frame_idx += 1
    
    cap.release()
    
    return frames


def extract_frames_batch(video_paths: List[str], 
                        output_base_dir: Optional[str] = None,
                        frame_interval: int = 1) -> dict:
    """
    Extract frames from multiple video files
    
    Args:
        video_paths (List[str]): List of paths to video files
        output_base_dir (Optional[str]): Base directory to save extracted frames
        frame_interval (int): Extract every nth frame (default: 1)
        
    Returns:
        dict: Dictionary with video paths as keys and extracted frames as values
    """
    results = {}
    
    for video_path in video_paths:
        output_dir = None
        if output_base_dir is not None:
            video_name = Path(video_path).stem
            output_dir = str(Path(output_base_dir) / video_name)
        
        frames = extract_frames(video_path, output_dir, frame_interval)
        results[video_path] = frames
    
    return results


def get_frame_at_index(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """
    Get a specific frame at given index
    
    Args:
        video_path (str): Path to the video file
        frame_index (int): Index of the frame to extract
        
    Returns:
        Optional[np.ndarray]: Frame at the specified index or None if index is out of bounds
    """
    cap = load_video(video_path)
    video_info = get_video_info(cap)
    
    if frame_index >= video_info["frame_count"] or frame_index < 0:
        cap.release()
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None
