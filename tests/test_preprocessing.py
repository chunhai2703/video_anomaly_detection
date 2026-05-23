"""
Test module for preprocessing functions
Tests video_loader, frame_extractor, and preprocessing modules
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import (
    load_video,
    get_video_info,
    extract_frames,
    resize_frame,
    grayscale,
    normalize,
    preprocess,
    preprocess_batch
)


def test_video_loader():
    """Test video_loader functions"""
    print("\n=== Testing video_loader ===")
    
    video_path = str(project_root / "test.mp4")
    
    if not Path(video_path).exists():
        print(f"Warning: Test video not found at {video_path}")
        print("Skipping video_loader tests")
        return False
    
    try:
        # Test load_video
        print("Testing load_video()...")
        cap = load_video(video_path)
        print(f"✓ Successfully loaded video")
        
        # Test get_video_info
        print("Testing get_video_info()...")
        info = get_video_info(cap)
        print(f"✓ Video info: {info}")
        print(f"  - FPS: {info['fps']}")
        print(f"  - Frames: {info['frame_count']}")
        print(f"  - Resolution: {info['width']}x{info['height']}")
        print(f"  - Duration: {info['duration']:.2f}s")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Error in video_loader test: {e}")
        return False


def test_frame_extractor():
    """Test frame_extractor functions"""
    print("\n=== Testing frame_extractor ===")
    
    video_path = str(project_root / "test.mp4")
    
    if not Path(video_path).exists():
        print(f"Warning: Test video not found at {video_path}")
        print("Skipping frame_extractor tests")
        return False
    
    try:
        # Test extract_frames
        print("Testing extract_frames()...")
        frames = extract_frames(video_path, frame_interval=5)
        print(f"✓ Extracted {len(frames)} frames (interval=5)")
        print(f"  - Frame shape: {frames[0].shape}")
        print(f"  - Frame dtype: {frames[0].dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in frame_extractor test: {e}")
        return False


def test_resize_frame():
    """Test resize_frame function"""
    print("\n=== Testing resize_frame ===")
    
    try:
        # Create a test frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print(f"Original frame shape: {frame.shape}")
        
        # Test resize
        resized = resize_frame(frame, 224, 224)
        assert resized.shape == (224, 224, 3), f"Expected (224, 224, 3), got {resized.shape}"
        print(f"✓ Resized to: {resized.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in resize_frame test: {e}")
        return False


def test_grayscale():
    """Test grayscale function"""
    print("\n=== Testing grayscale ===")
    
    try:
        # Create a test frame
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print(f"Original frame shape: {frame.shape}")
        
        # Test grayscale
        gray = grayscale(frame)
        assert gray.ndim == 2, f"Expected 2D array, got {gray.ndim}D"
        assert gray.shape == (480, 640), f"Expected (480, 640), got {gray.shape}"
        print(f"✓ Converted to grayscale: {gray.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in grayscale test: {e}")
        return False


def test_normalize():
    """Test normalize function"""
    print("\n=== Testing normalize ===")
    
    try:
        frame = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
        
        # Test minmax normalization
        print("Testing minmax normalization...")
        normalized = normalize(frame, method="minmax")
        assert normalized.min() >= 0 and normalized.max() <= 1, \
            f"Expected values in [0, 1], got min={normalized.min()}, max={normalized.max()}"
        print(f"✓ Minmax normalization: min={normalized.min():.4f}, max={normalized.max():.4f}")
        
        # Test zscore normalization
        print("Testing zscore normalization...")
        normalized = normalize(frame, method="zscore")
        mean = normalized.mean()
        std = normalized.std()
        assert abs(mean) < 0.1 and abs(std - 1.0) < 0.1, \
            f"Expected mean≈0, std≈1, got mean={mean:.4f}, std={std:.4f}"
        print(f"✓ Z-score normalization: mean={mean:.4f}, std={std:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in normalize test: {e}")
        return False


def test_preprocess():
    """Test preprocess function"""
    print("\n=== Testing preprocess (full pipeline) ===")
    
    try:
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        print(f"Original frame shape: {frame.shape}")
        
        # Test full preprocessing
        preprocessed = preprocess(frame, 224, 224, to_grayscale=True)
        assert preprocessed.shape == (224, 224), f"Expected (224, 224), got {preprocessed.shape}"
        assert preprocessed.dtype == np.float32, f"Expected float32, got {preprocessed.dtype}"
        assert preprocessed.min() >= 0 and preprocessed.max() <= 1, \
            f"Expected normalized values in [0, 1], got min={preprocessed.min()}, max={preprocessed.max()}"
        print(f"✓ Preprocessed frame shape: {preprocessed.shape}")
        print(f"  - dtype: {preprocessed.dtype}")
        print(f"  - range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in preprocess test: {e}")
        return False


def test_preprocess_batch():
    """Test preprocess_batch function"""
    print("\n=== Testing preprocess_batch ===")
    
    try:
        # Create test frames
        frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
        print(f"Input: {len(frames)} frames of shape {frames[0].shape}")
        
        # Test batch preprocessing
        batch_preprocessed = preprocess_batch(frames, 224, 224, to_grayscale=True)
        assert batch_preprocessed.shape == (5, 224, 224), \
            f"Expected (5, 224, 224), got {batch_preprocessed.shape}"
        print(f"✓ Batch preprocessed shape: {batch_preprocessed.shape}")
        print(f"  - dtype: {batch_preprocessed.dtype}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in preprocess_batch test: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("PREPROCESSING MODULE TESTS")
    print("=" * 60)
    
    results = {
        "video_loader": test_video_loader(),
        "frame_extractor": test_frame_extractor(),
        "resize_frame": test_resize_frame(),
        "grayscale": test_grayscale(),
        "normalize": test_normalize(),
        "preprocess": test_preprocess(),
        "preprocess_batch": test_preprocess_batch(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
