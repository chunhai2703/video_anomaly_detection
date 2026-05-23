"""
Verification Script - Check Dataset Setup & Preprocessing Pipeline
Tests if everything is set up correctly and working
"""

import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent  # verify_setup.py is in project root
sys.path.insert(0, str(project_root))

from src.preprocessing import (
    load_video,
    get_video_info,
    extract_frames,
    preprocess,
    preprocess_batch
)


def check_directory_structure():
    """Verify dataset directory structure"""
    print("\n" + "="*70)
    print("1️⃣  CHECKING DIRECTORY STRUCTURE")
    print("="*70)
    
    dataset_path = project_root / "dataset"
    expected_dirs = ["train", "test"]
    
    print(f"\nDataset path: {dataset_path}")
    
    all_exists = True
    for dir_name in expected_dirs:
        dir_path = dataset_path / dir_name
        exists = dir_path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {dir_name}/ - {'EXISTS' if exists else 'MISSING'}")
        all_exists = all_exists and exists
    
    return all_exists


def check_test_video():
    """Check if test video exists"""
    print("\n" + "="*70)
    print("2️⃣  CHECKING TEST VIDEO")
    print("="*70)
    
    test_video = project_root / "test.mp4"
    
    if test_video.exists():
        size_mb = test_video.stat().st_size / (1024 * 1024)
        print(f"✓ Test video found: {test_video}")
        print(f"  Size: {size_mb:.2f} MB")
        return str(test_video)
    else:
        print(f"✗ Test video NOT found at: {test_video}")
        return None


def test_video_loader(video_path):
    """Test video loading functionality"""
    print("\n" + "="*70)
    print("3️⃣  TESTING VIDEO LOADER")
    print("="*70)
    
    try:
        print(f"\nLoading video: {video_path}")
        cap = load_video(video_path)
        print("✓ Video loaded successfully")
        
        info = get_video_info(cap)
        print("\n📊 Video Information:")
        print(f"  • FPS: {info['fps']}")
        print(f"  • Frame Count: {info['frame_count']}")
        print(f"  • Resolution: {info['width']}x{info['height']}")
        print(f"  • Duration: {info['duration']:.2f}s")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"✗ Error loading video: {e}")
        return False


def test_frame_extraction(video_path):
    """Test frame extraction"""
    print("\n" + "="*70)
    print("4️⃣  TESTING FRAME EXTRACTION")
    print("="*70)
    
    try:
        print(f"\nExtracting frames (every 5th frame)...")
        frames = extract_frames(video_path, frame_interval=5)
        
        print(f"✓ Successfully extracted {len(frames)} frames")
        print(f"  • Frame shape: {frames[0].shape}")
        print(f"  • Frame dtype: {frames[0].dtype}")
        
        return frames if len(frames) > 0 else None
        
    except Exception as e:
        print(f"✗ Error extracting frames: {e}")
        return None


def test_preprocessing(frames):
    """Test preprocessing pipeline"""
    print("\n" + "="*70)
    print("5️⃣  TESTING PREPROCESSING PIPELINE")
    print("="*70)
    
    try:
        print(f"\nPreprocessing {len(frames)} frames...")
        print("  • Resizing to 224x224")
        print("  • Converting to grayscale")
        print("  • Normalizing (minmax)")
        
        preprocessed = preprocess_batch(
            frames, 
            target_width=224, 
            target_height=224,
            to_grayscale=True,
            normalize_method="minmax"
        )
        
        print(f"\n✓ Preprocessing successful")
        print(f"  • Output shape: {preprocessed.shape}")
        print(f"  • Output dtype: {preprocessed.dtype}")
        print(f"  • Value range: [{preprocessed.min():.4f}, {preprocessed.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in preprocessing: {e}")
        return False


def test_dataset_ready():
    """Test if dataset is ready for training"""
    print("\n" + "="*70)
    print("6️⃣  DATASET READINESS CHECK")
    print("="*70)
    
    dataset_path = project_root / "dataset"
    train_path = dataset_path / "train"
    test_path = dataset_path / "test"
    
    train_videos = list(train_path.glob("**/*.avi")) + list(train_path.glob("**/*.mp4"))
    test_videos = list(test_path.glob("**/*.avi")) + list(test_path.glob("**/*.mp4"))
    
    print(f"\nTraining videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")
    
    if len(train_videos) == 0 and len(test_videos) == 0:
        print("\n⚠️  STATUS: NOT READY FOR TRAINING")
        print("   → Need to download UCSD Ped2 dataset")
        print("   → See dataset/README.md for instructions")
        return False
    else:
        print("\n✓ STATUS: DATASET READY")
        if train_videos:
            print(f"   → Training data: {len(train_videos)} videos found")
        if test_videos:
            print(f"   → Test data: {len(test_videos)} videos found")
        return True
    

def generate_summary_report(results):
    """Generate final report"""
    print("\n" + "="*70)
    print("📋 VERIFICATION SUMMARY")
    print("="*70)
    
    tests = [
        ("Directory Structure", results.get("dir_structure", False)),
        ("Test Video Found", results.get("test_video", False)),
        ("Video Loader Works", results.get("video_loader", False)),
        ("Frame Extraction Works", results.get("frame_extraction", False)),
        ("Preprocessing Works", results.get("preprocessing", False)),
    ]
    
    print("\nTest Results:")
    for test_name, passed in tests:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    passed_count = sum(1 for _, p in tests if p)
    total_count = len(tests)
    
    print(f"\nOverall: {passed_count}/{total_count} checks passed")
    
    print("\n" + "-"*70)
    if passed_count == total_count:
        print("✅ SETUP SUCCESSFUL - Everything is working correctly!")
        print("\nNext Steps:")
        print("  1. Download UCSD Ped2 dataset (see dataset/README.md)")
        print("  2. Extract to dataset/train/ and dataset/test/")
        print("  3. Run train.py to start training the autoencoder")
    else:
        print("⚠️  SETUP INCOMPLETE - Some checks failed")
        print("\nPlease review the errors above and fix them.")
    
    print("-"*70 + "\n")


def main():
    """Run all verification tests"""
    print("\n" + "🔍 VERIFICATION SCRIPT - Video Anomaly Detection Setup 🔍\n")
    
    results = {}
    
    # Test 1: Directory structure
    results["dir_structure"] = check_directory_structure()
    
    # Test 2: Test video
    test_video = check_test_video()
    results["test_video"] = test_video is not None
    
    if test_video:
        # Test 3: Video loader
        results["video_loader"] = test_video_loader(test_video)
        
        # Test 4: Frame extraction
        frames = test_frame_extraction(test_video)
        results["frame_extraction"] = frames is not None
        
        # Test 5: Preprocessing
        if frames:
            results["preprocessing"] = test_preprocessing(frames)
    
    # Test 6: Dataset readiness
    test_dataset_ready()
    
    # Generate report
    generate_summary_report(results)


if __name__ == "__main__":
    main()
