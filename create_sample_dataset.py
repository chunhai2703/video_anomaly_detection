#!/usr/bin/env python3
"""
Create sample UCSD Ped2 dataset structure for testing
Generates synthetic videos to simulate real data
"""

import cv2
import numpy as np
from pathlib import Path
import os


def create_sample_video(output_path, num_frames=100, width=360, height=240, fps=30):
    """Create a synthetic video for testing"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating {output_path.name}...", end=" ")
    
    for i in range(num_frames):
        # Create synthetic frame with moving objects
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add background with some texture
        frame[:, :] = np.random.randint(20, 60, (height, width, 3), dtype=np.uint8)
        
        # Add moving pedestrians (simplified as rectangles)
        # Pedestrian 1
        x1 = int(50 + i * 1.5) % width
        y1 = 80
        cv2.rectangle(frame, (x1, y1), (x1 + 30, y1 + 60), (0, 255, 0), -1)
        
        # Pedestrian 2
        x2 = int(200 - i * 1.2) % width
        y2 = 150
        cv2.rectangle(frame, (x2, y2), (x2 + 30, y2 + 60), (0, 255, 0), -1)
        
        # Occasional anomaly (detected at frame 80-90)
        if 80 <= i <= 90:
            # Anomalous object (faster moving)
            x_anom = int(300 + (i - 80) * 4) % width
            cv2.rectangle(frame, (x_anom, 100), (x_anom + 20, x_anom + 40), (0, 0, 255), -1)
        
        out.write(frame)
    
    out.release()
    print(f"✓ ({num_frames} frames, {width}x{height})")


def create_ground_truth_masks(output_dir, num_frames=100, height=240, width=360):
    """Create simple ground truth masks"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating ground truth masks...", end=" ")
    
    for i in range(num_frames):
        # Normal frames have zeros, anomaly frames have ones in certain regions
        if 80 <= i <= 90:
            mask = np.ones((height, width), dtype=np.uint8) * 255  # Anomaly
        else:
            mask = np.zeros((height, width), dtype=np.uint8)  # Normal
        
        cv2.imwrite(str(output_dir / f"{i:03d}.bmp"), mask)
    
    print(f"✓ ({num_frames} masks)")


def main():
    print("\n" + "="*70)
    print("🎬 Creating Sample UCSD Ped2 Dataset")
    print("="*70)
    
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    
    # Create training videos (3 videos × 100 frames each)
    print("\n📹 Training Videos:")
    train_dir = dataset_dir / "train" / "Train"
    
    for i in range(1, 4):
        video_dir = train_dir / f"{i:02d}"
        video_path = video_dir / f"{i:02d}.avi"
        gt_dir = video_dir / "gt"
        
        create_sample_video(video_path, num_frames=100)
        create_ground_truth_masks(gt_dir, num_frames=100)
    
    # Create test videos (2 videos × 100 frames each)
    print("\n📹 Test Videos:")
    test_dir = dataset_dir / "test" / "Test"
    
    for i in range(1, 3):
        video_dir = test_dir / f"Test{i:03d}_x264"
        video_path = video_dir / f"Test{i:03d}_x264.avi"
        gt_dir = video_dir / "gt"
        
        create_sample_video(video_path, num_frames=100)
        create_ground_truth_masks(gt_dir, num_frames=100)
    
    # Create ROI files (placeholder)
    print("\n📄 ROI Files:")
    for roi_path in dataset_dir.glob("*/*/ROI.mat"):
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        with open(roi_path, 'w') as f:
            f.write("% ROI file placeholder")
        print(f"✓ Created {roi_path.relative_to(dataset_dir)}")
    
    # Verify structure
    print("\n" + "-"*70)
    print("✅ Sample Dataset Created!")
    print("-"*70)
    
    print("\n📂 Dataset Structure:")
    print(f"""
    {dataset_dir}/
    ├── train/
    │   └── Train/
    │       ├── 01/
    │       │   ├── 01.avi (100 frames)
    │       │   └── gt/ (100 masks)
    │       ├── 02/
    │       └── 03/
    └── test/
        └── Test/
            ├── Test001_x264/
            │   ├── Test001_x264.avi (100 frames)
            │   └── gt/ (100 masks)
            └── Test002_x264/
    """)
    
    print("📊 Summary:")
    print("  • Training videos: 3")
    print("  • Test videos: 2")
    print("  • Frames per video: 100")
    print("  • Resolution: 360×240")
    print("  • FPS: 30")
    print("  • Total size: ~50MB")
    
    print("\n🚀 Next Steps:")
    print("  1. Run verification: python3 verify_setup.py")
    print("  2. Run preprocessing tests: python3 tests/test_preprocessing.py")
    print("  3. Start developing other modules (Dev 2-5)")
    print("  4. Later replace with real UCSD Ped2 data")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
