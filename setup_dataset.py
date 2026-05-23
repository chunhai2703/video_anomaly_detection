#!/usr/bin/env python3
"""
Simple UCSD Ped2 Dataset Downloader
Tải dataset từ Kaggle bằng 1 command đơn giản
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run shell command and return success/failure"""
    if description:
        print(f"\n{description}")
    print(f"$ {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def check_kaggle_installed():
    """Check if kaggle CLI is installed"""
    result = subprocess.run("which kaggle", shell=True, capture_output=True)
    return result.returncode == 0


def check_kaggle_configured():
    """Check if kaggle is configured"""
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_config.exists()


def main():
    print("="*70)
    print("📥 UCSD PED2 DATASET DOWNLOADER")
    print("="*70)
    
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    
    print(f"\nProject: {project_root}")
    print(f"Dataset: {dataset_dir}")
    
    print("\n⚠️  Note: Official UCSD website links are currently inactive")
    print("Using Kaggle as primary source (reliable & tested)")
    
    # Step 1: Check if kaggle is installed
    print("\n" + "-"*70)
    print("STEP 1: Check Kaggle CLI")
    print("-"*70)
    
    if not check_kaggle_installed():
        print("✗ Kaggle CLI not found")
        print("\nInstalling kaggle...")
        if run_command("pip install -q kaggle", ""):
            print("✓ Kaggle installed successfully")
        else:
            print("✗ Failed to install kaggle")
            print("\nTry manually: pip install kaggle")
            sys.exit(1)
    else:
        print("✓ Kaggle CLI found")
    
    # Step 2: Check if kaggle is configured
    print("\n" + "-"*70)
    print("STEP 2: Check Kaggle Configuration")
    print("-"*70)
    
    if not check_kaggle_configured():
        print("✗ Kaggle not configured")
        print("\nSetup Instructions:")
        print("1. Visit: https://www.kaggle.com/settings/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Run:")
        print("   mkdir -p ~/.kaggle")
        print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        print("\nAfter setup, run this script again!")
        sys.exit(1)
    else:
        print("✓ Kaggle configured")
    
    # Step 3: Download dataset
    print("\n" + "-"*70)
    print("STEP 3: Download Dataset from Kaggle")
    print("-"*70)
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = f"kaggle datasets download -d naveenk48/ucsd-anomaly-detection-dataset -q --unzip -p {dataset_dir}"
    
    print(f"\nDownloading to: {dataset_dir}")
    print("This may take 5-15 minutes depending on internet speed...")
    print("(Progress bar will show during download)\n")
    
    if run_command(cmd):
        print("✓ Download completed successfully!")
    else:
        print("✗ Download failed")
        sys.exit(1)
    
    # Step 4: Verify structure
    print("\n" + "-"*70)
    print("STEP 4: Verify Dataset Structure")
    print("-"*70)
    
    train_files = list((dataset_dir / "train").glob("**/*"))
    test_files = list((dataset_dir / "test").glob("**/*"))
    
    if train_files:
        print(f"✓ Training data: {len(train_files)} items found")
        for item in list((dataset_dir / "train").iterdir())[:3]:
            print(f"  • {item.name}")
    else:
        print("✗ Training data: Not found")
    
    if test_files:
        print(f"✓ Test data: {len(test_files)} items found")
        for item in list((dataset_dir / "test").iterdir())[:3]:
            print(f"  • {item.name}")
    else:
        print("✗ Test data: Not found")
    
    # Step 5: Run verification
    print("\n" + "-"*70)
    print("STEP 5: Verify Complete Setup")
    print("-"*70)
    
    print("\nRunning verification script...")
    verify_script = project_root / "verify_setup.py"
    
    if verify_script.exists():
        result = subprocess.run(
            f"source {project_root}/venv/bin/activate && python3 {verify_script}",
            shell=True,
            executable="/bin/bash"
        )
        if result.returncode == 0:
            print("\n✓ All verifications passed!")
        else:
            print("\n⚠️ Some verifications failed - check output above")
    
    # Final summary
    print("\n" + "="*70)
    print("✅ DATASET SETUP COMPLETE!")
    print("="*70)
    
    print("\n📊 Dataset Structure:")
    print(f"""
    {dataset_dir}/
    ├── train/
    │   ├── Train/         ← 36 training videos
    │   └── (archive files)
    │
    └── test/
        ├── Test/         ← 12 test videos  
        └── (archive files)
    """)
    

    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
