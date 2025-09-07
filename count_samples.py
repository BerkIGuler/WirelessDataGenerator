#!/usr/bin/env python3
"""
Count samples in same_freq_different_env dataset

This script counts the total number of samples in the train and test sets
of the datasets/same_freq_different_env_beam_prediction directory.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


# Dataset directory
DATASET_DIR = Path("datasets/same_freq_different_env_beam_prediction")
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


def count_samples_in_directory(directory_path, set_name):
    """
    Count total samples in a directory.
    
    Args:
        directory_path: Path to the directory
        set_name: Name of the set (train/test)
        
    Returns:
        tuple: (total_samples, file_count, file_details)
    """
    if not directory_path.exists():
        print(f"Directory {directory_path} does not exist!")
        return 0, 0, []
    
    files = list(directory_path.glob("*.npz"))
    total_samples = 0
    file_details = []
    
    print(f"\nCounting samples in {set_name} set...")
    print(f"Found {len(files)} files")
    
    for file_path in tqdm(files, desc=f"Processing {set_name} files"):
        try:
            # Load the file
            data = np.load(file_path, allow_pickle=True)
            
            # Get the number of samples (usually from channels or beam_labels)
            if 'channels' in data:
                num_samples = data['channels'].shape[0]
            elif 'beam_labels' in data:
                num_samples = data['beam_labels'].shape[0]
            else:
                # Try to find any array with samples
                for key, value in data.items():
                    if hasattr(value, 'shape') and len(value.shape) > 0:
                        num_samples = value.shape[0]
                        break
                else:
                    num_samples = 0
            
            total_samples += num_samples
            
            # Store file details
            file_details.append({
                'filename': file_path.name,
                'samples': num_samples,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024)
            })
            
            print(f"  {file_path.name}: {num_samples:,} samples")
            
        except Exception as e:
            print(f"  Error reading {file_path.name}: {e}")
            file_details.append({
                'filename': file_path.name,
                'samples': 0,
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'error': str(e)
            })
    
    return total_samples, len(files), file_details


def main():
    """Main function to count samples in both train and test sets."""
    print("Sample Count for same_freq_different_env Dataset")
    print("=" * 50)
    
    # Count train samples
    train_samples, train_files, train_details = count_samples_in_directory(TRAIN_DIR, "train")
    
    # Count test samples
    test_samples, test_files, test_details = count_samples_in_directory(TEST_DIR, "test")
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SAMPLE COUNT SUMMARY")
    print("=" * 60)
    print(f"Train set:")
    print(f"  - Files: {train_files}")
    print(f"  - Total samples: {train_samples:,}")
    print(f"  - Average samples per file: {train_samples/train_files:,.0f}")
    
    print(f"\nTest set:")
    print(f"  - Files: {test_files}")
    print(f"  - Total samples: {test_samples:,}")
    print(f"  - Average samples per file: {test_samples/test_files:,.0f}")
    
    print(f"\nOverall:")
    print(f"  - Total files: {train_files + test_files}")
    print(f"  - Total samples: {train_samples + test_samples:,}")
    print(f"  - Train/Test ratio: {train_samples/test_samples:.2f}:1")
    
    # Detailed breakdown
    print(f"\n" + "=" * 60)
    print("DETAILED BREAKDOWN")
    print("=" * 60)
    
    print(f"\nTrain files:")
    for detail in train_details:
        if 'error' not in detail:
            print(f"  {detail['filename']}: {detail['samples']:,} samples ({detail['file_size_mb']:.1f} MB)")
        else:
            print(f"  {detail['filename']}: ERROR - {detail['error']}")
    
    print(f"\nTest files:")
    for detail in test_details:
        if 'error' not in detail:
            print(f"  {detail['filename']}: {detail['samples']:,} samples ({detail['file_size_mb']:.1f} MB)")
        else:
            print(f"  {detail['filename']}: ERROR - {detail['error']}")
    
    return {
        'train_samples': train_samples,
        'test_samples': test_samples,
        'train_files': train_files,
        'test_files': test_files,
        'total_samples': train_samples + test_samples
    }


if __name__ == "__main__":
    main() 