#!/usr/bin/env python3
"""
Create Cross-Frequency Task Dataset

This script creates a cross-frequency task dataset under processed_channels/cross_freq/task/
that combines:
- Train: Direct copy from sub6/task/train
- Test: sub6 channels + mmWave beam labels

The result is a test set with 3.5GHz channels (sub6) and 28GHz beam labels (mmWave).
"""

import numpy as np
import shutil
from pathlib import Path
import os
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

# Source directories
SUB6_TRAIN_DIR = Path("processed_channels/sub6/task/train")
SUB6_TEST_DIR = Path("processed_channels/sub6/task/test")
MMWAVE_TEST_DIR = Path("processed_channels/mmwave/test")

# Target directory
CROSS_FREQ_DIR = Path("processed_channels/cross_freq/task")
CROSS_FREQ_TRAIN_DIR = CROSS_FREQ_DIR / "train"
CROSS_FREQ_TEST_DIR = CROSS_FREQ_DIR / "test"


# =============================================================================
# Utility Functions
# =============================================================================

def setup_directories():
    """Create the cross-frequency task directory structure."""
    CROSS_FREQ_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    CROSS_FREQ_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure:")
    print(f"  - {CROSS_FREQ_TRAIN_DIR}")
    print(f"  - {CROSS_FREQ_TEST_DIR}")


def get_matching_files():
    """
    Find matching files between sub6 and mmWave test directories.
    
    Returns:
        list: List of tuples (sub6_file, mmwave_file, city_name)
    """
    matching_files = []
    
    # Get all sub6 test files
    sub6_files = list(SUB6_TEST_DIR.glob("*.npz"))
    
    for sub6_file in sub6_files:
        # Extract city name and base station from sub6 filename
        # Example: city_3_houston_3p5_bs000.npz
        filename = sub6_file.name
        parts = filename.split('_')
        
        if len(parts) >= 4 and parts[0] == 'city':
            city_id = parts[1]
            city_name = parts[2]
            bs_part = parts[4]  # bs000, bs001, etc.
            
            # Construct corresponding mmWave filename
            # Example: city_3_houston_28_bs000.npz
            mmwave_filename = f"city_{city_id}_{city_name}_28_{bs_part}"
            mmwave_file = MMWAVE_TEST_DIR / mmwave_filename
            
            if mmwave_file.exists():
                matching_files.append((sub6_file, mmwave_file, city_name))
                print(f"  ✓ {filename} <-> {mmwave_filename}")
            else:
                print(f"  ✗ {filename} -> {mmwave_filename} (not found)")
    
    return matching_files


def copy_train_data():
    """Copy train data directly from sub6/task/train."""
    print(f"\nCopying train data from {SUB6_TRAIN_DIR}...")
    
    train_files = list(SUB6_TRAIN_DIR.glob("*.npz"))
    copied_count = 0
    
    for train_file in tqdm(train_files, desc="Copying train files"):
        target_file = CROSS_FREQ_TRAIN_DIR / train_file.name
        
        if not target_file.exists():
            shutil.copy2(train_file, target_file)
            copied_count += 1
        else:
            print(f"  Skipped (exists): {train_file.name}")
    
    print(f"Copied {copied_count} train files to {CROSS_FREQ_TRAIN_DIR}")
    return copied_count


def create_cross_frequency_test_data(matching_files):
    """
    Create test data by combining sub6 channels with mmWave beam labels.
    
    Args:
        matching_files: List of tuples (sub6_file, mmwave_file, city_name)
    """
    print(f"\nCreating cross-frequency test data...")
    
    processed_count = 0
    failed_count = 0
    
    for sub6_file, mmwave_file, city_name in tqdm(matching_files, desc="Processing test files"):
        try:
            # Load sub6 data (channels, positions, etc.)
            print(f"  Processing {sub6_file.name} + {mmwave_file.name}")
            
            sub6_data = np.load(sub6_file, allow_pickle=True)
            mmwave_data = np.load(mmwave_file, allow_pickle=True)
            
            # Check if both files have beam labels
            if 'beam_labels' not in sub6_data or 'beam_labels' not in mmwave_data:
                print(f"    Warning: Missing beam_labels in one of the files, skipping...")
                failed_count += 1
                continue
            
            # Create output filename (use sub6 naming convention)
            output_filename = sub6_file.name
            output_file = CROSS_FREQ_TEST_DIR / output_filename
            
            if output_file.exists():
                print(f"    Skipped (exists): {output_filename}")
                continue
            
            # Prepare data to save
            save_data = {}
            
            # Copy all data from sub6 file (channels, positions, etc.)
            for key in sub6_data.keys():
                if key != 'beam_labels':  # We'll replace this with mmWave beam labels
                    save_data[key] = sub6_data[key]
            
            # Add mmWave beam labels
            save_data['beam_labels'] = mmwave_data['beam_labels']
            
            # Add metadata about the cross-frequency combination
            save_data['cross_freq_info'] = {
                'channels_source': 'sub6_3p5ghz',
                'beam_labels_source': 'mmwave_28ghz',
                'sub6_file': str(sub6_file),
                'mmwave_file': str(mmwave_file),
                'combination_method': 'cross_frequency_dataset'
            }
            
            # Save the combined data
            np.savez_compressed(output_file, **save_data)
            
            print(f"    ✓ Created: {output_filename}")
            processed_count += 1
            
            # Clean up memory
            del sub6_data, mmwave_data
            
        except Exception as e:
            print(f"    ✗ Error processing {sub6_file.name}: {str(e)}")
            failed_count += 1
    
    print(f"\nCross-frequency test data creation complete:")
    print(f"  - Successfully processed: {processed_count}")
    print(f"  - Failed: {failed_count}")
    
    return processed_count, failed_count


def verify_dataset():
    """Verify the created cross-frequency dataset."""
    print(f"\nVerifying cross-frequency dataset...")
    
    # Check train directory
    train_files = list(CROSS_FREQ_TRAIN_DIR.glob("*.npz"))
    print(f"  Train files: {len(train_files)}")
    
    # Check test directory
    test_files = list(CROSS_FREQ_TEST_DIR.glob("*.npz"))
    print(f"  Test files: {len(test_files)}")
    
    # Verify a few test files
    if test_files:
        print(f"\n  Verifying test file structure...")
        sample_file = test_files[0]
        data = np.load(sample_file, allow_pickle=True)
        
        print(f"    Sample file: {sample_file.name}")
        print(f"    Keys: {list(data.keys())}")
        print(f"    Channels shape: {data['channels'].shape}")
        print(f"    Beam labels shape: {data['beam_labels'].shape}")
        
        if 'cross_freq_info' in data:
            print(f"    Cross-freq info: {data['cross_freq_info']}")
    
    print(f"  Total files in cross-frequency dataset: {len(train_files) + len(test_files)}")


def main():
    """Main function to create the cross-frequency dataset."""
    print("Creating Cross-Frequency Task Dataset")
    print("=" * 50)
    print(f"This will create a dataset combining:")
    print(f"  - Train: sub6 channels + sub6 beam labels")
    print(f"  - Test: sub6 channels + mmWave beam labels")
    print(f"  - Output: {CROSS_FREQ_DIR}")
    
    # Setup directories
    setup_directories()
    
    # Find matching files
    print(f"\nFinding matching files between sub6 and mmWave test directories...")
    matching_files = get_matching_files()
    
    if not matching_files:
        print("No matching files found. Exiting.")
        return
    
    print(f"Found {len(matching_files)} matching file pairs")
    
    # Copy train data
    train_count = copy_train_data()
    
    # Create cross-frequency test data
    test_processed, test_failed = create_cross_frequency_test_data(matching_files)
    
    # Verify the dataset
    verify_dataset()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"CROSS-FREQUENCY DATASET CREATION COMPLETE")
    print(f"=" * 60)
    print(f"Train files copied: {train_count}")
    print(f"Test files created: {test_processed}")
    print(f"Test files failed: {test_failed}")
    print(f"Total files: {train_count + test_processed}")
    print(f"Output directory: {CROSS_FREQ_DIR.absolute()}")
    
    if test_failed == 0:
        print(f"\n🎉 All test files processed successfully!")
    else:
        print(f"\n⚠️  {test_failed} test files failed to process.")
    
    print(f"\nThe cross-frequency dataset is ready for use!")


if __name__ == "__main__":
    main() 