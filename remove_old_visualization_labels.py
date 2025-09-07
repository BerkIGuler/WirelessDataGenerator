#!/usr/bin/env python3
"""
Remove existing visualization_labels from cross-freq test files.

This script removes the 'visualization_labels' key from all cross-freq test files
so we can regenerate them correctly with the fixed implementation.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


# Target directory
CROSS_FREQ_TEST_DIR = Path("datasets/cross_freq_different_env_beam_prediction/test")


def remove_visualization_labels(file_path):
    """
    Remove visualization_labels from a single file.
    
    Args:
        file_path: Path to the .npz file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"  Processing {file_path.name}")
        
        # Load the file
        data = np.load(file_path, allow_pickle=True)
        
        # Convert to dict for easier manipulation
        data_dict = dict(data)
        
        # Check if visualization_labels exist
        if 'visualization_labels' not in data_dict:
            print(f"    No visualization_labels found, skipping...")
            data.close()
            return True
        
        # Remove visualization_labels
        del data_dict['visualization_labels']
        print(f"    Removed visualization_labels")
        
        # Close the loaded data
        data.close()
        
        # Save updated file
        print(f"    Saving updated file...")
        np.savez_compressed(file_path, **data_dict)
        
        print(f"    ✓ Updated {file_path.name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing {file_path.name}: {str(e)}")
        return False


def main():
    """Main function to remove visualization_labels from all cross-freq test files."""
    print("Removing Old Visualization Labels from Cross-Freq Test Files")
    print("=" * 60)
    
    if not CROSS_FREQ_TEST_DIR.exists():
        print(f"Error: Cross-freq test directory not found: {CROSS_FREQ_TEST_DIR}")
        return
    
    # Find all .npz files
    npz_files = list(CROSS_FREQ_TEST_DIR.glob("*.npz"))
    
    if not npz_files:
        print("No .npz files found in cross-freq test directory")
        return
    
    print(f"Found {len(npz_files)} .npz files")
    
    # Process files
    print(f"\nRemoving visualization_labels from {len(npz_files)} files...")
    
    successful = 0
    failed = 0
    
    for file_path in tqdm(npz_files, desc="Removing visualization_labels"):
        if remove_visualization_labels(file_path):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n" + "=" * 60)
    print("REMOVAL COMPLETE")
    print("=" * 60)
    print(f"Total files: {len(npz_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} files failed to process.")
    else:
        print(f"\n🎉 All visualization_labels removed successfully!")
        print(f"Now you can regenerate them with the correct implementation.")


if __name__ == "__main__":
    main() 