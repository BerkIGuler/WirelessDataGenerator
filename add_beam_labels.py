#!/usr/bin/env python3
"""
Script to add beam labels to processed DeepMIMO channel data files.

This script reads each .npz file in the specified data folder, calculates
best beam indices using the calculate_best_beams function, and saves the
beam labels as a new key in the same file.
"""

import numpy as np
import argparse
from pathlib import Path
import deepmimo as dm
from tqdm import tqdm
import sys


def to_dbm(power):
    """Convert power values to dBm."""
    return 10 * np.log10(power) + 30


def calculate_best_beams(dataset, beam_sizes, fov=360):
    """
    Calculate best beam indices for given beam sizes.
    
    Args:
        dataset: Loaded .npz data
        beam_sizes: List of beam sizes to test
        fov: Field of view in degrees (default: 360)
        
    Returns:
        numpy.ndarray: Beam labels for each user and beam size
    """
    start = -fov/2
    active_users = dataset['channels'].shape[0]
    bs_antenna_shape = dataset['ch_params_info'].item()["bs_antenna_shape"]

    beam_labels = np.zeros((active_users, len(beam_sizes)))
    for beam_idx, beam_size in enumerate(beam_sizes):
        end = fov/2 - fov/beam_size
        beam_angles = np.around(np.linspace(start, end, beam_size), 2)
        F1 = np.array([dm.steering_vec(bs_antenna_shape, phi=azi).squeeze()
               for azi in beam_angles])
        mean_amplitude = np.abs(F1 @ dataset['channels']).mean(axis=-1).squeeze(1)
        mean_amplitude_dbm = np.around(to_dbm(mean_amplitude), 3)

        best_beams = np.argmax(mean_amplitude_dbm, axis=1)
        beam_labels[:, beam_idx] = best_beams
    
    return beam_labels


def process_single_file(file_path, beam_sizes, fov=360):
    """
    Process a single .npz file: load, calculate beams, save.
    
    Args:
        file_path: Path to the .npz file
        beam_sizes: List of beam sizes to test
        fov: Field of view in degrees
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load the file
        data = np.load(file_path, allow_pickle=True)
        
        # Convert to dict for easier manipulation
        data_dict = dict(data)
        
        # Check if beam labels already exist
        if 'beam_labels' in data_dict:
            print(f"    Warning: {file_path.name} already has beam_labels, skipping...")
            data.close()
            return True
        
        # Check required keys exist
        required_keys = ['channels', 'active_users_count', 'ch_params_info']
        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            print(f"    Error: Missing required keys: {missing_keys}")
            data.close()
            return False
        
        # Calculate beam labels
        beam_labels = calculate_best_beams(data_dict, beam_sizes, fov)
        
        # Add beam labels to data
        data_dict['beam_labels'] = beam_labels
        
        # Add metadata about beam calculation
        data_dict['beam_calculation_info'] = {
            'beam_sizes': beam_sizes,
            'fov_degrees': fov,
            'calculation_method': 'steering_vector_optimization'
        }
        
        # Close the loaded data
        data.close()
        
        # Save back to the same file
        np.savez_compressed(file_path, **data_dict)
        
        print(f"    ✓ Added beam labels for beam sizes {beam_sizes}")
        return True
        
    except Exception as e:
        print(f"    Error processing {file_path.name}: {str(e)}")
        return False


def main():
    """Main function to process all .npz files in the data folder."""
    parser = argparse.ArgumentParser(
        description="Add beam labels to processed DeepMIMO channel data files"
    )
    parser.add_argument(
        "data_folder", 
        type=str, 
        help="Path to folder containing .npz files"
    )
    parser.add_argument(
        "--beam-sizes", 
        type=int, 
        nargs="+", 
        default=[16, 32, 64],
        help="Beam sizes to test (default: 16 32 64)"
    )
    parser.add_argument(
        "--fov", 
        type=float, 
        default=180.0,
        help="Field of view in degrees (default: 360)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    data_folder = Path(args.data_folder)
    
    if not data_folder.exists():
        print(f"Error: Data folder does not exist: {data_folder}")
        sys.exit(1)
    
    if not data_folder.is_dir():
        print(f"Error: {data_folder} is not a directory")
        sys.exit(1)
    
    # Find all .npz files
    npz_files = list(data_folder.glob("*.npz"))
    
    if not npz_files:
        print(f"No .npz files found in {data_folder}")
        sys.exit(0)
    
    print(f"Found {len(npz_files)} .npz files in {data_folder}")
    print(f"Beam sizes: {args.beam_sizes}")
    print(f"Field of view: {args.fov}°")
    
    if args.dry_run:
        print("\nDRY RUN - Files that would be processed:")
        for file_path in npz_files:
            print(f"  - {file_path.name}")
        print("\nRun without --dry-run to actually process the files.")
        return
    
    # Process files
    print(f"\nProcessing {len(npz_files)} files...")
    
    successful = 0
    failed = 0
    
    for file_path in tqdm(npz_files, desc="Processing files"):
        print(f"\nProcessing: {file_path.name}")
        
        if process_single_file(file_path, args.beam_sizes, args.fov):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files: {len(npz_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*successful/len(npz_files):.1f}%")
    
    if failed > 0:
        print(f"\n⚠️  {failed} files failed to process. Check the error messages above.")
    else:
        print(f"\n🎉 All files processed successfully!")


if __name__ == "__main__":
    main() 