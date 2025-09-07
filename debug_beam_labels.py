#!/usr/bin/env python3
"""
Debug script to replicate add_beam_labels.py functionality.

This script:
1. Loads mmWave test channels from processed_channels/mmwave/test
2. Generates beam labels for sizes [4, 8, 16, 32, 64] with FOV=180
3. Saves debug dataset in a new folder
4. Replicates exactly what add_beam_labels.py does
"""

import numpy as np
import deepmimo as dm
from pathlib import Path
from tqdm import tqdm
import shutil


# =============================================================================
# Configuration
# =============================================================================

# Source and target directories
MMWAVE_TEST_DIR = Path("processed_channels/mmwave/test")
DEBUG_DIR = Path("debug_beam_labels")

# Beam sizes to test (same as add_beam_labels.py default)
BEAM_SIZES = [4, 8, 16, 32, 64]

# Field of view (same as add_beam_labels.py default)
FOV = 180


# =============================================================================
# Utility Functions (EXACTLY from add_beam_labels.py)
# =============================================================================

def to_dbm(power):
    """Convert power values to dBm."""
    return 10 * np.log10(power) + 30


def calculate_best_beams(dataset, beam_sizes, fov=180):
    """
    Calculate best beam indices for given beam sizes.
    EXACTLY the same as add_beam_labels.py
    
    Args:
        dataset: Loaded .npz data
        beam_sizes: List of beam sizes to test
        fov: Field of view in degrees (default: 180)
        
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
        
        # Debug: print beam angles for all beam sizes
        print(f"      Beam size {beam_size}: start={start}, end={end}, angles={beam_angles}")
        
        F1 = np.array([dm.steering_vec(bs_antenna_shape, phi=azi).squeeze()
               for azi in beam_angles])
        mean_amplitude = np.abs(F1 @ dataset['channels']).mean(axis=-1).squeeze(1)
        mean_amplitude_dbm = np.around(to_dbm(mean_amplitude), 3)

        best_beams = np.argmax(mean_amplitude_dbm, axis=1)
        beam_labels[:, beam_idx] = best_beams
        
        # Debug: print unique beam labels for this size
        unique_beams = np.unique(best_beams)
        print(f"      Beam size {beam_size}: unique labels {unique_beams} (range: {unique_beams.min()}-{unique_beams.max()})")
    
    return beam_labels


def setup_debug_directory():
    """Create debug directory structure."""
    DEBUG_DIR.mkdir(exist_ok=True)
    print(f"Created debug directory: {DEBUG_DIR.absolute()}")


def process_single_mmwave_file(mmwave_file):
    """
    Process a single mmWave file: generate beam labels and save debug dataset.
    
    Args:
        mmwave_file: Path to mmWave file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"\nProcessing {mmwave_file.name}")
        
        # Load mmWave data
        print(f"  Loading mmWave data...")
        mmwave_data = np.load(mmwave_file, allow_pickle=True)
        
        # Check if mmWave data has required fields
        required_keys = ['channels', 'ch_params_info']
        missing_keys = [key for key in required_keys if key not in mmwave_data]
        if missing_keys:
            print(f"  Error: Missing required keys: {missing_keys}")
            return False
        
        print(f"  Channels shape: {mmwave_data['channels'].shape}")
        print(f"  BS antenna shape: {mmwave_data['ch_params_info'].item()['bs_antenna_shape']}")
        
        # Generate beam labels for all sizes
        print(f"  Generating beam labels for beam sizes {BEAM_SIZES} with FOV={FOV}°...")
        beam_labels = calculate_best_beams(mmwave_data, BEAM_SIZES, FOV)
        
        print(f"  Beam labels shape: {beam_labels.shape}")
        
        # Create debug filename
        debug_filename = f"debug_{mmwave_file.stem}_beam_labels.npz"
        debug_file = DEBUG_DIR / debug_filename
        
        # Prepare debug data
        debug_data = {
            'original_file': str(mmwave_file),
            'beam_sizes': BEAM_SIZES,
            'fov_degrees': FOV,
            'beam_labels': beam_labels,
            'channels_shape': mmwave_data['channels'].shape,
            'bs_antenna_shape': mmwave_data['ch_params_info'].item()['bs_antenna_shape'],
            'calculation_method': 'exact_replica_of_add_beam_labels'
        }
        
        # Save debug dataset
        print(f"  Saving debug dataset to {debug_file.name}")
        np.savez_compressed(debug_file, **debug_data)
        
        # Close the loaded data
        mmwave_data.close()
        
        print(f"  ✓ Created debug dataset: {debug_file.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {mmwave_file.name}: {str(e)}")
        return False


def main():
    """Main function to create debug beam label dataset."""
    print("Debug Beam Labels - Replicating add_beam_labels.py")
    print("=" * 60)
    print(f"This will:")
    print(f"  1. Load mmWave test channels from {MMWAVE_TEST_DIR}")
    print(f"  2. Generate beam labels for sizes {BEAM_SIZES} with FOV={FOV}°")
    print(f"  3. Save debug dataset in {DEBUG_DIR}")
    print(f"  4. Replicate EXACTLY what add_beam_labels.py does")
    
    # Setup
    setup_debug_directory()
    
    # Check source directory exists
    if not MMWAVE_TEST_DIR.exists():
        print(f"Error: mmWave test directory not found: {MMWAVE_TEST_DIR}")
        return
    
    # Get all mmWave test files
    mmwave_files = list(MMWAVE_TEST_DIR.glob("*.npz"))
    
    if not mmwave_files:
        print("No mmWave test files found. Exiting.")
        return
    
    print(f"\nFound {len(mmwave_files)} mmWave test files")
    
    # Process each file
    print(f"\nProcessing {len(mmwave_files)} mmWave files...")
    
    successful = 0
    failed = 0
    
    for mmwave_file in tqdm(mmwave_files, desc="Processing mmWave files"):
        if process_single_mmwave_file(mmwave_file):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n" + "=" * 60)
    print("DEBUG BEAM LABELS COMPLETE")
    print("=" * 60)
    print(f"Total mmWave files: {len(mmwave_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Debug datasets saved in: {DEBUG_DIR.absolute()}")
    
    if failed > 0:
        print(f"\n⚠️  {failed} files failed to process.")
    else:
        print(f"\n🎉 All debug datasets created successfully!")
    
    print(f"\nNow you can examine the debug datasets to see what's happening with the beam labels.")


if __name__ == "__main__":
    main() 