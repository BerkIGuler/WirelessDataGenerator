#!/usr/bin/env python3
"""
Add visualization labels (beam sizes 4 and 8) to cross-freq test files.

This script:
1. Loads mmWave files from processed_channels/mmwave/test
2. Generates beam labels with sizes 4 and 8
3. Adds them as 'visualization_labels' to cross-freq test files
4. Preserves all existing data
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
CROSS_FREQ_TEST_DIR = Path("datasets/cross_freq_different_env_beam_prediction/test")

# Beam sizes for visualization labels
VISUALIZATION_BEAM_SIZES = [4, 8]

# Field of view for beam calculation (same as add_beam_labels.py)
FOV = 180


# =============================================================================
# Utility Functions
# =============================================================================

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
        
        # Debug: print beam angles for small beam sizes
        if beam_size <= 8:
            print(f"      Beam size {beam_size}: start={start}, end={end}, angles={beam_angles}")
        
        F1 = np.array([dm.steering_vec(bs_antenna_shape, phi=azi).squeeze()
               for azi in beam_angles])
        mean_amplitude = np.abs(F1 @ dataset['channels']).mean(axis=-1).squeeze(1)
        mean_amplitude_dbm = np.around(to_dbm(mean_amplitude), 3)

        best_beams = np.argmax(mean_amplitude_dbm, axis=1)
        beam_labels[:, beam_idx] = best_beams
    
    return beam_labels


def get_matching_files():
    """
    Find matching files between mmWave test and cross-freq test directories.
    
    Returns:
        list: List of tuples (mmwave_file, cross_freq_file, city_name)
    """
    matching_files = []
    
    # Get all mmWave test files
    mmwave_files = list(MMWAVE_TEST_DIR.glob("*.npz"))
    
    for mmwave_file in mmwave_files:
        # Extract city name and base station from mmWave filename
        # Example: city_3_houston_28_bs000.npz
        filename = mmwave_file.name
        parts = filename.split('_')
        
        if len(parts) >= 4 and parts[0] == 'city':
            city_id = parts[1]
            city_name = parts[2]
            bs_part = parts[4]  # bs000, bs001, etc.
            
            # Construct corresponding cross-freq filename
            # Example: city_3_houston_3p5_bs000.npz
            cross_freq_filename = f"city_{city_id}_{city_name}_3p5_{bs_part}"
            cross_freq_file = CROSS_FREQ_TEST_DIR / cross_freq_filename
            
            if cross_freq_file.exists():
                matching_files.append((mmwave_file, cross_freq_file, city_name))
                print(f"  ✓ {filename} <-> {cross_freq_filename}")
            else:
                print(f"  ✗ {filename} -> {cross_freq_filename} (not found)")
    
    return matching_files


def process_single_pair(mmwave_file, cross_freq_file, city_name):
    """
    Process a single file pair: generate visualization labels and update cross-freq file.
    
    Args:
        mmwave_file: Path to mmWave file
        cross_freq_file: Path to cross-freq file
        city_name: Name of the city
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"  Processing {mmwave_file.name} -> {cross_freq_file.name}")
        
        # Load mmWave data to generate visualization labels
        print(f"    Loading mmWave data...")
        mmwave_data = np.load(mmwave_file, allow_pickle=True)
        
        # Check if mmWave data has required fields
        required_keys = ['channels', 'ch_params_info']
        missing_keys = [key for key in required_keys if key not in mmwave_data]
        if missing_keys:
            print(f"    Error: Missing required keys in mmWave file: {missing_keys}")
            return False
        
        # Generate visualization labels (beam sizes 4 and 8)
        print(f"    Generating visualization labels for beam sizes {VISUALIZATION_BEAM_SIZES}...")
        visualization_labels = calculate_best_beams(mmwave_data, VISUALIZATION_BEAM_SIZES, FOV)
        
        print(f"    Visualization labels shape: {visualization_labels.shape}")
        
        # Load cross-freq data
        print(f"    Loading cross-freq data...")
        cross_freq_data = np.load(cross_freq_file, allow_pickle=True)
        
        # Convert to dict for easier manipulation
        cross_freq_dict = dict(cross_freq_data)
        
        # Check if visualization_labels already exist
        if 'visualization_labels' in cross_freq_dict:
            print(f"    Warning: {cross_freq_file.name} already has visualization_labels, skipping...")
            return True
        
        # Add visualization labels to cross-freq data
        cross_freq_dict['visualization_labels'] = visualization_labels
        
        # Close the loaded data
        mmwave_data.close()
        cross_freq_data.close()
        
        # Save updated cross-freq file
        print(f"    Saving updated cross-freq file...")
        np.savez_compressed(cross_freq_file, **cross_freq_dict)
        
        print(f"    ✓ Added visualization labels to {cross_freq_file.name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error processing {mmwave_file.name}: {str(e)}")
        return False


def main():
    """Main function to add visualization labels to cross-freq test files."""
    print("Adding Visualization Labels to Cross-Freq Test Files")
    print("=" * 60)
    print(f"This will:")
    print(f"  1. Load mmWave files from {MMWAVE_TEST_DIR}")
    print(f"  2. Generate beam labels with sizes {VISUALIZATION_BEAM_SIZES}")
    print(f"  3. Add them as 'visualization_labels' to cross-freq test files")
    print(f"  4. Preserve all existing data")
    
    # Check directories exist
    if not MMWAVE_TEST_DIR.exists():
        print(f"Error: mmWave test directory not found: {MMWAVE_TEST_DIR}")
        return
    
    if not CROSS_FREQ_TEST_DIR.exists():
        print(f"Error: Cross-freq test directory not found: {CROSS_FREQ_TEST_DIR}")
        return
    
    # Find matching files
    print(f"\nFinding matching files between mmWave and cross-freq test directories...")
    matching_files = get_matching_files()
    
    if not matching_files:
        print("No matching files found. Exiting.")
        return
    
    print(f"Found {len(matching_files)} matching file pairs")
    
    # Process each pair
    print(f"\nProcessing {len(matching_files)} file pairs...")
    
    successful = 0
    failed = 0
    
    for mmwave_file, cross_freq_file, city_name in tqdm(matching_files, desc="Processing file pairs"):
        if process_single_pair(mmwave_file, cross_freq_file, city_name):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n" + "=" * 60)
    print("VISUALIZATION LABELS ADDITION COMPLETE")
    print("=" * 60)
    print(f"Total file pairs: {len(matching_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {100*successful/len(matching_files):.1f}%")
    
    if failed > 0:
        print(f"\n⚠️  {failed} files failed to process. Check the error messages above.")
    else:
        print(f"\n🎉 All files processed successfully!")
    
    print(f"\nVisualization labels with beam sizes {VISUALIZATION_BEAM_SIZES} have been added to cross-freq test files.")
    print(f"The 'visualization_labels' key now contains beam indices for 4 and 8 beam directions.")


if __name__ == "__main__":
    main() 