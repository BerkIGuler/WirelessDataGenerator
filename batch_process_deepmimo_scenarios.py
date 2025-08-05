"""
Batch Processing Script for DeepMIMO Scenarios

This script processes downloaded DeepMIMO scenarios and saves the channel data
for each base station as individual .npz files with metadata.

Processes:
- All city scenarios (excluding LWM training scenarios)
- Additional specific scenarios: asu_campus_3p5, boston5g_3p5
- Automatically detects and processes appropriate dataset types

Features:
- Uses same channel parameters as load_sample_deepmimo_scenario.py
- Applies filtering (remove inactive users) and scaling (10^6)
- Saves as <scenario_name>_bs<index>.npz
- Includes LOS, user positions, BS positions, and channels
- Skips existing files to avoid reprocessing
- Memory-efficient processing (one scenario at a time)
- Progress tracking with tqdm
- Handles both MacroDataset (multiple BS) and Dataset (single BS) types
- Comprehensive error reporting and continues on failures
"""

import deepmimo as dm
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import gc
import sys


# =============================================================================
# Configuration Parameters
# =============================================================================

# Directories
SCENARIOS_DIR = Path("./deepmimo_scenarios")
OUTPUT_DIR = Path("./processed_channels")

# Wireless communication parameters (same as load_sample_deepmimo_scenario.py)
SUBCARRIER_SPACING = 30_000  # Hz - Standard 5G NR subcarrier spacing
SCALING_FACTOR = 1e6  # Channel scaling factor


# =============================================================================
# Channel Parameter Setup
# =============================================================================

def setup_channel_parameters():
    """
    Configure DeepMIMO channel parameters (same as load_sample_deepmimo_scenario.py).
    
    Returns:
        dm.ChannelParameters: Configured channel parameters object
    """
    ch_params = dm.ChannelParameters()
    
    # OFDM Configuration
    ch_params.ofdm.subcarriers = 32  # Number of OFDM subcarriers
    ch_params.ofdm.bandwidth = SUBCARRIER_SPACING * ch_params.ofdm.subcarriers
    ch_params.ofdm.selected_subcarriers = np.arange(0, ch_params.ofdm.subcarriers)
    
    # Base Station (BS) Antenna Configuration
    ch_params.bs_antenna.shape = np.array([32, 1])  # 32-element linear array
    ch_params.bs_antenna.rotation = np.array([0, 0, -135])  # Antenna rotation (degrees)
    
    # Multipath Configuration
    ch_params.num_paths = 20  # Maximum number of multipath components
    
    return ch_params


# =============================================================================
# Utility Functions
# =============================================================================

def setup_output_directory():
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.absolute()}")


def get_scenario_list():
    """
    Get list of scenarios to process, including city scenarios plus specific additional scenarios.
    
    Returns:
        list: List of scenario names (directory names)
    """
    if not SCENARIOS_DIR.exists():
        print(f"Error: Scenarios directory not found: {SCENARIOS_DIR}")
        return []
    
    # Get all available scenarios
    all_scenarios = [d.name for d in SCENARIOS_DIR.iterdir() if d.is_dir()]
    
    # Filter for scenarios we want to process
    target_scenarios = []
    
    # Add all city scenarios (excluding lwm training scenarios)
    city_scenarios = [s for s in all_scenarios if "city" in s and not s.endswith("lwm")]
    target_scenarios.extend(city_scenarios)
    
    # Add specific additional scenarios
    additional_scenarios = ["asu_campus_3p5", "boston5g_3p5"]
    for scenario in additional_scenarios:
        # Check for exact match or scenarios containing these names
        matching_scenarios = [s for s in all_scenarios if scenario in s.lower()]
        target_scenarios.extend(matching_scenarios)
    
    # Remove duplicates and sort
    target_scenarios = sorted(list(set(target_scenarios)))
    
    print(f"Found {len(all_scenarios)} total scenarios in {SCENARIOS_DIR}")
    print(f"Target scenarios for processing: {len(target_scenarios)}")
    
    # Show breakdown
    city_count = len([s for s in target_scenarios if "city" in s])
    additional_count = len(target_scenarios) - city_count
    
    print(f"  - City scenarios: {city_count}")
    print(f"  - Additional scenarios (asu_campus, boston): {additional_count}")
    
    if additional_count > 0:
        additional_found = [s for s in target_scenarios if not "city" in s]
        print(f"  - Additional scenarios found: {additional_found}")
    
    return target_scenarios


def get_output_filename(scenario_name, bs_index):
    """Generate output filename for a scenario and base station."""
    return OUTPUT_DIR / f"{scenario_name}_bs{bs_index:03d}.npz"


def file_exists(scenario_name, bs_index):
    """Check if output file already exists."""
    return get_output_filename(scenario_name, bs_index).exists()


# =============================================================================
# Data Processing Functions
# =============================================================================

def process_channel_data(channels_raw, los_data):
    """
    Process channel data: filter inactive users and apply scaling.
    
    Args:
        channels_raw: Raw channel data
        los_data: Line-of-sight data for filtering
        
    Returns:
        tuple: (processed_channels, active_mask) or (None, None) if no active users
    """
    # Filter for active users (where LOS data is not -1, meaning paths exist)
    active_mask = los_data != -1
    
    if np.sum(active_mask) == 0:
        print("    Warning: No active users found, skipping...")
        return None, None
    
    # Apply filtering and scaling
    channels_filtered = channels_raw[active_mask]
    channels_scaled = channels_filtered * SCALING_FACTOR
    
    return channels_scaled, active_mask


def save_bs_data(scenario_name, bs_index, dataset, active_mask):
    """
    Save base station data to .npz file.
    
    Args:
        scenario_name: Name of the scenario
        bs_index: Base station index
        dataset: Base station dataset
        active_mask: Boolean mask for active users
    """
    output_file = get_output_filename(scenario_name, bs_index)
    
    # Prepare data to save
    save_data = {
        # Channel data (key requirement)
        'channels': dataset.channel[active_mask] * SCALING_FACTOR,
        
        # Position information
        'rx_pos': dataset.rx_pos[active_mask],  # Only active user positions
        'tx_pos': dataset.tx_pos,  # Base station position
        
        # LOS information
        'los': dataset.los[active_mask],  # Only active user LOS data
        
        # Metadata
        'scenario_name': scenario_name,
        'bs_index': bs_index,
        'scaling_factor': SCALING_FACTOR,
        'active_mask_original_indices': np.where(active_mask)[0],  # Original indices of active users
        'total_users_original': len(active_mask),
        'active_users_count': np.sum(active_mask),
    }
    
    # Add channel parameters for reference
    if hasattr(dataset, 'ch_params'):
        save_data['ch_params_info'] = {
            'subcarriers': dataset.ch_params.ofdm.subcarriers,
            'bandwidth': dataset.ch_params.ofdm.bandwidth,
            'num_paths': dataset.ch_params.num_paths,
            'bs_antenna_shape': dataset.ch_params.bs_antenna.shape,
            'bs_antenna_rotation': dataset.ch_params.bs_antenna.rotation,
        }
    
    # Save to .npz file
    np.savez_compressed(output_file, **save_data)
    
    return output_file


def process_single_scenario(scenario_name, ch_params):
    """
    Process a single scenario and save all base station data.
    Handles both MacroDataset (multiple BS) and Dataset (single BS) types.
    
    Args:
        scenario_name: Name of the scenario to process
        ch_params: Channel parameters to use
        
    Returns:
        tuple: (success, num_bs_processed, error_message)
    """
    try:
        print(f"\nProcessing scenario: {scenario_name}")
        
        # Load scenario first (lightweight operation)
        print("  Loading dataset...")
        dataset = dm.load(scenario_name)
        
        # Determine dataset type and handle accordingly
        dataset_type = type(dataset).__name__
        print(f"  Dataset type: {dataset_type}")
        
        if dataset_type == "MacroDataset":
            # Multiple base stations - can be indexed as dataset[0], dataset[1], etc.
            num_bs = len(dataset)
            print(f"  Found {num_bs} base stations (MacroDataset)")
            
            # Check which base station files already exist
            existing_files = []
            missing_files = []
            
            for bs_idx in range(num_bs):
                output_file = get_output_filename(scenario_name, bs_idx)
                if output_file.exists():
                    existing_files.append(bs_idx)
                else:
                    missing_files.append(bs_idx)
            
            print(f"  Existing files: {len(existing_files)}/{num_bs}")
            print(f"  Missing files: {len(missing_files)}/{num_bs}")
            
            # If all files exist, skip channel computation entirely
            if len(missing_files) == 0:
                print("  All base station files already exist - skipping channel computation")
                return True, num_bs, None
            
            # Only compute channels if some files are missing (expensive operation)
            print("  Computing channels for missing files...")
            dataset.compute_channels(ch_params)
            
            # Process each missing base station
            bs_processed = len(existing_files)  # Count existing files as processed
            
            for bs_idx in missing_files:
                print(f"    Processing BS {bs_idx}...")
                
                # Get base station dataset
                bs_dataset = dataset[bs_idx]
                
                # Process channel data
                active_mask = bs_dataset.los != -1
                
                if np.sum(active_mask) == 0:
                    print(f"    BS {bs_idx}: No active users, skipping...")
                    continue
                
                # Save base station data
                saved_file = save_bs_data(scenario_name, bs_idx, bs_dataset, active_mask)
                
                active_count = np.sum(active_mask)
                total_count = len(active_mask)
                print(f"    BS {bs_idx}: Saved {active_count}/{total_count} users to {saved_file.name}")
                
                bs_processed += 1
            
            # Report on existing files that were skipped
            for bs_idx in existing_files:
                print(f"    BS {bs_idx}: Skipped (file exists)")
            
            return True, bs_processed, None
            
        elif dataset_type == "Dataset":
            # Single base station - dataset is already the base station data
            print(f"  Single base station dataset (Dataset)")
            
            # For single BS datasets, we treat it as BS index 0
            bs_idx = 0
            output_file = get_output_filename(scenario_name, bs_idx)
            
            # Check if file already exists
            if output_file.exists():
                print("  Base station file already exists - skipping channel computation")
                return True, 1, None
            
            # Compute channels (expensive operation)
            print("  Computing channels...")
            dataset.compute_channels(ch_params)
            
            # Process the single base station (dataset itself)
            print(f"    Processing BS {bs_idx}...")
            
            # Process channel data
            active_mask = dataset.los != -1
            
            if np.sum(active_mask) == 0:
                print(f"    BS {bs_idx}: No active users, skipping...")
                return True, 0, None
            
            # Save base station data
            saved_file = save_bs_data(scenario_name, bs_idx, dataset, active_mask)
            
            active_count = np.sum(active_mask)
            total_count = len(active_mask)
            print(f"    BS {bs_idx}: Saved {active_count}/{total_count} users to {saved_file.name}")
            
            return True, 1, None
            
        else:
            # Unknown dataset type
            error_msg = f"Unknown dataset type: {dataset_type}"
            print(f"  {error_msg}")
            return False, 0, error_msg
        
    except Exception as e:
        error_msg = f"Error processing {scenario_name}: {str(e)}"
        print(f"  {error_msg}")
        return False, 0, error_msg
    
    finally:
        # Free memory
        try:
            del dataset
            gc.collect()
        except:
            pass


def check_processed_scenarios(scenarios):
    """
    Check which scenarios are already completely processed.
    
    Args:
        scenarios (list): List of scenario names to check
        
    Returns:
        dict: Information about processed vs unprocessed scenarios
    """
    if not OUTPUT_DIR.exists():
        return {
            'fully_processed': [],
            'partially_processed': [],
            'unprocessed': scenarios,
            'total_files_found': 0
        }
    
    fully_processed = []
    partially_processed = []
    unprocessed = []
    total_files_found = 0
    
    print("Checking for already processed scenarios...")
    
    for scenario in scenarios:
        # Find all files for this scenario
        scenario_files = list(OUTPUT_DIR.glob(f"{scenario}_bs*.npz"))
        total_files_found += len(scenario_files)
        
        if len(scenario_files) == 0:
            unprocessed.append(scenario)
        else:
            # For now, we'll consider any scenario with files as at least partially processed
            # We could enhance this later to check if all expected base stations are present
            partially_processed.append((scenario, len(scenario_files)))
    
    return {
        'fully_processed': fully_processed,
        'partially_processed': partially_processed,
        'unprocessed': unprocessed,
        'total_files_found': total_files_found
    }


def display_processing_status(scenarios, status_info):
    """
    Display the current processing status of scenarios.
    
    Args:
        scenarios (list): List of all target scenarios
        status_info (dict): Processing status information
    """
    print("\n" + "=" * 60)
    print("CURRENT PROCESSING STATUS")
    print("=" * 60)
    
    total_scenarios = len(scenarios)
    unprocessed_count = len(status_info['unprocessed'])
    partially_processed_count = len(status_info['partially_processed'])
    
    print(f"📊 Total target scenarios: {total_scenarios}")
    print(f"✅ Scenarios with processed files: {partially_processed_count}")
    print(f"⏳ Scenarios not yet processed: {unprocessed_count}")
    print(f"📁 Total processed files found: {status_info['total_files_found']}")
    
    if status_info['partially_processed']:
        print(f"\n✅ Scenarios with existing processed files:")
        for scenario, file_count in status_info['partially_processed']:
            print(f"   - {scenario}: {file_count} files")
    
    if status_info['unprocessed']:
        print(f"\n⏳ Scenarios to be processed:")
        for i, scenario in enumerate(status_info['unprocessed'][:10], 1):
            print(f"   {i:2d}. {scenario}")
        
        if len(status_info['unprocessed']) > 10:
            print(f"   ... and {len(status_info['unprocessed']) - 10} more")
    
    print(f"\n💡 Note: Individual base station files will be checked during processing")
    print(f"💡 Existing files will be skipped to avoid reprocessing")


# =============================================================================
# Main Processing Function
# =============================================================================

def main():
    """Main function to batch process all scenarios."""
    print("DeepMIMO Scenario Batch Processing")
    print("=" * 60)
    
    # Setup
    setup_output_directory()
    scenarios = get_scenario_list()
    
    if not scenarios:
        print("No scenarios found. Exiting.")
        return
    
    # Setup channel parameters
    print("\nConfiguring channel parameters...")
    ch_params = setup_channel_parameters()
    print("Channel parameters configured.")
    
    # Check processing status
    status_info = check_processed_scenarios(scenarios)
    display_processing_status(scenarios, status_info)
    
    # Process scenarios
    print(f"\nProcessing {len(scenarios)} scenarios...")
    
    total_scenarios_processed = 0
    total_bs_processed = 0
    failed_scenarios = []
    successful_scenarios = []
    
    # Progress bar for scenarios
    for scenario_name in tqdm(scenarios, desc="Processing scenarios"):
        success, bs_count, error = process_single_scenario(scenario_name, ch_params)
        
        if success:
            total_scenarios_processed += 1
            total_bs_processed += bs_count
            successful_scenarios.append((scenario_name, bs_count))
        else:
            failed_scenarios.append((scenario_name, error))
            print(f"\nWarning: Failed to process {scenario_name}")
            print(f"Error: {error}")
            print("Continuing with next scenario...")
    
    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    
    # Overall statistics
    print(f"Total scenarios found: {len(scenarios)}")
    print(f"Scenarios processed successfully: {total_scenarios_processed}")
    print(f"Scenarios failed: {len(failed_scenarios)}")
    print(f"Success rate: {100*total_scenarios_processed/len(scenarios):.1f}%")
    print(f"Total base stations processed: {total_bs_processed}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    
    # Detailed success report
    if successful_scenarios:
        print(f"\n" + "=" * 50)
        print(f"SUCCESSFUL SCENARIOS ({len(successful_scenarios)})")
        print("=" * 50)
        for scenario, bs_count in successful_scenarios:
            print(f"  ✓ {scenario}: {bs_count} base stations processed")
    
    # Detailed failure report
    if failed_scenarios:
        print(f"\n" + "=" * 50)
        print(f"FAILED SCENARIOS ({len(failed_scenarios)})")
        print("=" * 50)
        
        # Group failures by error type for better analysis
        error_types = {}
        for scenario, error in failed_scenarios:
            # Extract error type (first part of error message)
            error_type = error.split(':')[0] if ':' in error else 'Unknown error'
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append((scenario, error))
        
        # Report by error type
        for error_type, scenario_errors in error_types.items():
            print(f"\n  {error_type} ({len(scenario_errors)} scenarios):")
            for scenario, full_error in scenario_errors:
                print(f"    ✗ {scenario}")
                print(f"      Error: {full_error}")
        
        # Summary of error types
        print(f"\n  Error Type Summary:")
        for error_type, scenario_errors in error_types.items():
            percentage = 100 * len(scenario_errors) / len(failed_scenarios)
            print(f"    - {error_type}: {len(scenario_errors)} scenarios ({percentage:.1f}%)")
    
    # Final status
    if len(failed_scenarios) == 0:
        print(f"\n🎉 All scenarios processed successfully!")
    elif total_scenarios_processed > 0:
        print(f"\n⚠️  Processing completed with {len(failed_scenarios)} failures out of {len(scenarios)} scenarios.")
        print(f"   {total_scenarios_processed} scenarios were processed successfully.")
    else:
        print(f"\n❌ All scenarios failed to process. Please check the errors above.")
    
    print("\nProcessing complete!")
    
    # Return summary for potential scripting use
    return {
        'total_scenarios': len(scenarios),
        'successful': total_scenarios_processed,
        'failed': len(failed_scenarios),
        'total_bs_processed': total_bs_processed,
        'failed_scenarios': failed_scenarios,
        'successful_scenarios': successful_scenarios
    }


if __name__ == "__main__":
    main() 