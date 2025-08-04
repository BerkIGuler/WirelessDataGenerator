"""
DeepMIMO Dataset Exploration Script

mostly taken from the DeepMIMO documentation:
https://www.deepmimo.net/docs/manual_full.html

This script demonstrates how to:
1. Configure channel parameters for wireless communication simulation
2. Load a DeepMIMO scenario dataset
3. Compute wireless channels with specified parameters
4. Explore and analyze the resulting dataset structure
5. Visualize channel characteristics through coverage maps
6. Analyze and plot active vs inactive receiver positions

Compatible with ContraWiMAE/WiMAE dataset requirements.
"""

import deepmimo as dm
import pprint as pp
import numpy as np
import textwrap
import matplotlib.pyplot as plt
import os
from pathlib import Path


# =============================================================================
# Configuration Parameters
# =============================================================================

# Wireless communication parameters
SUBCARRIER_SPACING = 30_000  # Hz - Standard 5G NR subcarrier spacing
SCENARIO_NAME = "city_0_newyork_3p5"  # New York City scenario at 3.5 GHz

# Output directory for saving plots
PLOTS_DIR = Path("plots")

# Pretty printer for formatted output
printer = pp.PrettyPrinter(indent=2, width=100)


# =============================================================================
# Utility Functions
# =============================================================================

def setup_plots_directory():
    """Create plots directory if it doesn't exist."""
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"Plots will be saved to: {PLOTS_DIR.absolute()}")


# =============================================================================
# Channel Parameter Configuration
# =============================================================================

def setup_channel_parameters():
    """
    Configure DeepMIMO channel parameters for OFDM communication system.
    
    Returns:
        dm.ChannelParameters: Configured channel parameters object
    """
    print("=" * 60)
    print("CONFIGURING CHANNEL PARAMETERS")
    print("=" * 60)
    
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
    
    print("Channel parameters configured:")
    printer.pprint(ch_params)
    print()
    
    return ch_params


# =============================================================================
# Dataset Loading and Channel Computation
# =============================================================================

def load_and_compute_dataset(scenario_name, ch_params):
    """
    Load DeepMIMO scenario and compute wireless channels.
    
    Args:
        scenario_name (str): Name of the scenario to load
        ch_params (dm.ChannelParameters): Channel parameters for computation
        
    Returns:
        tuple: (dataset, first_bs_dataset) - Full dataset and first BS data
    """
    print("=" * 60)
    print(f"LOADING SCENARIO: {scenario_name}")
    print("=" * 60)
    
    # Display scenario summary
    dm.summary(scenario_name)
    print()
    
    # Load the scenario dataset
    print("Loading dataset...")
    dataset = dm.load(scenario_name)
    
    # Compute channels with specified parameters
    print("Computing wireless channels...")
    dataset.compute_channels(ch_params)
    
    print(f"Dataset loaded successfully!")
    print(f"Number of Base Stations (BS): {len(dataset)}")
    print()
    
    # Extract first BS dataset for detailed analysis
    first_bs_dataset = dataset[0]
    
    return dataset, first_bs_dataset


# =============================================================================
# Dataset Structure Analysis
# =============================================================================

def analyze_dataset_structure(dataset):
    """
    Analyze and categorize the structure of the dataset.
    
    Args:
        dataset: DeepMIMO dataset object to analyze
    """
    print("=" * 60)
    print("DATASET STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Get all non-private keys (excluding those starting with '_')
    all_keys = [key for key in dataset.keys() if not key.startswith('_')]
    
    # Categorize keys by data type
    array_keys = [key for key in all_keys if isinstance(dataset[key], np.ndarray)]
    dict_keys = [key for key in all_keys if hasattr(dataset[key], 'keys')]
    other_keys = [key for key in all_keys if key not in array_keys + dict_keys]
    
    print(f"Total data fields: {len(all_keys)}")
    print(f"  - NumPy arrays: {len(array_keys)}")
    print(f"  - Dictionaries: {len(dict_keys)}")
    print(f"  - Other types: {len(other_keys)}")
    print()
    
    return array_keys, dict_keys, other_keys


def display_numpy_arrays(dataset, array_keys):
    """Display information about NumPy arrays in the dataset."""
    print("=" * 60)
    print("NUMPY ARRAYS - Channel and Propagation Data")
    print("=" * 60)
    
    # Group arrays by category for better organization
    angle_arrays = [k for k in array_keys if any(x in k for x in ['aoa', 'aod'])]
    channel_arrays = [k for k in array_keys if any(x in k for x in ['channel', 'power', 'phase', 'delay'])]
    position_arrays = [k for k in array_keys if 'pos' in k]
    other_arrays = [k for k in array_keys if k not in angle_arrays + channel_arrays + position_arrays]
    
    categories = [
        ("Angle Information", angle_arrays),
        ("Channel Characteristics", channel_arrays),
        ("Position Data", position_arrays),
        ("Other Arrays", other_arrays)
    ]
    
    for category_name, keys in categories:
        if keys:
            print(f"\n{category_name}:")
            print("-" * 40)
            for key in keys:
                shape_info = f"dataset_0.{key}.shape: {dataset[key].shape}"
                print(f"  {shape_info}")


def display_dictionaries(dataset, dict_keys):
    """Display information about dictionary structures in the dataset."""
    print("\n" + "=" * 60)
    print("CONFIGURATION DICTIONARIES")
    print("=" * 60)
    
    for key in dict_keys:
        dict_subkeys = list(dataset[key].keys())
        print(f"\ndataset_0.{key}: DotDict with {len(dict_subkeys)} keys:")
        
        # Format the keys nicely
        formatted_keys = textwrap.fill(
            str(dict_subkeys), 
            width=80, 
            initial_indent="    ",
            subsequent_indent="    "
        )
        print(formatted_keys)


def display_other_types(dataset, other_keys):
    """Display information about other data types in the dataset."""
    print("\n" + "=" * 60)
    print("METADATA AND OBJECTS")
    print("=" * 60)
    
    for key in other_keys:
        value = dataset[key]
        print(f"\ndataset_0.{key}:")
        print(f"  Type: {type(value).__name__}")
        print(f"  Value: {value}")


# =============================================================================
# Dataset Statistics and Summary
# =============================================================================

def display_dataset_statistics(dataset):
    """Display key statistics about the loaded dataset."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    try:
        print(f"Pathloss data shape: {dataset.pathloss.shape}")
        print(f"Maximum paths per location: {dataset.num_paths.max()}")
        print(f"Line-of-sight (LOS) data shape: {dataset.los.shape}")
        print(f"Configured number of paths: {dataset.ch_params.num_paths}")
        print(f"Number of UE locations: {dataset.n_ue}")
    except AttributeError as e:
        print(f"Some statistics unavailable: {e}")


def analyze_channel_statistics(dataset):
    """
    Calculate and display statistical properties of the complex channel data.
    Filters out inactive users (no LOS or NLOS paths available).
    
    Args:
        dataset: DeepMIMO dataset object containing channel information
    """
    print("\n" + "=" * 60)
    print("COMPLEX CHANNEL STATISTICS ANALYSIS")
    print("=" * 60)
    
    try:
        # Get channel data and filter out inactive users
        channels_raw = dataset.channel
        los_data = dataset.los
        
        # Filter for active users (where LOS data is not -1, meaning paths exist)
        active_mask = los_data != -1
        channels_raw_filtered = channels_raw[active_mask]
        
        # Display filtering information
        total_users = channels_raw.shape[0]
        active_users = np.sum(active_mask)
        filtered_users = total_users - active_users
        
        print(f"Total users: {total_users}")
        print(f"Active users (with paths): {active_users} ({100*active_users/total_users:.1f}%)")
        print(f"Filtered out users (no paths): {filtered_users} ({100*filtered_users/total_users:.1f}%)")
        
        if active_users == 0:
            print("No active users found - cannot calculate channel statistics!")
            return None
        
        # Apply scaling factor to filtered data
        scaling_factor = 1e6
        channels = channels_raw_filtered * scaling_factor
        
        print(f"\nChannel data shape (after filtering): {channels.shape}")
        print(f"Channel data type: {channels.dtype}")
        print(f"Applied scaling factor: {scaling_factor:.0e}")
        print(f"Original range (active users): [{np.min(np.abs(channels_raw_filtered)):.2e}, {np.max(np.abs(channels_raw_filtered)):.2e}]")
        print(f"Scaled range (active users): [{np.min(np.abs(channels)):.2e}, {np.max(np.abs(channels)):.2e}]")
        
        # Separate real and imaginary parts
        real_part = np.real(channels)
        imag_part = np.imag(channels)
        
        # Calculate statistics for real part
        real_mean = np.mean(real_part)
        real_std = np.std(real_part)
        real_min = np.min(real_part)
        real_max = np.max(real_part)
        
        # Calculate statistics for imaginary part
        imag_mean = np.mean(imag_part)
        imag_std = np.std(imag_part)
        imag_min = np.min(imag_part)
        imag_max = np.max(imag_part)
        
        # Calculate magnitude and phase statistics
        magnitude = np.abs(channels)
        phase = np.angle(channels)
        
        mag_mean = np.mean(magnitude)
        mag_std = np.std(magnitude)
        phase_mean = np.mean(phase)
        phase_std = np.std(phase)
        
        # Display results
        print("\n" + "=" * 40)
        print("REAL PART STATISTICS (SCALED, ACTIVE USERS)")
        print("=" * 40)
        print(f"Mean:              {real_mean:.6f}")
        print(f"Standard deviation: {real_std:.6f}")
        print(f"Minimum:           {real_min:.6f}")
        print(f"Maximum:           {real_max:.6f}")
        
        print("\n" + "=" * 40)
        print("IMAGINARY PART STATISTICS (SCALED, ACTIVE USERS)")
        print("=" * 40)
        print(f"Mean:              {imag_mean:.6f}")
        print(f"Standard deviation: {imag_std:.6f}")
        print(f"Minimum:           {imag_min:.6f}")
        print(f"Maximum:           {imag_max:.6f}")
        
        print("\n" + "=" * 40)
        print("MAGNITUDE AND PHASE STATISTICS (SCALED, ACTIVE USERS)")
        print("=" * 40)
        print(f"Magnitude mean:    {mag_mean:.6f}")
        print(f"Magnitude std:     {mag_std:.6f}")
        print(f"Phase mean (rad):  {phase_mean:.6f}")
        print(f"Phase std (rad):   {phase_std:.6f}")
        print(f"Phase mean (deg):  {np.degrees(phase_mean):.2f}°")
        print(f"Phase std (deg):   {np.degrees(phase_std):.2f}°")
        
        # Additional analysis per subcarrier if relevant
        if channels.shape[2] > 1:  # Multiple subcarriers
            print("\n" + "=" * 40)
            print("PER-SUBCARRIER STATISTICS (SCALED, ACTIVE USERS)")
            print("=" * 40)
            
            for sc in range(min(5, channels.shape[2])):  # Show first 5 subcarriers
                sc_data = channels[:, :, sc, :]
                sc_real_mean = np.mean(np.real(sc_data))
                sc_real_std = np.std(np.real(sc_data))
                sc_imag_mean = np.mean(np.imag(sc_data))
                sc_imag_std = np.std(np.imag(sc_data))
                
                print(f"Subcarrier {sc}:")
                print(f"  Real: μ={sc_real_mean:.6f}, σ={sc_real_std:.6f}")
                print(f"  Imag: μ={sc_imag_mean:.6f}, σ={sc_imag_std:.6f}")
            
            if channels.shape[2] > 5:
                print(f"  ... (showing first 5 of {channels.shape[2]} subcarriers)")
        
        # Calculate power statistics (squared magnitude)
        power = np.abs(channels)**2
        power_mean_db = 10 * np.log10(np.mean(power))
        power_std_db = 10 * np.log10(np.std(power))
        
        print("\n" + "=" * 40)
        print("POWER STATISTICS (SCALED, ACTIVE USERS)")
        print("=" * 40)
        print(f"Average power (linear): {np.mean(power):.6e}")
        print(f"Average power (dB):     {power_mean_db:.2f} dB")
        print(f"Power std (dB):         {power_std_db:.2f} dB")
        print(f"Note: Power scaled by factor of {scaling_factor**2:.0e}")
        
        # Return statistics for potential further use
        stats = {
            'scaling_factor': scaling_factor,
            'total_users': total_users,
            'active_users': active_users,
            'filtered_users': filtered_users,
            'real': {'mean': real_mean, 'std': real_std, 'min': real_min, 'max': real_max},
            'imag': {'mean': imag_mean, 'std': imag_std, 'min': imag_min, 'max': imag_max},
            'magnitude': {'mean': mag_mean, 'std': mag_std},
            'phase': {'mean': phase_mean, 'std': phase_std},
            'power': {'mean_linear': np.mean(power), 'mean_db': power_mean_db, 'std_db': power_std_db}
        }
        
        return stats
        
    except AttributeError as e:
        print(f"Channel data not available: {e}")
        return None
    except Exception as e:
        print(f"Error analyzing channel statistics: {e}")
        return None


# =============================================================================
# Position Activity Analysis
# =============================================================================

def analyze_position_activity(dataset):
    """
    Analyze active vs inactive receiver positions based on number of paths.
    
    Args:
        dataset: DeepMIMO dataset object to analyze
        
    Returns:
        tuple: (active_mask, inactive_mask) - Boolean masks for position activity
    """
    print("\n" + "=" * 60)
    print("ANALYZING RECEIVER POSITION ACTIVITY")
    print("=" * 60)
    
    # Create mask for positions with multipath components
    active_mask = dataset.num_paths > 0
    inactive_mask = ~active_mask
    
    # Display statistics
    num_active = np.sum(active_mask)
    num_inactive = np.sum(inactive_mask)
    total_positions = len(active_mask)
    
    print(f"Total receiver positions: {total_positions}")
    print(f"Active positions (with paths): {num_active} ({100*num_active/total_positions:.1f}%)")
    print(f"Inactive positions (no paths): {num_inactive} ({100*num_inactive/total_positions:.1f}%)")
    print()
    
    return active_mask, inactive_mask


def plot_position_activity(dataset, active_mask, inactive_mask, save_plots=True):
    """
    Create scatter plots showing active vs inactive receiver positions.
    
    Args:
        dataset: DeepMIMO dataset object
        active_mask: Boolean mask for active positions
        inactive_mask: Boolean mask for inactive positions
        save_plots: Whether to save plots to disk
    """
    print("=" * 60)
    print("PLOTTING POSITION ACTIVITY")
    print("=" * 60)
    
    # Create scatter plot showing active vs inactive positions
    plt.figure(figsize=(12, 8))
    
    # Plot inactive positions first (so active ones appear on top)
    plt.scatter(dataset.rx_pos[inactive_mask, 0], dataset.rx_pos[inactive_mask, 1],
               alpha=0.6, s=2, c='red', label=f'Inactive ({np.sum(inactive_mask)})')
    
    plt.scatter(dataset.rx_pos[active_mask, 0], dataset.rx_pos[active_mask, 1],
               alpha=0.6, s=2, c='green', label=f'Active ({np.sum(active_mask)})')
    
    # Add base station position if available
    if hasattr(dataset, 'tx_pos'):
        plt.scatter(dataset.tx_pos[0, 0], dataset.tx_pos[0, 1], 
                   s=200, c='blue', marker='^', edgecolors='black', linewidth=2,
                   label='Base Station', zorder=5)
    
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Receiver Position Activity Map\n(Active = Multipath Available, Inactive = No Paths)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_plots:
        plt.savefig(PLOTS_DIR / 'position_activity_scatter.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: position_activity_scatter.png")
    
    plt.close()
    
    # Create DeepMIMO coverage plot for LOS status
    print("Creating LoS coverage plot...")
    try:
        # Use DeepMIMO's plot_coverage function
        dataset.plot_coverage(
            dataset.los != -1,  # Convert LOS values to boolean
            title="Line-of-Sight Coverage Map",
            cbar_title="LoS Available",
            cmap=['red', 'green']
        )
        
        if save_plots:
            plt.savefig(PLOTS_DIR / 'los_coverage_map.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: los_coverage_map.png")
        
        plt.close()
        
    except Exception as e:
        print(f"  Error creating LoS coverage plot: {e}")


# =============================================================================
# Visualization and Plotting
# =============================================================================

def plot_channel_characteristics(dataset, save_plots=True):
    """
    Create coverage maps for various channel parameters.
    
    Args:
        dataset: DeepMIMO dataset object to visualize
        save_plots: Whether to save plots to disk instead of showing
    """
    print("\n" + "=" * 60)
    print("GENERATING CHANNEL VISUALIZATION PLOTS")
    print("=" * 60)
    
    # Define the main channel parameters to visualize
    main_keys = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'delay', 'power', 'phase',
                 'los', 'num_paths']
    
    # Corresponding colorbar labels for each parameter
    cbar_lbls = ['Azimuth of Arrival (°)', 'Elevation of Arrival (°)',
                 'Azimuth of Departure (°)', 'Elevation of Departure (°)',
                 'Delay (s)', 'Power (dBW)', 'Phase (°)', 
                 'Line-of-Sight Status', 'Number of Paths']
    
    print(f"Creating coverage plots for {len(main_keys)} channel parameters...")
    
    # Create coverage plots for each parameter
    for i, key in enumerate(main_keys):
        try:
            print(f"  Plotting {key}...")
            
            # Handle multidimensional arrays by taking the first path/component
            if hasattr(dataset, key):
                plt_var = dataset[key]
                
                # For 2D arrays (like angles, power, etc.), take the first path
                if plt_var.ndim == 2:
                    plt_var = plt_var[:, 0]
                
                # Create the coverage plot
                dataset.plot_coverage(
                    plt_var, 
                    title=f"{key.upper()} - {cbar_lbls[i]}", 
                    cbar_title=cbar_lbls[i]
                )
                
                if save_plots:
                    plt.savefig(PLOTS_DIR / f'coverage_{key}.png', dpi=300, bbox_inches='tight')
                    print(f"    Saved: coverage_{key}.png")
                
                plt.close()
                
            else:
                print(f"    Warning: Parameter '{key}' not found in dataset")
                
        except Exception as e:
            print(f"    Error plotting {key}: {e}")
    
    print("Coverage plots generated successfully!")


def plot_los_coverage_3d(dataset, save_plots=True):
    """
    Create a detailed 3D Line-of-Sight coverage plot.
    
    Args:
        dataset: DeepMIMO dataset object to visualize
        save_plots: Whether to save plots to disk
    """
    print("\n" + "=" * 60)
    print("GENERATING 3D LINE-OF-SIGHT COVERAGE")
    print("=" * 60)
    
    try:
        # Create 3D LoS coverage plot with base station information
        dm.plot_coverage(
            dataset.rx_pos,
            dataset["los"],
            bs_pos=dataset.tx_pos.T,
            bs_ori=dataset.tx_ori,
            title="3D Line-of-Sight Coverage Map",
            cbar_title="LoS Status",
            proj_3D=True,
            scat_sz=0.1,
        )
        
        if save_plots:
            plt.savefig(PLOTS_DIR / 'los_coverage_3d.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: los_coverage_3d.png")
        
        plt.close()
        print("3D LoS coverage plot created successfully!")
        
    except Exception as e:
        print(f"Error creating 3D LoS plot: {e}")


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to orchestrate the dataset exploration and visualization."""
    print("DeepMIMO Dataset Exploration & Visualization")
    print("=" * 60)
    
    # Setup output directory
    setup_plots_directory()
    
    # Setup and configuration
    ch_params = setup_channel_parameters()
    
    # Load dataset and compute channels
    _, first_bs_dataset = load_and_compute_dataset(SCENARIO_NAME, ch_params)
    
    # Analyze dataset structure
    array_keys, dict_keys, other_keys = analyze_dataset_structure(first_bs_dataset)
    
    # Display detailed information about each data type
    display_numpy_arrays(first_bs_dataset, array_keys)
    display_dictionaries(first_bs_dataset, dict_keys)
    display_other_types(first_bs_dataset, other_keys)
    
    # Show dataset statistics
    display_dataset_statistics(first_bs_dataset)
    
    # Analyze channel statistics
    analyze_channel_statistics(first_bs_dataset)
    
    # Analyze position activity
    active_mask, inactive_mask = analyze_position_activity(first_bs_dataset)
    
    # Generate all visualizations (saved to disk, not displayed)
    plot_position_activity(first_bs_dataset, active_mask, inactive_mask, save_plots=True)
    plot_los_coverage_3d(first_bs_dataset, save_plots=True)
    plot_channel_characteristics(first_bs_dataset, save_plots=True)
    
    print("\n" + "=" * 60)
    print("EXPLORATION & VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()