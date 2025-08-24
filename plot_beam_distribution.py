#!/usr/bin/env python3
"""
Script to plot the distribution of best beam indices for test data.

This script reads all .npz files in the processed_channels/test folder,
extracts the beam labels for beam size 16, and plots the distribution
of best beam indices across all users.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import sys


def load_beam_labels_from_file(file_path, beam_size=16):
    """
    Load beam labels from a single .npz file for a specific beam size.
    
    Args:
        file_path: Path to the .npz file
        beam_size: Beam size to extract (default: 16)
        
    Returns:
        numpy.ndarray: Beam labels for the specified beam size, or None if not found
    """
    try:
        # Load the file
        data = np.load(file_path, allow_pickle=True)
        
        # Check if beam_labels exists
        if 'beam_labels' not in data:
            print(f"    Warning: {file_path.name} has no beam_labels")
            data.close()
            return None
        
        # Check if beam_calculation_info exists to verify beam sizes
        if 'beam_calculation_info' in data:
            beam_sizes = data['beam_calculation_info'].item()['beam_sizes']
            if beam_size not in beam_sizes:
                print(f"    Warning: {file_path.name} doesn't have beam size {beam_size}")
                print(f"    Available beam sizes: {beam_sizes}")
                data.close()
                return None
        
        # Get beam labels
        beam_labels = data['beam_labels']
        
        # Find the column index for the specified beam size
        if 'beam_calculation_info' in data:
            beam_sizes = data['beam_calculation_info'].item()['beam_sizes']
            try:
                beam_idx = beam_sizes.index(beam_size)
                beam_labels_for_size = beam_labels[:, beam_idx]
            except ValueError:
                print(f"    Error: Beam size {beam_size} not found in {file_path.name}")
                data.close()
                return None
        else:
            # Fallback: assume first column if no beam_calculation_info
            print(f"    Warning: No beam_calculation_info in {file_path.name}, using first column")
            beam_labels_for_size = beam_labels[:, 0]
        
        data.close()
        return beam_labels_for_size
        
    except Exception as e:
        print(f"    Error loading {file_path.name}: {str(e)}")
        return None


def plot_beam_distribution(beam_labels_all, beam_size=16, output_dir="./plots", folder_names="data"):
    """
    Create plots showing the distribution of best beam indices.
    
    Args:
        beam_labels_all: List of beam label arrays from all files
        beam_size: Beam size being analyzed
        output_dir: Directory to save plots
        folder_names: Names of folders being analyzed (for plot titles)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Combine all beam labels and convert to integers
    all_beams = np.concatenate(beam_labels_all)
    all_beams = all_beams.astype(int)  # Convert to integers
    
    print(f"\nPlotting distribution for {len(all_beams)} total beam selections")
    print(f"Beam size: {beam_size}")
    print(f"Beam index range: 0 to {beam_size-1}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Distribution of Best Beam Indices - {folder_names} (Beam Size: {beam_size})', fontsize=16, fontweight='bold')
    
    # 1. Histogram
    axes[0, 0].hist(all_beams, bins=beam_size, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Beam Index')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Histogram of Best Beam Indices')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bar plot
    unique_beams, counts = np.unique(all_beams, return_counts=True)
    axes[0, 1].bar(unique_beams, counts, alpha=0.7, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Beam Index')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Count of Each Beam Index')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    sorted_beams = np.sort(all_beams)
    cumulative = np.arange(1, len(sorted_beams) + 1) / len(sorted_beams)
    axes[1, 0].plot(sorted_beams, cumulative, linewidth=2, marker='o', markersize=3)
    axes[1, 0].set_xlabel('Beam Index')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution of Beam Indices')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot
    axes[1, 1].boxplot(all_beams, vert=False)
    axes[1, 1].set_xlabel('Beam Index')
    axes[1, 1].set_title('Box Plot of Beam Indices')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot with folder information
    folder_suffix = folder_names.replace(" ", "_").replace("and", "&")
    plot_filename = output_path / f"beam_distribution_{folder_suffix}_bs{beam_size}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {plot_filename}")
    
    # Show the plot
    plt.show()
    
    # Print statistics
    print(f"\n📊 BEAM DISTRIBUTION STATISTICS")
    print(f"=" * 50)
    print(f"Total beam selections: {len(all_beams)}")
    print(f"Mean beam index: {np.mean(all_beams):.2f}")
    print(f"Median beam index: {np.median(all_beams):.2f}")
    print(f"Standard deviation: {np.std(all_beams):.2f}")
    print(f"Min beam index: {np.min(all_beams)}")
    print(f"Max beam index: {np.max(all_beams)}")
    
    # Most and least common beams
    unique_beams, counts = np.unique(all_beams, return_counts=True)
    sorted_indices = np.argsort(counts)[::-1]  # Sort by count (descending)
    
    print(f"\n🏆 MOST COMMON BEAM INDICES:")
    for i in range(min(5, len(unique_beams))):
        idx = sorted_indices[i]
        beam_idx = int(unique_beams[idx])  # Ensure integer
        count = int(counts[idx])  # Ensure integer
        percentage = 100 * count / len(all_beams)
        print(f"   Beam {beam_idx:2d}: {count:6d} times ({percentage:5.1f}%)")
    
    print(f"\n📉 LEAST COMMON BEAM INDICES:")
    for i in range(min(5, len(unique_beams))):
        idx = sorted_indices[-(i+1)]  # Reverse order
        beam_idx = int(unique_beams[idx])  # Ensure integer
        count = int(counts[idx])  # Ensure integer
        percentage = 100 * count / len(all_beams)
        print(f"   Beam {beam_idx:2d}: {count:6d} times ({percentage:5.1f}%)")
    
    return all_beams


def main():
    """Main function to analyze beam label distributions."""
    parser = argparse.ArgumentParser(
        description="Plot distribution of best beam indices for DeepMIMO data"
    )
    parser.add_argument(
        "--data-folder", 
        type=str, 
        default="./processed_channels",
        help="Path to base folder containing data subdirectories (default: ./processed_channels)"
    )
    parser.add_argument(
        "--subfolder", 
        type=str, 
        choices=["test", "pretrain", "both"],
        default="both",
        help="Which subfolder to analyze: test, pretrain, or both (default: test)"
    )
    parser.add_argument(
        "--beam-size", 
        type=int, 
        default=16,
        help="Beam size to analyze (default: 16)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./plots",
        help="Directory to save plots (default: ./plots)"
    )
    
    args = parser.parse_args()
    
    # Convert to Path object
    base_folder = Path(args.data_folder)
    
    # Determine which subfolders to process
    if args.subfolder == "both":
        target_folders = ["pretrain", "test"]
        folder_names = "pretrain and test"
    else:
        target_folders = [args.subfolder]
        folder_names = args.subfolder
    
    # Find all .npz files in target folders
    npz_files = []
    folder_stats = {}
    
    for subfolder in target_folders:
        subfolder_path = base_folder / subfolder
        if not subfolder_path.exists():
            print(f"Warning: {subfolder} folder does not exist: {subfolder_path}")
            continue
        if not subfolder_path.is_dir():
            print(f"Warning: {subfolder_path} is not a directory")
            continue
            
        subfolder_files = list(subfolder_path.glob("*.npz"))
        npz_files.extend(subfolder_files)
        folder_stats[subfolder] = len(subfolder_files)
        print(f"Found {len(subfolder_files)} .npz files in {subfolder}/")
    
    if not npz_files:
        print(f"No .npz files found in {folder_names} folders")
        sys.exit(0)
    
    print(f"\nTotal: {len(npz_files)} .npz files in {folder_names}")
    print(f"Analyzing beam size: {args.beam_size}")
    
    # Load beam labels from all files
    print(f"\nLoading beam labels...")
    beam_labels_all = []
    successful_files = 0
    
    for file_path in tqdm(npz_files, desc="Loading files"):
        beam_labels = load_beam_labels_from_file(file_path, args.beam_size)
        if beam_labels is not None:
            beam_labels_all.append(beam_labels)
            successful_files += 1
    
    if not beam_labels_all:
        print("No beam labels found in any files. Exiting.")
        sys.exit(1)
    
    print(f"\nSuccessfully loaded beam labels from {successful_files}/{len(npz_files)} files")
    
    # Plot the distribution
    all_beams = plot_beam_distribution(beam_labels_all, args.beam_size, args.output_dir, folder_names)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Plots saved to: {args.output_dir}")
    
    # Show folder breakdown
    if args.subfolder == "both":
        print(f"\n📊 FOLDER BREAKDOWN:")
        for folder, count in folder_stats.items():
            print(f"   {folder}/: {count} files")


if __name__ == "__main__":
    main() 