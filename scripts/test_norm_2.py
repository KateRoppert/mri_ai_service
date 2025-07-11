#!/usr/bin/env python3
"""
MRI Intensity Normalization Validation Script

This script validates the results of intensity normalization on brain MRI images
by comparing statistics and generating histogram plots before and after normalization.

Requirements:
    - nibabel
    - numpy
    - matplotlib
    - scipy (optional, for additional statistics)

Usage:
    python validate_normalization.py raw_image.nii normalized_image.nii output_dir/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from datetime import datetime
from pathlib import Path


def load_mri_image(image_path):
    """
    Load MRI image from file path.
    
    Args:
        image_path (str): Path to the MRI image file
        
    Returns:
        numpy.ndarray: Image data as numpy array
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If image cannot be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        img = nib.load(image_path)
        data = img.get_fdata()
        return data
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {str(e)}")


def calculate_statistics(data, mask_zeros=True):
    """
    Calculate basic statistics for image data.
    
    Args:
        data (numpy.ndarray): Image data
        mask_zeros (bool): Whether to exclude zero values from statistics
        
    Returns:
        dict: Dictionary containing statistics
    """
    if mask_zeros:
        # Remove zero/background voxels for brain tissue statistics
        masked_data = data[data > 0]
    else:
        masked_data = data.flatten()
    
    if len(masked_data) == 0:
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'std': 0.0,
            'median': 0.0,
            'q25': 0.0,
            'q75': 0.0,
            'n_voxels': 0,
            'n_nonzero': 0
        }
    
    stats = {
        'min': float(np.min(masked_data)),
        'max': float(np.max(masked_data)),
        'mean': float(np.mean(masked_data)),
        'std': float(np.std(masked_data)),
        'median': float(np.median(masked_data)),
        'q25': float(np.percentile(masked_data, 25)),
        'q75': float(np.percentile(masked_data, 75)),
        'n_voxels': int(data.size),
        'n_nonzero': int(len(masked_data))
    }
    
    return stats


def generate_histogram_plot(raw_data, norm_data, output_path, bins=100):
    """
    Generate histogram comparison plot.
    
    Args:
        raw_data (numpy.ndarray): Raw image data
        norm_data (numpy.ndarray): Normalized image data
        output_path (str): Path to save the plot
        bins (int): Number of histogram bins
    """
    # Mask zero values for better visualization
    raw_masked = raw_data[raw_data > 0]
    norm_masked = norm_data[norm_data > 0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Raw image histogram
    ax1.hist(raw_masked, bins=bins, alpha=0.7, color='blue', edgecolor='black', linewidth=0.5)
    ax1.set_title('Raw Image Intensity Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Intensity Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    raw_stats = calculate_statistics(raw_data)
    stats_text = f"Mean: {raw_stats['mean']:.2f}\nStd: {raw_stats['std']:.2f}\nMin: {raw_stats['min']:.2f}\nMax: {raw_stats['max']:.2f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Normalized image histogram
    ax2.hist(norm_masked, bins=bins, alpha=0.7, color='red', edgecolor='black', linewidth=0.5)
    ax2.set_title('Normalized Image Intensity Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Intensity Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    norm_stats = calculate_statistics(norm_data)
    stats_text = f"Mean: {norm_stats['mean']:.2f}\nStd: {norm_stats['std']:.2f}\nMin: {norm_stats['min']:.2f}\nMax: {norm_stats['max']:.2f}"
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_overlay_histogram(raw_data, norm_data, output_path, bins=100):
    """
    Generate overlaid histogram comparison plot.
    
    Args:
        raw_data (numpy.ndarray): Raw image data
        norm_data (numpy.ndarray): Normalized image data
        output_path (str): Path to save the plot
        bins (int): Number of histogram bins
    """
    raw_masked = raw_data[raw_data > 0]
    norm_masked = norm_data[norm_data > 0]
    
    plt.figure(figsize=(12, 8))
    
    # Plot both histograms with transparency
    plt.hist(raw_masked, bins=bins, alpha=0.6, color='blue', label='Raw Image', density=True)
    plt.hist(norm_masked, bins=bins, alpha=0.6, color='red', label='Normalized Image', density=True)
    
    plt.title('Intensity Distribution Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Intensity Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_statistics_report(raw_stats, norm_stats, output_path, raw_path, norm_path):
    """
    Save detailed statistics report to text file.
    
    Args:
        raw_stats (dict): Raw image statistics
        norm_stats (dict): Normalized image statistics  
        output_path (str): Path to save the report
        raw_path (str): Path to raw image file
        norm_path (str): Path to normalized image file
    """
    with open(output_path, 'w') as f:
        f.write("MRI Intensity Normalization Validation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Raw image: {raw_path}\n")
        f.write(f"Normalized image: {norm_path}\n\n")
        
        f.write("INTENSITY STATISTICS (excluding zero voxels)\n")
        f.write("-" * 45 + "\n\n")
        
        f.write("Raw Image Statistics:\n")
        f.write(f"  Minimum:     {raw_stats['min']:.6f}\n")
        f.write(f"  Maximum:     {raw_stats['max']:.6f}\n")
        f.write(f"  Mean:        {raw_stats['mean']:.6f}\n")
        f.write(f"  Std Dev:     {raw_stats['std']:.6f}\n")
        f.write(f"  Median:      {raw_stats['median']:.6f}\n")
        f.write(f"  25th perc:   {raw_stats['q25']:.6f}\n")
        f.write(f"  75th perc:   {raw_stats['q75']:.6f}\n")
        f.write(f"  Total voxels: {raw_stats['n_voxels']}\n")
        f.write(f"  Non-zero:    {raw_stats['n_nonzero']}\n\n")
        
        f.write("Normalized Image Statistics:\n")
        f.write(f"  Minimum:     {norm_stats['min']:.6f}\n")
        f.write(f"  Maximum:     {norm_stats['max']:.6f}\n")
        f.write(f"  Mean:        {norm_stats['mean']:.6f}\n")
        f.write(f"  Std Dev:     {norm_stats['std']:.6f}\n")
        f.write(f"  Median:      {norm_stats['median']:.6f}\n")
        f.write(f"  25th perc:   {norm_stats['q25']:.6f}\n")
        f.write(f"  75th perc:   {norm_stats['q75']:.6f}\n")
        f.write(f"  Total voxels: {norm_stats['n_voxels']}\n")
        f.write(f"  Non-zero:    {norm_stats['n_nonzero']}\n\n")
        
        f.write("COMPARISON METRICS\n")
        f.write("-" * 20 + "\n\n")
        
        # Calculate changes
        mean_change = ((norm_stats['mean'] - raw_stats['mean']) / raw_stats['mean']) * 100 if raw_stats['mean'] != 0 else 0
        std_change = ((norm_stats['std'] - raw_stats['std']) / raw_stats['std']) * 100 if raw_stats['std'] != 0 else 0
        range_raw = raw_stats['max'] - raw_stats['min']
        range_norm = norm_stats['max'] - norm_stats['min']
        range_change = ((range_norm - range_raw) / range_raw) * 100 if range_raw != 0 else 0
        
        f.write(f"Mean change:           {mean_change:+.2f}%\n")
        f.write(f"Std deviation change:  {std_change:+.2f}%\n")
        f.write(f"Range change:          {range_change:+.2f}%\n")
        f.write(f"Coefficient of variation before: {(raw_stats['std']/raw_stats['mean']):+.4f}\n")
        f.write(f"Coefficient of variation after:  {(norm_stats['std']/norm_stats['mean']):+.4f}\n")


def main():
    parser = argparse.ArgumentParser(description='Validate MRI intensity normalization results')
    parser.add_argument('raw_image', help='Path to raw MRI image')
    parser.add_argument('normalized_image', help='Path to normalized MRI image')
    parser.add_argument('output_dir', help='Directory to save validation results')
    parser.add_argument('--bins', type=int, default=100, help='Number of histogram bins (default: 100)')
    parser.add_argument('--prefix', type=str, default='validation', help='Prefix for output files (default: validation)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.raw_image):
        print(f"Error: Raw image file not found: {args.raw_image}")
        sys.exit(1)
        
    if not os.path.exists(args.normalized_image):
        print(f"Error: Normalized image file not found: {args.normalized_image}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load images
        print("Loading images...")
        raw_data = load_mri_image(args.raw_image)
        norm_data = load_mri_image(args.normalized_image)
        
        # Check if images have the same shape
        if raw_data.shape != norm_data.shape:
            print(f"Warning: Image shapes don't match. Raw: {raw_data.shape}, Normalized: {norm_data.shape}")
        
        # Calculate statistics
        print("Calculating statistics...")
        raw_stats = calculate_statistics(raw_data)
        norm_stats = calculate_statistics(norm_data)
        
        # Generate output file paths
        stats_file = os.path.join(args.output_dir, f"{args.prefix}_statistics.txt")
        hist_file = os.path.join(args.output_dir, f"{args.prefix}_histograms.png")
        overlay_file = os.path.join(args.output_dir, f"{args.prefix}_comparison.png")
        
        # Save statistics report
        print("Generating statistics report...")
        save_statistics_report(raw_stats, norm_stats, stats_file, args.raw_image, args.normalized_image)
        
        # Generate histogram plots
        print("Generating histogram plots...")
        generate_histogram_plot(raw_data, norm_data, hist_file, bins=args.bins)
        generate_overlay_histogram(raw_data, norm_data, overlay_file, bins=args.bins)
        
        print(f"\nValidation completed successfully!")
        print(f"Results saved to:")
        print(f"  Statistics: {stats_file}")
        print(f"  Histograms: {hist_file}")
        print(f"  Comparison: {overlay_file}")
        
        # Print summary to console
        print(f"\nQuick Summary:")
        print(f"Raw image    - Mean: {raw_stats['mean']:.3f}, Std: {raw_stats['std']:.3f}")
        print(f"Normalized   - Mean: {norm_stats['mean']:.3f}, Std: {norm_stats['std']:.3f}")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()