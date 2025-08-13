#!/usr/bin/env python3
"""
Example usage of the MODIS Reconstruction Analysis Script

This script demonstrates how to use the modis_reconstruction_analysis.py
with different configurations and parameters.
"""

import subprocess
import sys
from pathlib import Path

def run_analysis_example():
    """Run example analysis with different configurations."""
    
    # Example 1: Basic analysis for Vietnam
    print("=== Example 1: Basic Analysis for Vietnam ===")
    cmd1 = [
        sys.executable, "modis_reconstruction_analysis.py",
        "--roi", "vietnam",
        "--gt-folder", "data/retrieved_data/vietnam/movis",
        "--recon-folder", "data/output/vietnam/reconstructed_lst_train",
        "--output-dir", "results/vietnam"
    ]
    
    print("Command:", " ".join(cmd1))
    print("This will:")
    print("- Filter GT images with >25% non-NaN pixels")
    print("- Create comparison plots with 10 images per plot")
    print("- Generate time series for 5 random pixels")
    print("- Calculate RMSE and MAE metrics")
    print()
    
    # Example 2: Custom parameters
    print("=== Example 2: Custom Parameters ===")
    cmd2 = [
        sys.executable, "modis_reconstruction_analysis.py",
        "--roi", "thailand",
        "--gt-folder", "data/retrieved_data/thailand/movis",
        "--recon-folder", "data/output/thailand/reconstructed_lst_train",
        "--output-dir", "results/thailand",
        "--min-percentage", "30.0",
        "--images-per-plot", "5",
        "--num-pixels", "3"
    ]
    
    print("Command:", " ".join(cmd2))
    print("This will:")
    print("- Filter GT images with >30% non-NaN pixels")
    print("- Create comparison plots with 5 images per plot")
    print("- Generate time series for 3 random pixels")
    print("- Calculate RMSE and MAE metrics")
    print()
    
    # Example 3: High-quality analysis
    print("=== Example 3: High-Quality Analysis ===")
    cmd3 = [
        sys.executable, "modis_reconstruction_analysis.py",
        "--roi", "cambodia",
        "--gt-folder", "data/retrieved_data/cambodia/movis",
        "--recon-folder", "data/output/cambodia/reconstructed_lst_train",
        "--output-dir", "results/cambodia",
        "--min-percentage", "50.0",
        "--images-per-plot", "8",
        "--num-pixels", "10"
    ]
    
    print("Command:", " ".join(cmd3))
    print("This will:")
    print("- Filter GT images with >50% non-NaN pixels (higher quality)")
    print("- Create comparison plots with 8 images per plot")
    print("- Generate time series for 10 random pixels")
    print("- Calculate RMSE and MAE metrics")
    print()
    
    # Ask user if they want to run any of these examples
    print("To run any of these examples, uncomment the corresponding line below:")
    print()
    print("# subprocess.run(cmd1)  # Basic Vietnam analysis")
    print("# subprocess.run(cmd2)  # Custom Thailand analysis")
    print("# subprocess.run(cmd3)  # High-quality Cambodia analysis")
    print()
    print("Or run the script directly with your own parameters:")
    print("python modis_reconstruction_analysis.py --help")

def check_data_structure():
    """Check if the expected data structure exists."""
    print("=== Checking Data Structure ===")
    
    expected_paths = [
        "data/retrieved_data",
        "data/output",
        "modis_reconstruction_analysis.py"
    ]
    
    for path in expected_paths:
        if Path(path).exists():
            print(f"✓ {path} exists")
        else:
            print(f"✗ {path} missing")
    
    print()
    print("Expected folder structure:")
    print("data/")
    print("├── retrieved_data/")
    print("│   └── {ROI}/")
    print("│       └── movis/")
    print("│           └── *_YYYY-MM-DD.tif")
    print("└── output/")
    print("    └── {ROI}/")
    print("        └── reconstructed_lst_train/")
    print("            └── *_YYYYMMDD.tif")
    print()

if __name__ == "__main__":
    print("MODIS Reconstruction Analysis - Example Usage")
    print("=" * 50)
    print()
    
    check_data_structure()
    run_analysis_example() 