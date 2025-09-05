#!/usr/bin/env python3
"""
Example usage of the comprehensive GEE downloader

This script demonstrates how to use the comprehensive_gee_downloader.py
for downloading multiple satellite datasets from Google Earth Engine
using both grid-based and region-based approaches.
"""

import os
import ee
from datetime import datetime
from comprehensive_gee_downloader import (
    GEEDataDownloader, 
    find_grid_feature, 
    get_roi_geometry_from_geojson,
    get_roi_geometry
)

def example_grid_based_usage():
    """Grid-based usage example (recommended approach)"""
    
    # Initialize Earth Engine
    try:
        ee.Initialize(project='ee-hadat-461702-p4')
    except:
        ee.Authenticate()
        ee.Initialize(project='ee-hadat-461702-p4')
    
    # Grid-based approach using PhienHieu
    grid_file = "data/Grid_50K_MatchedDates.geojson"  # Update path as needed
    phien_hieu = "D-49-49-A"  # Example grid identifier
    
    # Find the grid feature
    feature = find_grid_feature(phien_hieu, grid_file)
    if not feature:
        print(f"Grid feature {phien_hieu} not found!")
        return
    
    # Get geometry from grid feature
    roi_geometry = get_roi_geometry_from_geojson(feature['geometry'])
    
    # Define date range
    start_date = datetime(2023, 6, 15)
    end_date = datetime(2023, 6, 20)
    
    # Output directory (will create subdirectory with phien_hieu)
    output_dir = f"./grid_downloads/{phien_hieu}"
    
    # Initialize downloader
    downloader = GEEDataDownloader(output_dir, roi_geometry)
    
    # Download specific datasets
    datasets_to_download = ['era5', 'aster', 'landsat8_l2']
    
    print(f"Starting grid-based download for {phien_hieu}...")
    task_mapping = downloader.download_all_for_date_range(
        start_date, 
        end_date,
        datasets_to_download
    )
    
    print(f"Download complete! {len(task_mapping)} tasks completed.")
    print(f"Check {output_dir}/downloaded_zips/ for ZIP files")
    print(f"Check {output_dir}/task_mapping.json for task mapping")

def example_region_based_usage():
    """Region-based usage example (legacy approach)"""
    
    # Initialize Earth Engine
    try:
        ee.Initialize(project='ee-hadat-461702-p4')
    except:
        ee.Authenticate()
        ee.Initialize(project='ee-hadat-461702-p4')
    
    # Define region of interest (Vietnam example)
    # Format: [[minLon, minLat], [maxLon, minLat], [maxLon, maxLat], [minLon, maxLat], [minLon, minLat]]
    vietnam_region = [
        [105.0, 10.0],   # Southwest corner
        [106.0, 10.0],   # Southeast corner  
        [106.0, 11.0],   # Northeast corner
        [105.0, 11.0],   # Northwest corner
        [105.0, 10.0]    # Close polygon
    ]
    
    # Convert to ee.Geometry
    roi_geometry = get_roi_geometry(vietnam_region)
    
    # Define date range
    start_date = datetime(2023, 6, 15)
    end_date = datetime(2023, 6, 20)
    
    # Output directory
    output_dir = "./region_downloads"
    
    # Initialize downloader
    downloader = GEEDataDownloader(output_dir, roi_geometry)
    
    # Download specific datasets
    datasets_to_download = ['era5', 'aster', 'landsat8_l2']
    
    print("Starting region-based download...")
    task_mapping = downloader.download_all_for_date_range(
        start_date, 
        end_date,
        datasets_to_download
    )
    
    print(f"Download complete! {len(task_mapping)} tasks completed.")
    print(f"Check {output_dir}/downloaded_zips/ for ZIP files")
    print(f"Check {output_dir}/task_mapping.json for task mapping")

def example_full_dataset_grid():
    """Example downloading all available datasets using grid approach"""
    
    # Initialize Earth Engine
    try:
        ee.Initialize(project='ee-hadat-461702-p4')
    except:
        ee.Authenticate()
        ee.Initialize(project='ee-hadat-461702-p4')
    
    # Grid-based approach
    grid_file = "data/Grid_50K_MatchedDates.geojson" 
    phien_hieu = "D-49-49-A"  # Example grid identifier
    
    # Find the grid feature
    feature = find_grid_feature(phien_hieu, grid_file)
    if not feature:
        print(f"Grid feature {phien_hieu} not found!")
        return
    
    roi_geometry = get_roi_geometry_from_geojson(feature['geometry'])
    
    # Week-long period
    start_date = datetime(2023, 7, 1)
    end_date = datetime(2023, 7, 7)
    
    output_dir = f"./full_grid_downloads/{phien_hieu}"
    
    # Initialize downloader
    downloader = GEEDataDownloader(output_dir, roi_geometry)
    
    # Download all datasets (default behavior)
    print(f"Starting full dataset download for grid {phien_hieu}...")
    task_mapping = downloader.download_all_for_date_range(start_date, end_date)
    
    print(f"Full download complete! {len(task_mapping)} tasks completed.")

if __name__ == "__main__":
    print("=== GEE Downloader Examples ===")
    print("Choose an example:")
    print("1. Grid-based usage (recommended)")
    print("2. Region-based usage (legacy)")
    print("3. Full dataset download with grid")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        example_grid_based_usage()
    elif choice == "2":
        example_region_based_usage()
    elif choice == "3":
        example_full_dataset_grid()
    else:
        print("Invalid choice. Running grid-based example...")
        example_grid_based_usage()
