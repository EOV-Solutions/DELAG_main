#!/bin/bash

# Test script for the comprehensive GEE downloader
# This script demonstrates various command line usage patterns

echo "=== Google Earth Engine Comprehensive Downloader Test ==="
echo ""

# Test 1: Grid-based download (recommended approach)
echo "Test 1: Grid-based ERA5 and ASTER download"
echo "Command:"
echo "python comprehensive_gee_downloader.py \\"
echo "  --start_date 2023-06-15 \\"
echo "  --end_date 2023-06-16 \\"
echo "  --grid_file \"data/Grid_50K_MatchedDates.geojson\" \\"
echo "  --phien_hieu \"D-49-49-A\" \\"
echo "  --output_dir \"./test_grid\" \\"
echo "  --datasets era5 aster"
echo ""

# Uncomment to run:
# python comprehensive_gee_downloader.py \
#   --start_date 2023-06-15 \
#   --end_date 2023-06-16 \
#   --grid_file "data/Grid_50K_MatchedDates.geojson" \
#   --phien_hieu "D-49-49-A" \
#   --output_dir "./test_grid" \
#   --datasets era5 aster

echo "---"
echo ""

# Test 1b: Legacy region-based download
echo "Test 1b: Legacy region-based ERA5 and ASTER download"
echo "Command:"
echo "python comprehensive_gee_downloader.py \\"
echo "  --start_date 2023-06-15 \\"
echo "  --end_date 2023-06-16 \\"
echo "  --region \"105.0,10.0,106.0,11.0\" \\"
echo "  --output_dir \"./test_vietnam\" \\"
echo "  --datasets era5 aster"
echo ""

# Uncomment to run:
# python comprehensive_gee_downloader.py \
#   --start_date 2023-06-15 \
#   --end_date 2023-06-16 \
#   --region "105.0,10.0,106.0,11.0" \
#   --output_dir "./test_vietnam" \
#   --datasets era5 aster

echo "---"
echo ""

# Test 2: Landsat data for San Francisco
echo "Test 2: Landsat 8 L2 download for San Francisco Bay Area"
echo "Command:"
echo "python comprehensive_gee_downloader.py \\"
echo "  --start_date 2023-07-01 \\"
echo "  --end_date 2023-07-02 \\"
echo "  --region \"-122.5,37.5,-122.0,38.0\" \\"
echo "  --output_dir \"./test_sf_bay\" \\"
echo "  --datasets landsat8_l2"
echo ""

# Uncomment to run:
# python comprehensive_gee_downloader.py \
#   --start_date 2023-07-01 \
#   --end_date 2023-07-02 \
#   --region "-122.5,37.5,-122.0,38.0" \
#   --output_dir "./test_sf_bay" \
#   --datasets landsat8_l2

echo "---"
echo ""

# Test 3: Full dataset download using grid approach
echo "Test 3: All datasets for a grid (1-day period) - Grid approach"
echo "Command:"
echo "python comprehensive_gee_downloader.py \\"
echo "  --start_date 2023-08-15 \\"
echo "  --end_date 2023-08-15 \\"
echo "  --grid_file \"data/Grid_50K_MatchedDates.geojson\" \\"
echo "  --phien_hieu \"D-49-49-A\" \\"
echo "  --output_dir \"./test_full_grid\""
echo ""

# Uncomment to run:
# python comprehensive_gee_downloader.py \
#   --start_date 2023-08-15 \
#   --end_date 2023-08-15 \
#   --grid_file "data/Grid_50K_MatchedDates.geojson" \
#   --phien_hieu "D-49-49-A" \
#   --output_dir "./test_full_grid"

echo "---"
echo ""

# Test 3b: Legacy full dataset download 
echo "Test 3b: All datasets for a small region (1-day period) - Legacy approach"
echo "Command:"
echo "python comprehensive_gee_downloader.py \\"
echo "  --start_date 2023-08-15 \\"
echo "  --end_date 2023-08-15 \\"
echo "  --region \"100.0,13.0,100.5,13.5\" \\"
echo "  --output_dir \"./test_thailand\""
echo ""

# Uncomment to run:
# python comprehensive_gee_downloader.py \
#   --start_date 2023-08-15 \
#   --end_date 2023-08-15 \
#   --region "100.0,13.0,100.5,13.5" \
#   --output_dir "./test_thailand"

echo "---"
echo ""

# Test 4: Help message
echo "Test 4: Display help message"
echo "Command:"
echo "python comprehensive_gee_downloader.py --help"
echo ""

python comprehensive_gee_downloader.py --help

echo ""
echo "=== Test Commands Ready ==="
echo "Uncomment the desired test commands in this script to execute them."
echo "Make sure you have:"
echo "1. Google Earth Engine authentication set up"
echo "2. Required Python packages installed"
echo "3. Internet connection for downloads"
echo ""
echo "Expected output structure:"
echo "Grid-based approach:"
echo "  output_directory/{phien_hieu}/"
echo "  ├── downloaded_zips/"
echo "  │   ├── downloaded_era5_{task_id}.zip"
echo "  │   ├── downloaded_aster_{task_id}.zip"
echo "  │   └── landsat*_{task_id}.zip"
echo "  └── task_mapping.json"
echo ""
echo "Region-based approach:"
echo "  output_directory/"
echo "  ├── downloaded_zips/"
echo "  │   ├── downloaded_era5_{task_id}.zip"
echo "  │   ├── downloaded_aster_{task_id}.zip"
echo "  │   └── landsat*_{task_id}.zip"
echo "  └── task_mapping.json"
