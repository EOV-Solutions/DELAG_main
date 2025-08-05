#!/usr/bin/env python3
"""
Debug script to test ERA5 file detection logic
"""
import os
import re
from datetime import datetime

# Test the exact logic used in data_preprocessing.py
BASE_DATA_DIR = "/mnt/hdd12tb/code/nhatvm/DELAG_main/data/preprocessed_data"
ROI_NAME = "D-49-49-A-c-2"
ERA5_SKIN_TEMP_SUBDIR = "era5"

roi_base_path = os.path.join(BASE_DATA_DIR, ROI_NAME)
era5_skin_temp_path = os.path.join(roi_base_path, ERA5_SKIN_TEMP_SUBDIR)

print(f"Testing ERA5 file detection:")
print(f"BASE_DATA_DIR: {BASE_DATA_DIR}")
print(f"ROI_NAME: {ROI_NAME}")
print(f"roi_base_path: {roi_base_path}")
print(f"era5_skin_temp_path: {era5_skin_temp_path}")
print(f"Path exists: {os.path.exists(era5_skin_temp_path)}")

if os.path.exists(era5_skin_temp_path):
    all_files = os.listdir(era5_skin_temp_path)
    tif_files = [f for f in all_files if f.endswith('.tif')]
    print(f"Total files: {len(all_files)}")
    print(f"TIF files: {len(tif_files)}")
    
    if tif_files:
        print(f"First few TIF files: {tif_files[:5]}")
        
        # Test regex pattern
        all_era5_dates = []
        for era5_file in tif_files[:10]:  # Test first 10 files
            try:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', era5_file)
                if date_match:
                    era5_date = datetime.strptime(date_match.group(1), '%Y-%m-%d').date()
                    all_era5_dates.append(era5_date)
                    print(f"✓ {era5_file} -> {era5_date}")
                else:
                    print(f"✗ {era5_file} -> No date match")
            except Exception as e:
                print(f"✗ {era5_file} -> Error: {e}")
        
        print(f"Successfully parsed {len(all_era5_dates)} dates from first 10 files")
else:
    print("ERROR: Path does not exist!") 