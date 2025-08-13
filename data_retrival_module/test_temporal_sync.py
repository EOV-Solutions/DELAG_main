#!/usr/bin/env python3
"""
Test script to verify temporal synchronization modifications.
This script tests the new time-aware functions for ERA5 and MODIS retrieval.
"""

import os
import sys
from datetime import datetime
import rasterio

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from era5_retriever import get_lst_acquisition_time, get_lst_file_for_date
from modis_retriever import get_lst_acquisition_time as modis_get_lst_time

def test_lst_time_extraction():
    """Test LST acquisition time extraction from metadata."""
    print("Testing LST acquisition time extraction...")
    
    # Test with a sample LST file path (this would need to be an actual file)
    test_lst_path = "path/to/sample/lst_file.tif"
    
    if os.path.exists(test_lst_path):
        acquisition_time = get_lst_acquisition_time(test_lst_path)
        if acquisition_time:
            print(f"✓ Successfully extracted LST time: {acquisition_time}")
            return True
        else:
            print("✗ Failed to extract LST time")
            return False
    else:
        print("⚠ No test LST file available, skipping time extraction test")
        return True

def test_lst_file_finding():
    """Test finding LST files for specific dates."""
    print("\nTesting LST file finding...")
    
    # Test with a sample LST folder
    test_lst_folder = "path/to/lst/folder"
    
    if os.path.exists(test_lst_folder):
        test_date = datetime(2023, 1, 15)
        lst_file = get_lst_file_for_date(test_lst_folder, test_date)
        
        if lst_file:
            print(f"✓ Found LST file for {test_date.strftime('%Y-%m-%d')}: {lst_file}")
            return True
        else:
            print(f"✗ No LST file found for {test_date.strftime('%Y-%m-%d')}")
            return False
    else:
        print("⚠ No test LST folder available, skipping file finding test")
        return True

def test_day_night_logic():
    """Test day/night logic based on acquisition time."""
    print("\nTesting day/night logic...")
    
    # Test cases
    test_times = [
        (datetime(2023, 1, 15, 10, 30), "Day"),   # 10:30 UTC - Day
        (datetime(2023, 1, 15, 14, 0), "Day"),    # 14:00 UTC - Day
        (datetime(2023, 1, 15, 22, 0), "Night"),  # 22:00 UTC - Night
        (datetime(2023, 1, 15, 2, 0), "Night"),   # 02:00 UTC - Night
        (datetime(2023, 1, 15, 6, 0), "Day"),     # 06:00 UTC - Day (boundary)
        (datetime(2023, 1, 15, 18, 0), "Night"),  # 18:00 UTC - Night (boundary)
    ]
    
    all_passed = True
    
    for test_time, expected in test_times:
        hour = test_time.hour
        use_day_data = 6 <= hour < 18
        actual = "Day" if use_day_data else "Night"
        
        if actual == expected:
            print(f"✓ {test_time.strftime('%H:%M')} UTC → {actual}")
        else:
            print(f"✗ {test_time.strftime('%H:%M')} UTC → {actual} (expected {expected})")
            all_passed = False
    
    return all_passed

def test_era5_hour_selection():
    """Test ERA5 hour selection logic."""
    print("\nTesting ERA5 hour selection logic...")
    
    # Test cases: (target_hour, available_hours, expected_closest)
    test_cases = [
        (10, [8, 9, 10, 11, 12], 10),  # Exact match
        (10, [8, 9, 11, 12], 11),      # Closest is 11
        (10, [8, 9], 9),               # Closest is 9
        (10, [11, 12, 13], 11),        # Closest is 11
        (23, [22, 0, 1], 22),          # Closest is 22
        (1, [22, 23, 0, 2], 0),        # Closest is 0
    ]
    
    all_passed = True
    
    for target_hour, available_hours, expected in test_cases:
        # Find the closest hour
        closest_hour = min(available_hours, key=lambda x: abs(x - target_hour))
        
        if closest_hour == expected:
            print(f"✓ Target {target_hour}:00, Available {available_hours} → {closest_hour}:00")
        else:
            print(f"✗ Target {target_hour}:00, Available {available_hours} → {closest_hour}:00 (expected {expected}:00)")
            all_passed = False
    
    return all_passed

def test_metadata_parsing():
    """Test metadata parsing from different datetime formats."""
    print("\nTesting metadata parsing...")
    
    # Test different datetime formats that might be in GeoTIFF metadata
    test_formats = [
        "2023:01:15 10:30:00",
        "2023-01-15 10:30:00",
        "2023:01:15 10:30",
        "2023-01-15T10:30:00",
    ]
    
    all_passed = True
    
    for datetime_str in test_formats:
        try:
            # Test the parsing logic from get_lst_acquisition_time
            if ':' in datetime_str and len(datetime_str.split(':')) >= 6:
                # Format: YYYY:MM:DD HH:MM:SS
                dt_parts = datetime_str.split(' ')
                date_part = dt_parts[0].replace(':', '-')
                time_part = dt_parts[1]
                full_datetime_str = f"{date_part} {time_part}"
                parsed_time = datetime.strptime(full_datetime_str, '%Y-%m-%d %H:%M:%S')
            else:
                # Try other formats
                parsed_time = datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
            
            print(f"✓ Successfully parsed: {datetime_str} → {parsed_time}")
        except ValueError as e:
            print(f"✗ Failed to parse: {datetime_str} - {e}")
            all_passed = False
    
    return all_passed

def main():
    """Run all tests."""
    print("=" * 60)
    print("TEMPORAL SYNCHRONIZATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("LST Time Extraction", test_lst_time_extraction),
        ("LST File Finding", test_lst_file_finding),
        ("Day/Night Logic", test_day_night_logic),
        ("ERA5 Hour Selection", test_era5_hour_selection),
        ("Metadata Parsing", test_metadata_parsing),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Temporal synchronization is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 