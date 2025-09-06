import os
import shutil
import time
import rasterio
import numpy as np
import glob
import json
from datetime import datetime, timedelta
from typing import List, Optional

def write_metadata_to_tiff(tif_path, timestamp_ms=None, acquisition_type=None):
    """Writes timestamp metadata to a GeoTIFF file."""
    if os.path.exists(tif_path):
        try:
            with rasterio.open(tif_path, 'r+') as dst:
                tags = {}
                if timestamp_ms:
                    dt_object = datetime.fromtimestamp(timestamp_ms / 1000)
                    datetime_str = dt_object.strftime('%Y:%m:%d %H:%M:%S')
                    tags['DATETIME'] = datetime_str
                    print(f"  > Wrote DATETIME: {datetime_str}")
                
                if acquisition_type:
                    tags['ACQUISITION_TYPE'] = acquisition_type
                    print(f"  > Wrote ACQUISITION_TYPE: {acquisition_type}")
                
                if tags:
                    dst.update_tags(**tags)

        except Exception as e:
            print(f"Warning: Failed to write metadata to {tif_path}: {e}")

def count_nodata(tif_path):
    """Counts the number of NoData pixels in a GeoTIFF."""
    try:
        with rasterio.open(tif_path) as src:
            nodata_value = src.nodata
            band = src.read(1)
            if nodata_value is not None:
                # Handle numerical nodata values
                return np.count_nonzero(band == nodata_value)
            else:
                # Handle NaN as nodata
                return np.count_nonzero(np.isnan(band))
    except Exception as e:
        print(f"Warning: Could not count nodata for {tif_path}: {e}")
        # Return a large number to ensure this file is not preferred
        return float('inf')

# Import the Landsat LST computation module.
try:
    from .lst_module import Landsat_LST as LandsatLST
except ImportError:
    # Fallback for when running as script
    from lst_module import Landsat_LST as LandsatLST


def get_landsat_files_for_date(processed_data_dir: str, target_date: datetime, satellite: str) -> List[str]:
    """
    Find Landsat files for a specific date and satellite in the processed data directory.
    
    Args:
        processed_data_dir: Path to processed data directory
        target_date: Target date to find files for
        satellite: Satellite name (L8, L9)
    
    Returns:
        List of file paths for the target date and satellite
    """
    # Look in both L1 and L2 folders
    l1_folder = os.path.join(processed_data_dir, "landsat_l1")
    l2_folder = os.path.join(processed_data_dir, "landsat_l2")
    
    files = []
    
    # Check L2 first (preferred)
    if os.path.exists(l2_folder):
        date_str = target_date.strftime('%Y%m%d')
        pattern = os.path.join(l2_folder, f"*{satellite}*{date_str}*.tif")
        l2_files = glob.glob(pattern)
        files.extend(l2_files)
    
    # Check L1 if no L2 files found
    if not files and os.path.exists(l1_folder):
        date_str = target_date.strftime('%Y%m%d')
        pattern = os.path.join(l1_folder, f"*{satellite}*{date_str}*.tif")
        l1_files = glob.glob(pattern)
        files.extend(l1_files)
    
    return files

def lst_retrive_offline(processed_data_dir: str, output_folder: str, target_dates: List[datetime]):
    """
    Retrieves Landsat LST data from processed temp data for specific dates.
    Prioritizes L2 over L1 data and L9 over L8 when both are available.
    
    Args:
        processed_data_dir: Path to processed data directory
        output_folder: Path to output folder
        target_dates: List of target dates to process
    """
    satellites = ["L8", "L9"]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for target_date in target_dates:
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"Processing LST for date: {date_str}")
        
        # Track the best file for this date
        best_file = None
        best_satellite = None
        best_nodata_count = float('inf')
        
        # Check all satellites and find the best one
        for satellite in satellites:
            files = get_landsat_files_for_date(processed_data_dir, target_date, satellite)
            
            if not files:
                print(f"No {satellite} files found for {date_str}")
                continue
            
            # Use the first file (should be the best one from processing)
            candidate_file = files[0]
            nodata_count = count_nodata(candidate_file)
            
            print(f"Found {satellite} file: {os.path.basename(candidate_file)} (nodata: {nodata_count})")
            
            # Prefer L9 over L8, and files with less nodata
            if (best_file is None or 
                (satellite == "L9" and best_satellite == "L8") or
                (satellite == best_satellite and nodata_count < best_nodata_count)):
                best_file = candidate_file
                best_satellite = satellite
                best_nodata_count = nodata_count
        
        if best_file is None:
            print(f"No Landsat files found for {date_str}")
            continue
        
        # Copy the best file to output
        output_filename = f"{best_satellite}_lst16days_{date_str}.tif"
        output_path = os.path.join(output_folder, output_filename)
        
        if os.path.exists(output_path):
            print(f"Skipping LST copy for {best_satellite} on {date_str}: file already exists.")
            continue
        
        try:
            # Copy the file
            shutil.copy2(best_file, output_path)
            
            # Write metadata to the copied file
            timestamp_ms = int(target_date.replace(hour=10, minute=30, second=0).timestamp() * 1000)
            write_metadata_to_tiff(output_path, timestamp_ms, f'{best_satellite}_LST_Processed')
            
            print(f"Successfully copied LST for {best_satellite} on {date_str}")
            
        except Exception as e:
            print(f"Failed to copy LST file for {best_satellite} on {date_str}: {e}")

def lst_retrive(date_start, date_end, geometry, ROI, main_folder):
    """
    Retrieves Landsat LST data from processed temp data for a date range.
    This function matches the GEE version's functionality but works with local processed data.
    
    Args:
        date_start: Start date string (YYYY-MM-DD format)
        date_end: End date string (YYYY-MM-DD format) 
        geometry: Geometry object (not used in offline mode, kept for compatibility)
        ROI: Region of Interest name
        main_folder: Main output folder path
    """
    satellites = ["L8", "L9"]
    
    # Convert date strings to datetime objects
    start_date = datetime.strptime(date_start, '%Y-%m-%d')
    end_date = datetime.strptime(date_end, '%Y-%m-%d')
    
    # Generate target dates (daily intervals)
    target_dates = []
    current_date = start_date
    while current_date <= end_date:
        target_dates.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"Processing LST for date range: {date_start} to {date_end}")
    print(f"Generated {len(target_dates)} target dates")
    
    # Create output directory
    dest_folder_path = os.path.join(main_folder, ROI, "lst")
    os.makedirs(dest_folder_path, exist_ok=True)
    
    # Process each satellite
    for satellite in satellites:
        print(f"\nProcessing {satellite} satellite...")
        
        for target_date in target_dates:
            date_str = target_date.strftime('%Y-%m-%d')
            
            # Check if file already exists
            current_satellite_path = os.path.join(dest_folder_path, f"{satellite}_lst16days_{date_str}.tif")
            
            if os.path.exists(current_satellite_path):
                print(f"Skipping LST processing for {satellite} on {date_str}: file already exists.")
                continue
            
            # Find the best file for this date and satellite
            best_file = None
            best_nodata_count = float('inf')
            
            # Look for files in processed data directory
            processed_data_dir = "temp_processed_data"  # Default path, could be made configurable
            
            # Check both L1 and L2 folders
            for data_type in ["landsat_l2", "landsat_l1"]:  # L2 preferred over L1
                data_folder = os.path.join(processed_data_dir, ROI, data_type)
                if not os.path.exists(data_folder):
                    continue
                
                # Look for files matching the satellite and date
                date_str_search = target_date.strftime('%Y%m%d')
                pattern = os.path.join(data_folder, f"*{satellite}*{date_str_search}*.tif")
                matching_files = glob.glob(pattern)
                
                for file_path in matching_files:
                    nodata_count = count_nodata(file_path)
                    if nodata_count < best_nodata_count:
                        best_file = file_path
                        best_nodata_count = nodata_count
            
            if best_file is None:
                print(f"No {satellite} files found for {date_str}")
                continue
            
            print(f"Found {satellite} file: {os.path.basename(best_file)} (nodata: {best_nodata_count})")
            
            # Check if there's a competitor file (other satellite for same date)
            competitor_satellite = "L9" if satellite == "L8" else "L8"
            competitor_path = os.path.join(dest_folder_path, f"{competitor_satellite}_lst16days_{date_str}.tif")
            
            if os.path.exists(competitor_path):
                # A competitor file exists, compare nodata values
                nodata_existing = count_nodata(competitor_path)
                
                if best_nodata_count < nodata_existing:
                    # The new image is better (less nodata), so replace the old one
                    os.remove(competitor_path)
                    shutil.copy2(best_file, current_satellite_path)
                    # Write metadata to the new file
                    timestamp_ms = int(target_date.replace(hour=10, minute=30, second=0).timestamp() * 1000)
                    write_metadata_to_tiff(current_satellite_path, timestamp_ms, f'{satellite}_LST')
                    print(f"Successfully processed LST for {satellite} on {date_str} (replaced {competitor_satellite}).")
                else:
                    # The existing image is better, skip this one
                    print(f"Skipping LST processing for {satellite} on {date_str}: existing {competitor_satellite} image is better.")
            else:
                # No competitor exists, just copy the file
                shutil.copy2(best_file, current_satellite_path)
                # Write metadata to the new file
                timestamp_ms = int(target_date.replace(hour=10, minute=30, second=0).timestamp() * 1000)
                write_metadata_to_tiff(current_satellite_path, timestamp_ms, f'{satellite}_LST')
                print(f"Successfully processed LST for {satellite} on {date_str}.")
    
    print(f"\nLST processing completed for ROI: {ROI}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve Landsat LST data from processed temp data.")
    parser.add_argument("--processed_data_dir", required=True, help="Path to processed temp data directory (e.g., temp_processed_data/D-49-49-A).")
    parser.add_argument("--output_folder", required=True, help="Folder where the LST data will be saved.")
    parser.add_argument("--specific_dates", nargs='*', help="Specific dates to process (format: YYYY-MM-DD). If not provided, dates will be derived from processed data.")
    
    args = parser.parse_args()
    
    try:
        # Convert date strings to datetime objects
        if args.specific_dates:
            target_dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in args.specific_dates]
        else:
            # Try to get dates from processed mapping or use default
            mapping_file = os.path.join(args.processed_data_dir, "processed_mapping.json")
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    mapping_data = json.load(f)
                
                target_dates = []
                # Extract dates from Landsat L1 and L2 data
                for dataset in ['landsat_l1', 'landsat_l2']:
                    if dataset in mapping_data:
                        for file_info in mapping_data[dataset]:
                            try:
                                date_str = file_info.get('date', '')
                                if date_str:
                                    target_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                            except ValueError:
                                continue
            else:
                print("No specific dates provided and could not derive dates from processed mapping.")
                target_dates = []

        if not target_dates:
            print("No target dates available. Exiting.")
            exit(1)

        # Run the offline LST retrieval
        lst_retrive_offline(args.processed_data_dir, args.output_folder, target_dates)
        
    except Exception as e:
        print(f"An error occurred during LST retrieval: {e}")
        import traceback
        traceback.print_exc()