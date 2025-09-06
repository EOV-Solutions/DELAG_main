import os
import glob
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds, reproject, Resampling
import shutil
import numpy as np

# --- Global Configuration ---
SKIPPING = True
TARGET_CRS = 'EPSG:4326'

# --- Utility Functions (Adapted from new_main.py) ---

def get_lst_acquisition_time(lst_file_path):
    """
    Extracts the acquisition time from LST file metadata.
    Returns a datetime object with the acquisition time.
    """
    try:
        with rasterio.open(lst_file_path) as src:
            # Try to get DATETIME from tags
            tags = src.tags()
            datetime_str = tags.get('DATETIME')
            
            if datetime_str:
                # Parse the datetime string (format: YYYY:MM:DD HH:MM:SS)
                try:
                    # Handle different possible formats
                    if ':' in datetime_str and len(datetime_str.split(':')) >= 6:
                        # Format: YYYY:MM:DD HH:MM:SS
                        dt_parts = datetime_str.split(' ')
                        date_part = dt_parts[0].replace(':', '-')
                        time_part = dt_parts[1]
                        full_datetime_str = f"{date_part} {time_part}"
                        return datetime.strptime(full_datetime_str, '%Y-%m-%d %H:%M:%S')
                    else:
                        # Try other formats
                        return datetime.strptime(datetime_str, '%Y:%m:%d %H:%M:%S')
                except ValueError as e:
                    print(f"Warning: Could not parse DATETIME from {lst_file_path}: {datetime_str}, Error: {e}")
            
            # Fallback: try to get time from system:time_start if available
            # This would require additional metadata that might not be available
            print(f"Warning: No DATETIME metadata found in {lst_file_path}. Using default time (10:30 UTC).")
            
    except Exception as e:
        print(f"Error reading metadata from {lst_file_path}: {e}")
    
    # Default fallback: assume 10:30 UTC (typical Landsat overpass time)
    return None

def get_lst_file_for_date(lst_folder, target_date):
    """
    Finds the LST file for a specific date and returns its path.
    Returns None if no file is found.
    """
    date_str = target_date.strftime('%Y-%m-%d')
    pattern = f"*_{date_str}.tif"
    matching_files = glob.glob(os.path.join(lst_folder, pattern))
    
    if matching_files:
        # Return the first matching file (prefer L9 over L8 if both exist)
        l9_files = [f for f in matching_files if 'L9_' in f]
        if l9_files:
            return l9_files[0]
        return matching_files[0]
    
    return None

def get_roi_coords_from_tif(tif_path):
    """Reads bounds from a TIF and converts them to the target CRS."""
    with rasterio.open(tif_path) as dataset:
        bounds = dataset.bounds
        if dataset.crs.to_string() != TARGET_CRS:
            # print(f"Transforming bounds from {dataset.crs} to {TARGET_CRS}")
            bounds = transform_bounds(dataset.crs, TARGET_CRS, *bounds)
        
        coordinates = [
            [bounds[0], bounds[1]], [bounds[2], bounds[1]],
            [bounds[2], bounds[3]], [bounds[0], bounds[3]],
            [bounds[0], bounds[1]]
        ]
        return [[float(x), float(y)] for x, y in coordinates]

def get_dates_from_filenames(folder_path):
    """Gets a sorted list of unique dates from .tif filenames in a folder."""
    tif_files = glob.glob(os.path.join(folder_path, '*.tif'))
    dates = set()
    for tif in tif_files:
        base = os.path.basename(tif)
        try:
            date_str = base.split('_')[-1].replace('.tif', '')
            date = datetime.strptime(date_str, '%Y-%m-%d')
            dates.add(date)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse date from filename: {base}")
    return sorted(list(dates))

def verify_image(img_path):
    """Verifies that a downloaded image is a valid GeoTIFF."""
    try:
        with rasterio.open(img_path) as src:
            if src.crs and src.width > 0 and src.height > 0:
                # print(f"  Verification successful for {os.path.basename(img_path)} (CRS: {src.crs}, Size: {src.width}x{src.height})")
                return True
        print(f"Verification failed for {os.path.basename(img_path)}: Invalid raster data.")
        return False
    except (rasterio.errors.RasterioIOError, Exception) as e:
        print(f"Verification error for {img_path}: {e}")
        return False

def copy_era5_file(source_path, out_path, timestamp_ms=None, acquisition_type=None):
    """Copies an ERA5 file from processed data to the output location."""
    try:
        # Copy the file
        shutil.copy2(source_path, out_path)
        print(f"Successfully copied ERA5 image: {os.path.basename(out_path)}")

        # --- Write Metadata to GeoTIFF ---
        if os.path.exists(out_path):
            try:
                with rasterio.open(out_path, 'r+') as dst:
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
                print(f"Warning: Failed to write metadata to {out_path}: {e}")

    except Exception as e:
        print(f"Copy failed for {os.path.basename(out_path)}: {e}")

def resample_to_match_reference(source_path, reference_path):
    """
    Resamples a source GeoTIFF to match the metadata (CRS, transform, dimensions)
    of a reference GeoTIFF. This ensures the images are perfectly aligned.
    """
    try:
        with rasterio.open(reference_path) as ref:
            ref_meta = ref.meta.copy()

        with rasterio.open(source_path) as src:
            # Check if resampling is actually needed
            if (src.width == ref_meta['width'] and 
                src.height == ref_meta['height'] and 
                src.transform == ref_meta['transform']):
                # print(f"  > Alignment for {os.path.basename(source_path)} is already correct. No resampling needed.")
                return

            # print(f"  > Resampling {os.path.basename(source_path)} to match reference grid...")
            
            # Update the metadata for the output file
            ref_meta.update({
                'count': src.count, # Match the band count of the source
                'dtype': src.meta['dtype'], # Match the data type of the source
                'nodata': src.nodata # Preserve nodata value
            })

            # Create a temporary file for the resampled output
            temp_output_path = source_path + ".resampled.tif"

            with rasterio.open(temp_output_path, 'w', **ref_meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_meta['transform'],
                        dst_crs=ref_meta['crs'],
                        resampling=Resampling.bilinear # Good for continuous data like temperature
                    )
            
            # Replace the original source file with the new resampled file
            shutil.move(temp_output_path, source_path)
            # print(f"  > Successfully resampled and replaced {os.path.basename(source_path)}")

    except Exception as e:
        print(f"  > Resampling failed for {source_path}: {e}")

# --- Core ERA5 Function ---

def get_era5_for_date_offline(target_date, processed_data_dir, out_folder, reference_tif_path):
    """
    Retrieves ERA5 data from processed temp data for a specific date.
    Finds the closest hourly ERA5 file and copies it to the output folder.
    """
    date_str = target_date.strftime('%Y-%m-%d')
    out_path = os.path.join(out_folder, f'ERA5_data_{date_str}.tif')
    
    # 1. Check if the file already exists and skip if it does
    if SKIPPING and os.path.exists(out_path):
        print(f"Skipping ERA5 copy for {date_str}: file already exists.")
        resample_to_match_reference(out_path, reference_tif_path)
        return

    # 2. Look for ERA5 files in the processed data directory
    era5_folder = os.path.join(processed_data_dir, "era5")
    if not os.path.exists(era5_folder):
        print(f"Warning: ERA5 folder not found in {processed_data_dir}")
        return

    # 3. Find ERA5 files for the target date
    date_pattern = date_str.replace('-', '')
    era5_files = glob.glob(os.path.join(era5_folder, f"*{date_pattern}*.tif"))
    
    if not era5_files:
        print(f"No ERA5 files found for {date_str} in {era5_folder}")
        return

    # 4. Select the best ERA5 file (prefer skin_temperature if available)
    best_file = None
    for file_path in era5_files:
        filename = os.path.basename(file_path)
        if 'skin_temperature' in filename:
            best_file = file_path
            break
    
    # If no skin_temperature file found, use the first available file
    if not best_file:
        best_file = era5_files[0]
    
    print(f"Using ERA5 file: {os.path.basename(best_file)}")

    # 5. Copy the ERA5 file to the output location
    try:
        # Create timestamp for metadata (use target date at 10:00 UTC as default)
        timestamp_ms = int(target_date.replace(hour=10, minute=0, second=0).timestamp() * 1000)
        
        copy_era5_file(
            source_path=best_file,
            out_path=out_path,
            timestamp_ms=timestamp_ms,
            acquisition_type='Processed_ERA5'
        )

        # 6. Resample the copied image to match the reference
        if os.path.exists(out_path):
            resample_to_match_reference(out_path, reference_tif_path)

    except Exception as e:
        print(f"Failed to copy ERA5 file for {date_str}: {e}")

def get_era5_for_date(target_date, processed_data_dir, out_folder, reference_tif_path):
    """
    Legacy function for backward compatibility.
    Retrieves ERA5 data from processed temp data for a specific date.
    """
    return get_era5_for_date_offline(target_date, processed_data_dir, out_folder, reference_tif_path)

# --- Main Execution Logic ---

def main_offline(processed_data_dir, output_folder, specific_dates=None):
    """
    Main function to orchestrate ERA5 data retrieval from processed temp data.
    It can either derive dates from LST files in the input folder
    or use a specific list of provided dates.
    """
    overall_start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)

    # --- 1. Get Reference Grid from the first LST file ---
    lst_folder = os.path.join(processed_data_dir, "lst")
    if not os.path.exists(lst_folder):
        print(f"Error: LST folder not found in '{processed_data_dir}'. Cannot proceed.")
        return
    
    all_tifs = glob.glob(os.path.join(lst_folder, '*.tif'))
    if not all_tifs:
        print(f"Error: No reference .tif files found in '{lst_folder}'. Cannot proceed.")
        return
    reference_tif = all_tifs[0]
    
    try:
        roi_coords = get_roi_coords_from_tif(reference_tif)
        print(f"Successfully defined ROI from reference: {os.path.basename(reference_tif)}")
    except Exception as e:
        print(f"Fatal: Could not define ROI from reference TIF '{reference_tif}'. Error: {e}")
        return

    # --- 2. Determine which dates to process ---
    if specific_dates:
        # Use the provided list of dates
        dates_to_process = [datetime.strptime(d, '%Y-%m-%d') for d in specific_dates]
        print(f"Processing a specific list of {len(dates_to_process)} provided dates.")
    else:
        # Fallback to deriving dates from filenames if none are provided
        print("No specific dates provided. Deriving dates from LST filenames...")
        dates_to_process = get_dates_from_filenames(lst_folder)

    if not dates_to_process:
        print("No dates to process. Exiting.")
        return

    print(f"Found {len(dates_to_process)} total dates to process for ERA5 retrieval.")

    # --- 3. Process each date using offline retrieval ---
    for i, target_date in enumerate(dates_to_process):
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"--- Processing date {i+1}/{len(dates_to_process)}: {date_str} ---")
        get_era5_for_date_offline(target_date, processed_data_dir, output_folder, reference_tif)

    total_time = time.time() - overall_start_time
    print(f"\nERA5 retrieval complete. Total time: {total_time:.2f} seconds.")

def main(input_folder, output_folder, specific_dates=None):
    """
    Legacy main function for backward compatibility.
    """
    print("Warning: Using legacy main function. Consider using main_offline() for processed data.")
    # This would need to be updated to work with the new offline approach
    # For now, just redirect to the offline version
    main_offline(input_folder, output_folder, specific_dates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrieve and align ERA5 Land data from processed temp data to match a set of reference GeoTIFFs.")
    parser.add_argument("--processed_data_dir", required=True, help="Path to processed temp data directory (e.g., temp_processed_data/D-49-49-A).")
    parser.add_argument("--output_folder", required=True, help="Folder where the ERA5 data will be saved.")
    parser.add_argument("--specific_dates", nargs='*', help="Specific dates to process (format: YYYY-MM-DD). If not provided, dates will be derived from LST files.")
    
    args = parser.parse_args()
    main_offline(args.processed_data_dir, args.output_folder, args.specific_dates) 