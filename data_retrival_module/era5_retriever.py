import os
import glob
import time
import argparse
from datetime import datetime
from pathlib import Path
import rasterio
from rasterio.warp import transform_bounds
import ee
import requests
import shutil
import zipfile

# --- Global Configuration ---
SKIPPING = True
TARGET_CRS = 'EPSG:4326'
EXPORT_SCALE = 30  # ERA5 native resolution is much coarser, 30m is for alignment

# --- Earth Engine Initialization ---
try:
    ee.Initialize(project='ee-hadat-461702-p4')
except Exception:
    # print("Authenticating to Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project='ee-hadat-461702-p4')

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

def export_ee_image(image, bands, region, out_path, scale, crs=TARGET_CRS, timestamp_ms=None, acquisition_type=None):
    """Exports an Earth Engine image to a local path."""
    temp_dir = os.path.join(os.path.dirname(out_path), 'temp_dl')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        region_geometry = ee.Geometry.Polygon(region, proj=crs, evenOdd=False)
        image = image.clip(region_geometry).select(bands)

        band_info = image.bandNames().getInfo()
        if not band_info:
            print(f"Download failed for {os.path.basename(out_path)}: Image has no bands after clipping.")
            return

        url = image.getDownloadURL({
            'scale': scale, 'region': region, 'fileFormat': 'GeoTIFF', 'crs': crs
        })

        # print(f"Attempting download for {os.path.basename(out_path)}...")
        response = requests.get(url, stream=True, timeout=600)
        response.raise_for_status()

        temp_zip_path = os.path.join(temp_dir, 'download.zip')
        with open(temp_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)

        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif')]

            if not tif_files:
                print(f"Download failed for {os.path.basename(out_path)}: ZIP file did not contain any .tif files.")
                return

            # Extract all tif files so we can combine them into one multi-band image
            for tif in tif_files:
                zip_ref.extract(tif, temp_dir)

            # Order the bands to match the requested `bands` list when possible
            ordered_tifs = []
            for b in bands:
                match = next((t for t in tif_files if b in os.path.basename(t)), None)
                if match:
                    ordered_tifs.append(match)

            # Fallback: if we could not fully match by name, keep the original order
            if len(ordered_tifs) != len(bands):
                ordered_tifs = tif_files

            # Build a multi-band GeoTIFF from the individual single-band files
            first_tif_path = os.path.join(temp_dir, ordered_tifs[0])
            with rasterio.open(first_tif_path) as src0:
                profile = src0.profile
                band_arrays = [src0.read(1)]

            for tif in ordered_tifs[1:]:
                with rasterio.open(os.path.join(temp_dir, tif)) as src:
                    band_arrays.append(src.read(1))

            profile.update(count=len(band_arrays))

            with rasterio.open(out_path, 'w', **profile) as dst:
                for idx, arr in enumerate(band_arrays, start=1):
                    dst.write(arr, idx)

            print(f"Successfully downloaded ERA5 image: {os.path.basename(out_path)}")

            # --- Write Metadata to GeoTIFF ---
            if os.path.exists(out_path):
                try:
                    with rasterio.open(out_path, 'r+') as dst:
                        tags = {}
                        if timestamp_ms:
                            from datetime import datetime
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
        print(f"Download failed for {os.path.basename(out_path)}: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

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
                    rasterio.warp.reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_meta['transform'],
                        dst_crs=ref_meta['crs'],
                        resampling=rasterio.warp.Resampling.bilinear # Good for continuous data like temperature
                    )
            
            # Replace the original source file with the new resampled file
            shutil.move(temp_output_path, source_path)
            # print(f"  > Successfully resampled and replaced {os.path.basename(source_path)}")

    except Exception as e:
        print(f"  > Resampling failed for {source_path}: {e}")

# --- Core ERA5 Function ---

def get_era5_for_date_with_time(target_date, roi_geom, region, out_folder, reference_tif_path, lst_folder):
    """
    Fetches and exports ERA5 Land hourly data for a specific date and time.
    The time is determined from the LST acquisition time to ensure temporal consistency.
    """
    date_str = target_date.strftime('%Y-%m-%d')
    out_path = os.path.join(out_folder, f'ERA5_data_{date_str}.tif')
    
    # 1. Check if the file already exists and skip if it does
    if SKIPPING and os.path.exists(out_path):
        print(f"Skipping ERA5 download for {date_str}: file already exists.")
        resample_to_match_reference(out_path, reference_tif_path)
        return

    # 2. Get LST acquisition time to determine the target hour
    lst_file_path = get_lst_file_for_date(lst_folder, target_date)
    if not lst_file_path:
        print(f"Warning: No LST file found for {date_str}. Using default time (10:30 UTC).")
        target_hour = 10  # Default to 10:30 UTC
    else:
        acquisition_time = get_lst_acquisition_time(lst_file_path)
        if acquisition_time:
            target_hour = acquisition_time.hour
            print(f"Using LST acquisition time for {date_str}: {acquisition_time.strftime('%H:%M:%S')} UTC")
        else:
            target_hour = 10  # Default to 10:30 UTC
            print(f"Using default time for {date_str}: 10:30 UTC")

    try:
        # Use ERA5 hourly data instead of daily aggregated data
        start = ee.Date(target_date)
        end = ee.Date(target_date).advance(1, 'day')
        
        # Get hourly ERA5 data - using the standard ERA5 collection
        era5_hourly_collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
            .filterDate(start, end) \
            .filterBounds(roi_geom)

        if era5_hourly_collection.size().getInfo() == 0:
            print(f"Skipping ERA5 download for {date_str}: No hourly images found.")
            return
        
        # Find the closest hour to the target hour
        def find_closest_hour(image):
            image_time = ee.Date(image.get('system:time_start'))
            image_hour = image_time.get('hour')
            hour_diff = ee.Number(image_hour).subtract(target_hour).abs()
            return image.set('hour_diff', hour_diff)
        
        # Add hour difference to each image and sort by it
        era5_with_diff = era5_hourly_collection.map(find_closest_hour)
        sorted_collection = era5_with_diff.sort('hour_diff')
        
        # Get the image with the closest hour
        best_img = ee.Image(sorted_collection.first())
        
        # Get the actual time of the selected image for metadata
        actual_time = ee.Date(best_img.get('system:time_start'))
        actual_hour = actual_time.get('hour').getInfo()
        time_start_ms = best_img.get('system:time_start').getInfo()
        
        print(f"Selected ERA5 hourly data for {date_str} at {actual_hour}:00 UTC (target was {target_hour}:00)")
        
        # Export the hourly skin temperature data
        export_ee_image(
            image=best_img,
            bands=['skin_temperature'],
            region=region,
            out_path=out_path,
            scale=EXPORT_SCALE,
            crs=TARGET_CRS,
            timestamp_ms=time_start_ms,
            acquisition_type=f'Hourly_{actual_hour:02d}:00'
        )
        time.sleep(0.5)  # Pause to avoid overwhelming the server

        # 3. Resample the newly downloaded image to match the reference
        if os.path.exists(out_path):
            resample_to_match_reference(out_path, reference_tif_path)

    except ee.EEException as e:
        print(f"Download failed for ERA5 {date_str}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {date_str}: {e}")

def get_era5_for_date(target_date, roi_geom, region, out_folder, reference_tif_path):
    """
    Legacy function for backward compatibility.
    Fetches and exports the closest ERA5 Land image for a specific date.
    It downloads two key bands: 2m air temperature and skin temperature.
    After download, it resamples the image to match the reference TIF.
    """
    date_str = target_date.strftime('%Y-%m-%d')
    out_path = os.path.join(out_folder, f'ERA5_data_{date_str}.tif')
    
    # 1. Check if the file already exists and skip if it does
    if SKIPPING and os.path.exists(out_path):
        print(f"Skipping ERA5 download for {date_str}: file already exists.")
        # verify_image(out_path)
        # Even if skipped, ensure it's aligned
        # print(f"  > Checking alignment of existing file: {os.path.basename(out_path)}")
        resample_to_match_reference(out_path, reference_tif_path)
        return

    try:
        # Search within a +/- 1-day window to ensure we find the exact day's image
        start = ee.Date(target_date)
        end = ee.Date(target_date).advance(1, 'day')
        
        era5_collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY') \
            .filterDate(start, end) \
            .filterBounds(roi_geom)

        if era5_collection.size().getInfo() == 0:
            print(f"Skipping ERA5 download for {date_str}: No images found.")
            return
            
        # The daily aggregate should have one image per day, so we can take the first.
        # Use toList(1).get(0) instead of first() for more reliable image extraction
        best_img = ee.Image(era5_collection.toList(1).get(0))
        
        # Copy properties to ensure metadata is preserved through processing.
        # copyProperties returns an ee.Element, so we must cast it back to an ee.Image.
        image_with_props = ee.Image(best_img.copyProperties(best_img, ['system:time_start', 'system:time_end']))
        
        # Get the timestamp for metadata writing
        time_start_ms = image_with_props.get('system:time_start').getInfo()
        
        # print(f"Exporting ERA5 data for {date_str}...")
        export_ee_image(
            image=image_with_props,
            bands=['skin_temperature'],
            region=region,
            out_path=out_path,
            scale=EXPORT_SCALE,
            crs=TARGET_CRS,
            timestamp_ms=time_start_ms,
            acquisition_type='Daily'
        )
        time.sleep(0.5)  # Pause to avoid overwhelming the server

        # 2. Resample the newly downloaded image to match the reference
        if os.path.exists(out_path):
            resample_to_match_reference(out_path, reference_tif_path)

    except ee.EEException as e:
        print(f"Download failed for ERA5 {date_str}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {date_str}: {e}")

# --- Main Execution Logic ---

def main(input_folder, output_folder, specific_dates=None):
    """
    Main function to orchestrate ERA5 data retrieval.
    It can either derive dates from LST files in the input folder
    or use a specific list of provided dates.
    Uses time-aware retrieval based on LST acquisition times.
    """
    overall_start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)

    # --- 1. Get Reference Grid and ROI from the first LST file ---
    all_tifs = glob.glob(os.path.join(input_folder, '*.tif'))
    if not all_tifs:
        print(f"Error: No reference .tif files found in '{input_folder}'. Cannot proceed.")
        return
    reference_tif = all_tifs[0]
    
    try:
        roi_coords = get_roi_coords_from_tif(reference_tif)
        roi_geometry = ee.Geometry.Polygon(roi_coords)
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
        dates_to_process = get_dates_from_filenames(input_folder)

    if not dates_to_process:
        print("No dates to process. Exiting.")
        return

    print(f"Found {len(dates_to_process)} total dates to process for ERA5 retrieval.")

    # --- 3. Process each date using time-aware retrieval ---
    for i, target_date in enumerate(dates_to_process):
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"--- Processing date {i+1}/{len(dates_to_process)}: {date_str} ---")
        get_era5_for_date_with_time(target_date, roi_geometry, roi_coords, output_folder, reference_tif, input_folder)

    total_time = time.time() - overall_start_time
    print(f"\nERA5 retrieval complete. Total time: {total_time:.2f} seconds.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and align ERA5 Land data to match a set of reference GeoTIFFs.")
    parser.add_argument("--input_folder", required=True, help="Folder containing the reference LST .tif files.")
    parser.add_argument("--output_folder", required=True, help="Folder where the downloaded ERA5 data will be saved.")
    
    args = parser.parse_args()
    main(args.input_folder, args.output_folder) 