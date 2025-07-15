import ee
import os
import requests
import time
import math
# import json # Not used in this version
import shutil
import tempfile
import zipfile
import rasterio
from rasterio.profiles import DefaultGTiffProfile
from rasterio.warp import transform_bounds
from datetime import datetime, timedelta
import numpy as np
from glob import glob

# =============================================================================
# HYPERPARAMETERS / CONFIGURATION
# =============================================================================
# IMPORTANT: User needs to set their actual base data directory
# This directory should contain subfolders, one for each ROI.
# Each ROI subfolder must contain an 'lst' subfolder with .tif LST files.
# Example structure:
# /mnt/user_data/all_rois/
#  |- ROI_A/
#  |  |- lst/
#  |  |  |- LST_data_YYYY-MM-DD.tif
#  |  |  |- LST_data_YYYY-MM-DD.tif
#  |  |- (s2_b2b3b4b8_images/ will be created here by this script)
#  |- ROI_B/
#  |  |- lst/
#  |  |  |- scene1_LST_YYYY-MM-DD.tif
#  |  |- (s2_b2b3b4b8_images/ will be created here by this script)
DEFAULT_BASE_DATA_DIR = "/mnt/ssd1tb/code/nhatvm/data_lst_16days" # PLEASE UPDATE THIS PATH

S2_OUTPUT_SUBFOLDER_NAME = "s2_images"
BANDS_TO_PROCESS = ['B2', 'B3', 'B4', 'B8']

# Earth Engine Project ID (User might need to set this explicitly)
EE_PROJECT_ID = 'ee-maihadat2022' # Taken from download_s2_data.py

# Limit the number of ROI folders to process. 
# Set to a positive integer to limit (e.g., 5 for the first 5 ROIs found).
# Set to 0, None, or a negative number to process all ROIs.
ROI_PROCESSING_LIMIT = 50

# =============================================================================
# EARTH ENGINE INITIALIZATION
# =============================================================================
try:
    ee.Initialize(project=EE_PROJECT_ID)
    print("Earth Engine Initialized Successfully.")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")
    print(f"Please ensure you have authenticated GEE (e.g., via 'earthengine authenticate') and set your project ID ('{EE_PROJECT_ID}' or another valid one).")
    exit()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def create_ee_rectangle_from_bounds(left, bottom, right, top, crs="EPSG:4326"):
    """Creates an ee.Geometry.Rectangle from bounds, assuming EPSG:4326 for GEE."""
    # For planar geometries (geodesic=False with geographic CRS), GEE often expects evenOdd=True.
    return ee.Geometry.Rectangle([left, bottom, right, top], proj=crs, geodesic=False, evenOdd=True)

# =============================================================================
# EARTH ENGINE PROCESSING FUNCTIONS (Adapted from download_s2_data.py)
# =============================================================================
def get_sentinel_collection(start_date_ee, end_date_ee, roi):
    """
    Loads the Sentinel-2 collection, applies initial filters and cloud masking.
    Includes an 8-day buffer on date filtering for robustness.
    """
    s_date_filter = start_date_ee.advance(-8, 'day')
    e_date_filter = end_date_ee.advance(8, 'day')

    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    qa_band = 'cs'
    clear_threshold = 0.5  # CS+ threshold for clear pixels

    sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate(s_date_filter, e_date_filter)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 85)) # Pre-filter
                 .linkCollection(cs_plus, [qa_band]))

    cloud_masked = sentinel2.map(
        lambda img: img.updateMask(img.select(qa_band).gte(clear_threshold)).clip(roi)
    )
    return cloud_masked

def get_s2_image_collection(start_date_str, end_date_str, roi):
    """
    Fetches Sentinel-2 images for BANDS_TO_PROCESS within the date range.
    """
    start_date_ee = ee.Date(start_date_str)
    end_date_ee = ee.Date(end_date_str)

    s2_collection_masked = get_sentinel_collection(start_date_ee, end_date_ee, roi)

    def select_bands(image):
        return image.select(BANDS_TO_PROCESS).toFloat() # .toFloat() for GEE processing

    processed_collection = s2_collection_masked.map(select_bands)
    return processed_collection

def download_and_merge_s2_images(image_collection, output_folder_roi, roi_geom, roi_identifier_str):
    """
    Downloads each image from the collection, bands specified in BANDS_TO_PROCESS
    separately, then merges them into a single GeoTIFF.
    Filenames will include HHmmss to differentiate same-day acquisitions.
    """
    try:
        image_list = image_collection.toList(image_collection.size())
        num_images = image_collection.size().getInfo()
    except ee.EEException as e:
        print(f"  Error fetching image collection size for ROI {roi_identifier_str}: {e}")
        return

    if not os.path.exists(output_folder_roi):
        os.makedirs(output_folder_roi)

    print(f"  Found {num_images} cloud-masked images for ROI '{roi_identifier_str}'. Attempting downloads and merging to: {output_folder_roi}")

    if num_images == 0:
        print(f"    No images in collection for ROI '{roi_identifier_str}'. Skipping download.")
        return

    bands_to_download = BANDS_TO_PROCESS

    for i in range(num_images):
        image = ee.Image(image_list.get(i))
        image_temp_dir = None
        all_single_bands_downloaded = True
        temp_band_files = {} # {band_name: temp_file_path}

        try:
            # Get timestamp for unique naming, including time
            timestamp_obj = ee.Date(image.get('system:time_start'))
            # Format: YYYY-MM-DD_HHmmss (Java SimpleDateFormat)
            datetime_for_filename = timestamp_obj.format('YYYY-MM-dd_HHmmss').getInfo()
            
            file_prefix_merged = f"s2_{len(bands_to_download)}bands_"
            base_name_for_tif = f"{file_prefix_merged}{datetime_for_filename}"
            output_filename_candidate = f"{base_name_for_tif}.tif"
            
            final_merged_path = os.path.join(output_folder_roi, output_filename_candidate)
            
            # Handle potential (though rare with HHmmss) filename collisions with a version counter
            suffix_counter = 1
            while os.path.exists(final_merged_path):
                output_filename_candidate = f"{base_name_for_tif}_v{suffix_counter}.tif"
                final_merged_path = os.path.join(output_folder_roi, output_filename_candidate)
                suffix_counter += 1

            print(f"    Processing S2 image for datetime: {datetime_for_filename} (Image {i+1}/{num_images}) for ROI '{roi_identifier_str}' -> Saving as {os.path.basename(final_merged_path)}")

            image_temp_dir = tempfile.mkdtemp(prefix=f"s2_{roi_identifier_str}_{datetime_for_filename}_")

            for band_name in bands_to_download:
                # print(f"      Downloading band: {band_name}...") # Verbose, can be enabled if needed
                try:
                    band_image_to_download = image.select([band_name])
                    params = {
                        'bands': [band_name],
                        'scale': 10, # Nominal resolution for B2,B3,B4,B8
                        'region': roi_geom,
                        'fileFormat': 'ZIP',
                        'maxPixels': 1e13
                    }
                    download_url = band_image_to_download.getDownloadURL(params)

                    # Use datetime_for_filename for temp files too for consistency, though not strictly necessary here
                    temp_band_filename_stem = f"temp_{datetime_for_filename}_{band_name}"
                    final_band_tif_path = os.path.join(image_temp_dir, f"{temp_band_filename_stem}.tif")
                    temp_zip_path = os.path.join(image_temp_dir, f"{temp_band_filename_stem}.zip")

                    response = requests.get(download_url, timeout=300) # 5-minute timeout
                    response.raise_for_status()

                    with open(temp_zip_path, 'wb') as f:
                        f.write(response.content)

                    extracted_successfully = False
                    with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                        band_extract_dir = os.path.join(image_temp_dir, f"extract_{band_name}")
                        os.makedirs(band_extract_dir, exist_ok=True)
                        zip_ref.extractall(band_extract_dir)

                        tif_files_in_zip = [
                            os.path.join(band_extract_dir, f_name)
                            for f_name in os.listdir(band_extract_dir)
                            if f_name.lower().endswith('.tif')
                        ]
                        if tif_files_in_zip:
                            shutil.move(tif_files_in_zip[0], final_band_tif_path)
                            temp_band_files[band_name] = final_band_tif_path
                            # print(f"        Successfully downloaded and extracted {band_name} to {final_band_tif_path}") # Verbose
                            extracted_successfully = True
                        else:
                            print(f"        Error: No TIFF file found in ZIP archive for band {band_name}, datetime: {datetime_for_filename}")

                    if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
                    if os.path.exists(band_extract_dir): shutil.rmtree(band_extract_dir)

                    if not extracted_successfully:
                        all_single_bands_downloaded = False; break
                except requests.exceptions.Timeout:
                    print(f"        Timeout error downloading band {band_name} for ROI {roi_identifier_str}, datetime {datetime_for_filename}.")
                    all_single_bands_downloaded = False; break
                except requests.exceptions.RequestException as e_req:
                    print(f"        Request error downloading band {band_name} for ROI {roi_identifier_str}, datetime {datetime_for_filename}: {e_req}")
                    all_single_bands_downloaded = False; break
                except ee.EEException as e_ee:
                    print(f"        Earth Engine error processing band {band_name} for ROI {roi_identifier_str}, datetime {datetime_for_filename}: {e_ee}")
                    all_single_bands_downloaded = False; break
                except Exception as e_gen:
                    print(f"        An unexpected error occurred downloading/extracting band {band_name} for ROI {roi_identifier_str}, datetime {datetime_for_filename}: {e_gen}")
                    all_single_bands_downloaded = False; break
                time.sleep(0.5) # Small pause between band downloads

            if all_single_bands_downloaded and len(temp_band_files) == len(bands_to_download):
                print(f"      All bands for {datetime_for_filename} downloaded for {roi_identifier_str}. Merging to {os.path.basename(final_merged_path)}...")
                # final_merged_path is already determined and unique

                with rasterio.open(temp_band_files[bands_to_download[0]]) as src:
                    profile = src.profile.copy()
                    profile['count'] = len(bands_to_download)
                    profile['dtype'] = rasterio.float64 
                    profile['driver'] = 'GTiff'
                    profile['nodata'] = -9999.0
                    if 'compress' in profile: del profile['compress']
                    if 'blockxsize' in profile: del profile['blockxsize']
                    if 'blockysize' in profile: del profile['blockysize']
                    profile['tiled'] = False

                with rasterio.open(final_merged_path, 'w', **profile) as dst:
                    for idx, band_name in enumerate(bands_to_download):
                        with rasterio.open(temp_band_files[band_name]) as src_band:
                            data = src_band.read(1).astype(rasterio.float64)
                            dst.write(data, idx + 1)
                print(f"        Successfully merged bands into {final_merged_path}")
            else:
                print(f"      Failed to download all required bands for {datetime_for_filename} for ROI {roi_identifier_str}. Skipping merge.")

        except ee.EEException as e:
            print(f"    Earth Engine error processing S2 image {i+1}/{num_images} (datetime {datetime_for_filename if 'datetime_for_filename' in locals() else 'unknown'}) for ROI {roi_identifier_str}: {e}")
        except Exception as e:
            print(f"    An unexpected error occurred for S2 image {i+1}/{num_images} (datetime {datetime_for_filename if 'datetime_for_filename' in locals() else 'unknown'}) for ROI {roi_identifier_str}: {e}")
        finally:
            if image_temp_dir and os.path.exists(image_temp_dir):
                try:
                    shutil.rmtree(image_temp_dir)
                except Exception as e_clean:
                    print(f"      Error removing temporary S2 download directory {image_temp_dir}: {e_clean}")
        time.sleep(1) # Pause between processing different S2 image dates
    print(f"  Finished all S2 downloads for ROI '{roi_identifier_str}'.")

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================
def process_all_rois(base_dir, s2_output_subdir_name):
    """
    Processes all ROI folders within base_dir. For each, identifies LST files,
    determines time range and geometry, then downloads corresponding Sentinel-2 imagery.
    """
    if not os.path.isdir(base_dir):
        print(f"Error: Base data directory '{base_dir}' not found. Please check DEFAULT_BASE_DATA_DIR.")
        return

    # Find all subdirectories in base_dir; these are assumed to be ROI folders.
    roi_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d))]

    if not roi_folders:
        print(f"No ROI subdirectories found in '{base_dir}'. Exiting.")
        return

    print(f"Found {len(roi_folders)} potential ROI folders in '{base_dir}'.")

    rois_processed_count = 0
    for roi_folder_path in roi_folders:
        if ROI_PROCESSING_LIMIT and ROI_PROCESSING_LIMIT > 0 and rois_processed_count >= ROI_PROCESSING_LIMIT:
            print(f"\nReached ROI processing limit of {ROI_PROCESSING_LIMIT}. Stopping further ROI processing.")
            break

        roi_name = os.path.basename(roi_folder_path)
        print(f"\n===== Processing ROI {rois_processed_count + 1}/{len(roi_folders)} (Limit: {ROI_PROCESSING_LIMIT if ROI_PROCESSING_LIMIT and ROI_PROCESSING_LIMIT > 0 else 'All'}): {roi_name} (Path: {roi_folder_path}) =====")
        rois_processed_count += 1 # Increment after deciding to process this ROI

        lst_folder_path = os.path.join(roi_folder_path, "lst")
        if not os.path.isdir(lst_folder_path):
            print(f"  'lst' subfolder not found in {roi_name}. Skipping this ROI.")
            continue

        lst_files = glob(os.path.join(lst_folder_path, "*.tif"))
        if not lst_files:
            print(f"  No .tif files found in 'lst' subfolder for {roi_name}. Skipping this ROI.")
            continue
        print(f"  Found {len(lst_files)} .tif files in 'lst' folder for {roi_name}.")

        # 1. Determine Time Range from LST filenames
        lst_dates = []
        for lst_file in lst_files:
            filename_no_ext = os.path.splitext(os.path.basename(lst_file))[0]
            try: # Attempt YYYY-MM-DD format, often at end or after '_'
                date_part = filename_no_ext.replace('-', '').replace('_', '') # Clean separators
                # Find the last 8 digits that could be a date
                potential_date_str = None
                for i in range(len(date_part) - 7):
                    substring = date_part[i:i+8]
                    if substring.isdigit():
                        potential_date_str = substring # Likely YYYYMMDD

                if potential_date_str:
                     dt_obj = datetime.strptime(potential_date_str, '%Y%m%d')
                     lst_dates.append(dt_obj)
                else: # Try parsing last part if it matches YYYY-MM-DD after splitting by '_'
                    date_parts_split = filename_no_ext.split('_')
                    if date_parts_split:
                        try_date_str = date_parts_split[-1]
                        dt_obj = datetime.strptime(try_date_str, '%Y-%m-%d')
                        lst_dates.append(dt_obj)
            except ValueError:
                # Fallback for YYYYMMDD if not caught by primary logic
                try:
                    if len(filename_no_ext) >= 8 and filename_no_ext[-8:].isdigit():
                        dt_obj = datetime.strptime(filename_no_ext[-8:], '%Y%m%d')
                        lst_dates.append(dt_obj)
                    else:
                        print(f"    Warning: Could not parse date robustly from LST filename: {os.path.basename(lst_file)}.")
                except ValueError:
                     print(f"    Warning: Could not parse date (final attempt) from LST filename: {os.path.basename(lst_file)}.")


        if not lst_dates:
            print(f"  Could not determine any valid dates from LST filenames in {roi_name}. Skipping this ROI.")
            continue

        min_lst_date = min(lst_dates)
        max_lst_date = max(lst_dates)
        roi_start_date_str = min_lst_date.strftime('%Y-%m-%d')
        roi_end_date_str = max_lst_date.strftime('%Y-%m-%d')
        print(f"  LST Date Range for {roi_name}: {roi_start_date_str} to {roi_end_date_str}")

        # 2. Determine ROI Geometry from the first LST GeoTIFF
        first_lst_file_path = lst_files[0] # Use the first LST file for geometry
        try:
            with rasterio.open(first_lst_file_path) as src:
                lst_bounds = src.bounds # (left, bottom, right, top)
                lst_crs = src.crs

                print(f"  Using LST file '{os.path.basename(first_lst_file_path)}' for ROI geometry.")
                print(f"    LST Bounds: {lst_bounds}, CRS: {lst_crs}")

                target_crs_str = "EPSG:4326" # GEE standard for ee.Geometry
                if lst_crs and lst_crs.is_geographic and lst_crs.to_string().upper() == target_crs_str:
                    final_bounds = lst_bounds
                    print(f"    LST file is already in {target_crs_str}.")
                elif lst_crs:
                    print(f"    Transforming LST bounds from {lst_crs.to_string()} to {target_crs_str} for GEE.")
                    final_bounds = transform_bounds(lst_crs, target_crs_str, *lst_bounds)
                else:
                    print(f"    Warning: LST file '{os.path.basename(first_lst_file_path)}' has no CRS defined. Assuming its coordinates are {target_crs_str}.")
                    final_bounds = lst_bounds # Proceed with caution

                roi_ee_geom = create_ee_rectangle_from_bounds(
                    final_bounds[0], final_bounds[1], final_bounds[2], final_bounds[3], crs=target_crs_str
                )
                # print(f"  Created ee.Geometry for ROI {roi_name}: MinLon={final_bounds[0]}, MinLat={final_bounds[1]}, MaxLon={final_bounds[2]}, MaxLat={final_bounds[3]}")
                # To get info for debugging: print(f"  ROI Geometry for GEE (first 5 coords): {roi_ee_geom.coordinates().get(0).getInfo()[:5]}")

        except Exception as e:
            print(f"  Error reading LST file '{os.path.basename(first_lst_file_path)}' for geometry: {e}. Skipping ROI {roi_name}.")
            continue

        # 3. Create S2 output subfolder
        s2_output_path = os.path.join(roi_folder_path, s2_output_subdir_name)
        if not os.path.exists(s2_output_path):
            os.makedirs(s2_output_path)
        print(f"  Sentinel-2 output subfolder: {s2_output_path}")

        # 4. Get Sentinel-2 image collection
        print(f"  Fetching Sentinel-2 collection for bands {BANDS_TO_PROCESS}...")
        try:
            s2_collection = get_s2_image_collection(roi_start_date_str, roi_end_date_str, roi_ee_geom)
            collection_size = s2_collection.size().getInfo()
            if collection_size == 0:
                print(f"    No Sentinel-2 images found for {roi_name} in date range {roi_start_date_str} - {roi_end_date_str} and specified ROI. Skipping S2 download for this ROI.")
                continue
            print(f"    Found {collection_size} potential Sentinel-2 images for {roi_name} before download attempt.")
        except ee.EEException as e:
            print(f"    Earth Engine error getting S2 collection for {roi_name}: {e}. Skipping S2 download.")
            continue
        except Exception as e:
            print(f"    Unexpected error getting S2 collection for {roi_name}: {e}. Skipping S2 download.")
            continue

        # 5. Download and merge S2 images
        download_and_merge_s2_images(s2_collection, s2_output_path, roi_ee_geom, roi_name)

        print(f"  Finished S2 processing for ROI: {roi_name}")
        time.sleep(1) # Pause between processing different ROIs to be kind to GEE

    print("===== All ROIs processed. =====")

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("Starting Sentinel-2 data processing script.")
    
    # --- Dummy Data Setup for Testing (Optional) ---
    # This section creates a mock directory structure if the DEFAULT_BASE_DATA_DIR
    # doesn't seem to have it, to allow the script to run without manual setup.
    # For real execution, ensure DEFAULT_BASE_DATA_DIR points to your actual data.
    # The dummy .tif files created are EMPTY and will cause rasterio to fail when
    # reading bounds/CRS. Replace with small, valid GeoTIFFs for thorough testing.
    
    # test_base_dir = DEFAULT_BASE_DATA_DIR # Use the configured path
    # test_roi1_lst_path = os.path.join(test_base_dir, "ROI_Test_Alpha", "lst")
    # test_roi2_lst_path = os.path.join(test_base_dir, "ROI_Test_Beta", "lst")

    # if not (os.path.exists(test_roi1_lst_path) and os.path.exists(test_roi2_lst_path)):
    #     print(f"Attempting to create dummy data structure in '{test_base_dir}' for testing purposes.")
    #     print("NOTE: Dummy .tif files will be EMPTY and likely cause errors with CRS/bounds reading.")
    #     print("For full testing, please use actual LST GeoTIFFs in this structure.")
        
    #     os.makedirs(test_roi1_lst_path, exist_ok=True)
    #     os.makedirs(test_roi2_lst_path, exist_ok=True)
        
    #     # Create empty dummy LST tif files
    #     dummy_files_roi1 = [
    #         os.path.join(test_roi1_lst_path, "LST_ProductA_2023-01-01.tif"),
    #         os.path.join(test_roi1_lst_path, "LST_ProductA_2023-01-15.tif")
    #     ]
    #     dummy_files_roi2 = [
    #         os.path.join(test_roi2_lst_path, "RegionX_LST_Field1_2022-11-05.tif"),
    #         os.path.join(test_roi2_lst_path, "RegionX_LST_Field1_2022-11-20.tif")
    #     ]
        
    #     for f_path in dummy_files_roi1 + dummy_files_roi2:
    #         if not os.path.exists(f_path):
    #             try:
    #                 open(f_path, 'a').close() # Create empty file
    #                 print(f"  Created dummy file: {f_path}")
    #             except Exception as e:
    #                 print(f"  Could not create dummy file {f_path}: {e}")
    #     print("Dummy structure attempt complete.")
    # else:
    #     print(f"Found existing test structure or part of it in '{test_base_dir}'. Skipping dummy creation.")

    # --- Run Main Processing ---
    # Ensure DEFAULT_BASE_DATA_DIR at the top of the script is set to your main data folder
    # containing the ROI subdirectories.
    process_all_rois(base_dir=DEFAULT_BASE_DATA_DIR,
                     s2_output_subdir_name=S2_OUTPUT_SUBFOLDER_NAME)

    print("Script finished.")
