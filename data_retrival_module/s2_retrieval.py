import ee
import os
import requests
import tempfile
import zipfile
import shutil
import time
import json
import logging
import rasterio
from dotenv import load_dotenv
import numpy as np
from datetime import datetime
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ndvi_retrieval.log')
    ]
)

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

# =============================================================================
# FUNCTIONS
# =============================================================================
def coor_to_geometry(json_file: str = None, geojson_data: dict = None):
    """
    Loads coordinates from a GeoJSON file or dictionary and converts to ee.Geometry.Polygon.
    Accepts either a file path or a dictionary as input.
    """
    if json_file and geojson_data:
        raise ValueError("Provide either json_file or geojson_data, not both.")
    if not (json_file or geojson_data):
        raise ValueError("Either json_file or geojson_data must be provided.")

    data_source = json_file if json_file else "provided dictionary"
    try:
        if json_file:
            with open(json_file, 'r') as f:
                geojson = json.load(f)
        else:
            geojson = geojson_data

        # Handle different GeoJSON types
        if geojson['type'] == 'FeatureCollection':
            coor_list = geojson['features'][0]['geometry']['coordinates']
        elif geojson['type'] == 'Feature':
            coor_list = geojson['geometry']['coordinates']
        elif geojson['type'] == 'Polygon':
            coor_list = geojson['coordinates']
        else:
            raise ValueError(f"Unsupported GeoJSON type: {geojson['type']}")
        logging.info(f"Successfully loaded ROI geometry from {data_source}")
        return ee.Geometry.Polygon(coor_list)
    except FileNotFoundError:
        logging.critical(f"ROI JSON file not found: {json_file}")
        raise
    except json.JSONDecodeError as e:
        logging.critical(f"Error decoding JSON from {json_file}: {e}")
        raise
    except Exception as e:
        logging.critical(f"An unexpected error occurred while processing ROI geometry from {data_source}: {e}")
        raise
def get_sentinel_collection(start_date, end_date, roi):
    """
    Loads the Sentinel-2 collection, applies initial filters and cloud masking.
    """
    # logging.info(f"Fetching Sentinel-2 collection for dates {start_date.format('YYYY-MM-dd').getInfo()} to {end_date.format('YYYY-MM-dd').getInfo()}")
    
    # Create a 9-day window (+/- 4 days) around the target date to find the best image.
    # This increases the chance of finding a cloud-free image for the composite.
    s_date = start_date.advance(-2, 'day')
    e_date = end_date.advance(2, 'day')
    
    cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    qa_band = 'cs'
    clear_threshold = 0.5 # Pixels with cloud score >= 0.5 are considered cloudy
    
    sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(roi)
                 .filterDate(s_date, e_date)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 85)) # Initial broad cloud filter
                 .linkCollection(cs_plus, [qa_band]))

    # Apply cloud mask using Cloud Score+ (masking out cloudy pixels)
    sentinel_masked_cloud = sentinel2.map(
        lambda img: img.updateMask(img.select(qa_band).gte(clear_threshold)).clip(roi)
    )
    # logging.info(f"Initial Sentinel-2 collection size (before cloud masking): {sentinel2.size().getInfo()}")
    # logging.info(f"Sentinel-2 collection size after cloud masking: {sentinel_masked_cloud.size().getInfo()}")
    return sentinel_masked_cloud


def download_s2_bands(image_collection, big_folder, roi, file_prefix, roi_name, folder_name):
    """
    Downloads each S2 composite image from the collection by fetching each band
    separately. This handles direct image downloads and zipped single-band
    downloads from GEE to avoid size limits. The individual band GeoTIFFs
    are then merged into a single multi-band file.
    """
    image_list = image_collection.toList(image_collection.size())
    size = image_collection.size().getInfo()
    out_folder = os.path.join(big_folder, roi_name, folder_name)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        # logging.info(f"Created output folder for downloads: {out_folder}")

    bands_to_export = ['B4', 'B3', 'B2', 'B8']  # Red, Green, Blue, NIR

    for i in range(size):
        image = ee.Image(image_list.get(i))

        # Check for placeholder images
        exclude_date_value = ee.Date('1900-01-01').millis().getInfo()
        if image.bandNames().length().eq(0).getInfo() or \
           image.get('system:time_start').getInfo() == exclude_date_value:
            logging.debug(f"Skipping empty or placeholder image at index {i}.")
            continue

        date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        # Get the timestamp for metadata writing
        time_start_ms = image.get('system:time_start').getInfo()
        final_tif_path = os.path.join(out_folder, f"{file_prefix}{date_str}.tif")

        if os.path.exists(final_tif_path):
            logging.info(f"Skipping S2 download for {date_str}: file already exists.")
            continue
        
        temp_dir = None
        max_retries = 3
        download_success = False

        for attempt in range(max_retries):
            temp_dir = tempfile.mkdtemp()
            downloaded_band_paths = []
            all_bands_downloaded = True
            
            try:
                # logging.info(f"Starting per-band download for {date_str} (attempt {attempt+1}/{max_retries})...")
                for band_name in bands_to_export:
                    band_downloaded_successfully = False
                    image_to_download = image.select(band_name)
                    params = {
                        'scale': 30,
                        'region': roi,
                        'fileFormat': 'GeoTIFF',
                        'maxPixels': 1e13,
                        'crs': 'EPSG:4326'
                    }
                    download_url = image_to_download.getDownloadURL(params)
                    
                    response = requests.get(download_url, timeout=300)
                    response.raise_for_status()

                    content_type = response.headers.get('Content-Type', '')
                    is_image = 'image' in content_type or 'geotiff' in content_type or 'octet-stream' in content_type
                    is_zip = 'application/zip' in content_type

                    if response.status_code == 200:
                        if is_image:
                            band_tif_path = os.path.join(temp_dir, f"{band_name}.tif")
                            with open(band_tif_path, 'wb') as f:
                                f.write(response.content)
                            downloaded_band_paths.append(band_tif_path)
                            logging.debug(f"Successfully downloaded band {band_name} as direct image.")
                            band_downloaded_successfully = True
                        
                        elif is_zip:
                            # logging.info(f"Received ZIP file for band {band_name}, unpacking...")
                            zip_path = os.path.join(temp_dir, f"{band_name}.zip")
                            with open(zip_path, 'wb') as f:
                                f.write(response.content)
                            
                            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                                tif_in_zip = [f for f in zip_ref.namelist() if f.lower().endswith('.tif')]
                                if tif_in_zip:
                                    extracted_name = tif_in_zip[0]
                                    zip_ref.extract(extracted_name, temp_dir)
                                    
                                    # Rename extracted file for consistency
                                    original_path = os.path.join(temp_dir, extracted_name)
                                    final_band_path = os.path.join(temp_dir, f"{band_name}.tif")
                                    os.rename(original_path, final_band_path)
                                    
                                    downloaded_band_paths.append(final_band_path)
                                    logging.debug(f"Successfully extracted band {band_name} from ZIP.")
                                    band_downloaded_successfully = True
                                else:
                                    logging.warning(f"No TIFF file found in ZIP for band {band_name}.")
                    
                    if not band_downloaded_successfully:
                        logging.warning(f"Download for band {band_name} failed. Status: {response.status_code}, Content-Type: '{content_type}'.")
                        try:
                            logging.warning(f"GEE Error Payload: {response.json()}")
                        except json.JSONDecodeError:
                            logging.warning(f"GEE Response Content (first 500 chars): {response.text[:500]}")
                        all_bands_downloaded = False
                        break # Exit band loop to trigger a retry for the whole image
                
                if all_bands_downloaded:
                    # logging.info(f"All bands for {date_str} downloaded. Merging into single GeoTIFF...")
                    with rasterio.open(downloaded_band_paths[0]) as src0:
                        profile = src0.profile
                        band_arrays = [src0.read(1)]
                    
                    for tif_path in downloaded_band_paths[1:]:
                        with rasterio.open(tif_path) as src:
                            band_arrays.append(src.read(1))
                    
                    profile.update(count=len(band_arrays), nodata=np.nan)

                    with rasterio.open(final_tif_path, 'w', **profile) as dst:
                        for idx, arr in enumerate(band_arrays, start=1):
                            dst.write(arr, idx)
                    
                    # Write metadata to the final file
                    write_metadata_to_tiff(final_tif_path, time_start_ms, 'S2_8days')
                    
                    logging.info(f"Successfully downloaded S2 image for {date_str}.")
                    download_success = True
                    break

            except requests.exceptions.RequestException as e:
                logging.warning(f"Request error for {date_str} (attempt {attempt+1}/{max_retries}): {e}.")
                all_bands_downloaded = False
            except Exception as e:
                logging.error(f"Unhandled error during download or processing for {date_str} (attempt {attempt+1}/{max_retries}): {e}", exc_info=True)
                all_bands_downloaded = False
            finally:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            
            if not download_success and attempt < max_retries - 1:
                # logging.info(f"Retrying download for {date_str}...")
                time.sleep(5 * (attempt + 1))
        
        if not download_success:
            logging.error(f"Failed to download and process S2 image for {date_str} after {max_retries} attempts.")


def main_s2_retrieval(target_composite_dates: list, roi, roi_name, big_folder):
    """
    Main function to orchestrate S2 RGB-NIR retrieval and local storage
    for a list of specific composite dates.
    """

    # logging.info(f"--- Starting S2 retrieval process for ROI '{roi_name}' for {len(target_composite_dates)} specific dates ---")
    
    exclude_date = ee.Date('1900-01-01').millis() # Placeholder for empty composites

    # ---------------------------------------------------------------------------
    # Run processing steps for each target composite date
    # ---------------------------------------------------------------------------
    all_valid_composites = []
    for center_date_ee in target_composite_dates:
        composite_start = center_date_ee.advance(0, 'day')
        composite_end = center_date_ee.advance(0, 'day')
        
        try:
            # 1. Load Sentinel-2 collection and mask clouds for the 8-day window.
            sentinel_collection = get_sentinel_collection(composite_start, composite_end, roi)
        except Exception as e:
            logging.critical(f"Failed to get Sentinel-2 collection for {center_date_ee.format('YYYY-MM-dd').getInfo()}: {e}")
            continue # Skip to next date

        # Create a single 8-day composite for the current center_date_ee
        single_composite_image = ee.Algorithms.If(
            sentinel_collection.size().gt(0),
            (sentinel_collection.median()
             .select(['B4', 'B3', 'B2', 'B8']) # R, G, B, NIR
             .unmask(-100) # Unmask with a NoData value of -100
             .set('system:time_start', center_date_ee.millis())),
            ee.Image().set('system:time_start', exclude_date) # Placeholder for empty periods
        )
        all_valid_composites.append(single_composite_image)

    s2_composites_collection = ee.ImageCollection(all_valid_composites)

    # Filter out empty placeholders before attempting download
    valid_s2_composites = s2_composites_collection.filter(ee.Filter.neq('system:time_start', exclude_date))
    valid_composites_count = valid_s2_composites.size().getInfo()
    # logging.info(f"Found {valid_composites_count} valid 8-day S2 composites for ROI '{roi_name}' for download.")

    if valid_composites_count == 0:
        logging.warning(f"No valid S2 composites found for ROI '{roi_name}'. Skipping download.")
        return

    # 3. Download S2 composites to local storage.
    download_s2_bands(valid_s2_composites, big_folder, roi, 'S2_8days_', roi_name, 's2_images')

    # logging.info(f"--- S2 retrieval process finished for ROI '{roi_name}'. ---")


if __name__ == '__main__':
    # =========================================================================
    # EXAMPLE USAGE
    # =========================================================================
    # This block demonstrates how to run the S2 retrieval script.
    # It will now process a grid from a GeoJSON file, download data for specific
    # dates for each grid cell, and save it to a local folder named 's2_data_output'.

    try:
        # 1. Initialize Earth Engine
        try:
            project_name = 'ee-hadat-461702-p4'
            ee.Initialize(project=project_name)
        except Exception:
            logging.info("Authenticating to Earth Engine...")
            ee.Authenticate()
            project_name = 'ee-hadat-461702-p4'
            ee.Initialize(project=project_name)

        # 2. Define script parameters
        grid_geojson_path = 'Grid_50K_MatchedDates.geojson'
        big_folder = 's2_data_output'  # Output will be in 's2_data_output/ROI_NAME/...'
        
        # Load the grid GeoJSON file
        with open(grid_geojson_path, 'r') as f:
            grid_data = json.load(f)
        
        # logging.info(f"Loaded grid GeoJSON from: {grid_geojson_path} containing {len(grid_data.get('features', []))} features.")

        # 3. Process each feature in the grid
        for i, feature in enumerate(grid_data.get('features', [])):
            # logging.info(f"Processing feature {i+1}/{len(grid_data['features'])}")
            
            # Extract ROI geometry directly
            roi_geometry = ee.Geometry.Polygon(feature['geometry']['coordinates'])
            
            # Use 'PhienHieu' as roi_name, fallback to 'id' if not present
            roi_name = feature['properties'].get('PhienHieu', feature.get('id', f'unknown_roi_{i}'))
            
            # Extract and parse m_dates
            m_dates_str = feature['properties'].get('m_dates', '')
            if not m_dates_str:
                logging.warning(f"No 'm_dates' found for ROI '{roi_name}'. Skipping.")
                continue
            
            m_dates_list_str = m_dates_str.split(';')
            # Convert date strings to ee.Date objects, taking only the first one for testing
            target_composite_dates = [ee.Date(m_dates_list_str[0].strip())] # Process only the first date

            if not target_composite_dates:
                logging.warning(f"No valid dates parsed for ROI '{roi_name}'. Skipping.")
                continue

            # 4. Run the main retrieval function for the current ROI and its dates
            main_s2_retrieval(target_composite_dates, roi_geometry, roi_name, big_folder)
            break # Process only the first feature for testing

    except Exception as e:
        logging.critical(f"An error occurred during the example run: {e}", exc_info=True)
    finally:
        # Clean up temporary ROI directory if it was created in previous runs
        # This section is largely removed as the new process reads directly from file
        pass 

    # logging.info("Example run finished.")

