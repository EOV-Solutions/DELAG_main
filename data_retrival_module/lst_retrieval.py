import ee
import requests  # used to download files
import os
import tempfile
import zipfile
import shutil
import time
import rasterio
import numpy as np
from datetime import datetime

ee.Initialize(project='ee-hadat-461702-p4')

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
from lst_module import Landsat_LST as LandsatLST


def lst_retrive(date_start, date_end, geometry, ROI, main_folder):
    satellites = ["L8", "L9"]

    for satellite in satellites:
        # Define parameters.
        # date_start = '2022-12-15'
        # date_end = '2023-01-01'
        use_ndvi = True

        # Get Landsat collection with added variables.
        LandsatColl = LandsatLST.collection(satellite, date_start, date_end, geometry, use_ndvi)
        # print('Landsat Collection:', LandsatColl.getInfo())

        # Convert the collection to a list.
        imageList = LandsatColl.toList(LandsatColl.size())
        imageCount = LandsatColl.size().getInfo()
        # print('Number of images to process:', imageCount)
        # print()

        for i in range(imageCount):
            image = ee.Image(imageList.get(i))
            # Get the image date formatted as YYYY-MM-dd.
            imageDate = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            # Get the timestamp for metadata writing
            time_start_ms = image.get('system:time_start').getInfo()

            dest_folder_path = os.path.join(main_folder, ROI, "lst")
            os.makedirs(dest_folder_path, exist_ok=True)
            
            # New filename format includes satellite name
            current_satellite_path = os.path.join(dest_folder_path, f"{satellite}_lst16days_{imageDate}.tif")

            if os.path.exists(current_satellite_path):
                print(f"Skipping LST download for {satellite} on {imageDate}: file already exists.")
                continue

            # Download to a temporary file for potential comparison
            temp_dir = tempfile.mkdtemp()
            temp_tif_path = None
            download_successful = False

            try:
                # Get a download URL for the LST band as a ZIP.
                download_params = {
                    'scale': 30, 'region': geometry, 'fileFormat': 'ZIP', 'crs': 'EPSG:4326'
                }
                download_url = image.select('LST').getDownloadURL(download_params)
                response = requests.get(download_url, timeout=120)

                if response.status_code == 200:
                    zip_filename = os.path.join(temp_dir, "download.zip")
                    with open(zip_filename, 'wb') as f: f.write(response.content)
                    
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        tif_files = [f for f in zip_ref.namelist() if f.lower().endswith('.tif')]
                        if tif_files:
                            source_tif = os.path.join(temp_dir, tif_files[0])
                            zip_ref.extract(tif_files[0], temp_dir)
                            temp_tif_path = os.path.join(temp_dir, f"temp_{satellite}_{imageDate}.tif")
                            shutil.move(source_tif, temp_tif_path)
                            download_successful = True
                        else:
                            print(f"Download failed for LST {satellite} on {imageDate}: No TIFF file found in ZIP.")
                else:
                    print(f"Download failed for LST {satellite} on {imageDate}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Download failed for LST {satellite} on {imageDate}: {e}")

            # If download was successful, proceed with comparison and saving logic
            if download_successful and temp_tif_path:
                competitor_satellite = "L9" if satellite == "L8" else "L8"
                competitor_path = os.path.join(dest_folder_path, f"{competitor_satellite}_lst16days_{imageDate}.tif")

                if os.path.exists(competitor_path):
                    # A competitor file exists, compare nodata values
                    nodata_new = count_nodata(temp_tif_path)
                    nodata_existing = count_nodata(competitor_path)

                    if nodata_new < nodata_existing:
                        # The new image is better (less nodata), so replace the old one
                        os.remove(competitor_path)
                        shutil.move(temp_tif_path, current_satellite_path)
                        # Write metadata to the new file
                        write_metadata_to_tiff(current_satellite_path, time_start_ms, f'{satellite}_LST')
                        print(f"Successfully downloaded LST for {satellite} on {imageDate} (replaced {competitor_satellite}).")
                    else:
                        # The existing image is better, discard the new one
                        print(f"Skipping LST download for {satellite} on {imageDate}: existing {competitor_satellite} image is better.")
                else:
                    # No competitor exists, just save the downloaded file
                    shutil.move(temp_tif_path, current_satellite_path)
                    # Write metadata to the new file
                    write_metadata_to_tiff(current_satellite_path, time_start_ms, f'{satellite}_LST')
                    print(f"Successfully downloaded LST for {satellite} on {imageDate}.")

            # Clean up the temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            # time.sleep(0.5)  # Pause briefly between downloads if necessary

# Run the function
# lst_retrive(date_start, date_end, geometry, "AnNinh-QuynhPhu-ThaiBinh", "/mnt/data1tb/LSTRetrieval/Code/download_data")

def cloud_mask_landsat(image):
    """
    Masks clouds and cloud shadows in Landsat images using the QA_PIXEL band.
    A more aggressive approach is used here, also masking dilated clouds to reduce
    the impact of cloud edges and haze.
    """
    qa = image.select('QA_PIXEL')
    # Bits 1 (Dilated Cloud), 3 (Cloud), and 4 (Cloud Shadow) are used for masking.
    dilated_cloud = 1 << 1
    cloud = 1 << 3
    cloud_shadow = 1 << 4
    mask = (qa.bitwiseAnd(dilated_cloud).eq(0)
              .And(qa.bitwiseAnd(cloud).eq(0))
              .And(qa.bitwiseAnd(cloud_shadow).eq(0)))
    return image.updateMask(mask)