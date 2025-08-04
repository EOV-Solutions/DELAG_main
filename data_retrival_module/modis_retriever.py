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
EXPORT_SCALE = 30  # MODIS LST native resolution is 1km

# --- Earth Engine Initialization ---
try:
    ee.Initialize(project='ee-hadat-461702-p4')
except Exception:
    ee.Authenticate()
    ee.Initialize(project='ee-hadat-461702-p4')

# --- Utility Functions (Unchanged) ---

def get_roi_coords_from_tif(tif_path):
    """Reads bounds from a TIF and converts them to the target CRS."""
    with rasterio.open(tif_path) as dataset:
        bounds = dataset.bounds
        if dataset.crs.to_string() != TARGET_CRS:
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
                return True
        print(f"Verification failed for {os.path.basename(img_path)}: Invalid raster data.")
        return False
    except (rasterio.errors.RasterioIOError, Exception) as e:
        print(f"Verification error for {img_path}: {e}")
        return False

def export_ee_image(image, bands, region, out_path, scale, crs=TARGET_CRS, timestamp_ms=None, acquisition_type=None):
    """Exports an Earth Engine image to a local path and writes metadata."""
    temp_dir = os.path.join(os.path.dirname(out_path), 'temp_dl')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        region_geometry = ee.Geometry.Polygon(region, proj=crs, evenOdd=False)
        image_clipped = image.clip(region_geometry).select(bands)

        band_info = image_clipped.bandNames().getInfo()
        if not band_info:
            print(f"Download failed for {os.path.basename(out_path)}: Image has no bands after clipping.")
            return

        url = image_clipped.getDownloadURL({
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

            first_tif_path = os.path.join(temp_dir, tif_files[0])
            zip_ref.extract(tif_files[0], temp_dir)
            
            shutil.move(first_tif_path, out_path)

            print(f"Successfully downloaded image: {os.path.basename(out_path)}")

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
    """Resamples a source GeoTIFF to match the metadata of a reference GeoTIFF."""
    try:
        with rasterio.open(reference_path) as ref:
            ref_meta = ref.meta.copy()
        with rasterio.open(source_path) as src:
            if (src.width == ref_meta['width'] and 
                src.height == ref_meta['height'] and 
                src.transform == ref_meta['transform']):
                return
            ref_meta.update({
                'count': src.count,
                'dtype': src.meta['dtype'],
                'nodata': src.nodata
            })
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
                        resampling=rasterio.warp.Resampling.bilinear
                    )
            shutil.move(temp_output_path, source_path)
    except Exception as e:
        print(f"Resampling failed for {source_path}: {e}")

# --- Updated MODIS Function ---

def get_modis_for_date(target_date, roi_geom, region, out_folder, reference_tif_path):
    """
    Fetches and exports MODIS Land Surface Temperature for a specific date.
    It downloads the LST_Day_1km band in Kelvin, writes metadata, 
    and resamples the image to match the reference TIF.
    """
    date_str = target_date.strftime('%Y-%m-%d')
    out_path = os.path.join(out_folder, f'MODIS_LST_{date_str}.tif')
    
    if SKIPPING and os.path.exists(out_path):
        print(f"Skipping MODIS download for {date_str}: file already exists.")
        resample_to_match_reference(out_path, reference_tif_path)
        return

    try:
        start_date = ee.Date(target_date)
        end_date = start_date.advance(1, 'day')
        
        modis_collection = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate(start_date, end_date) \
            .filterBounds(roi_geom)

        collection_size = modis_collection.size().getInfo()
        if collection_size == 0:
            print(f"Skipping MODIS download for {date_str}: No images found.")
            return
            
        print(f"Found {collection_size} MODIS images for {date_str}")
        # Use toList(1).get(0) instead of first() for more reliable image extraction
        image = ee.Image(modis_collection.toList(1).get(0))
        
        # Copy properties to ensure metadata is preserved through processing.
        # copyProperties returns an ee.Element, so we must cast it back to an ee.Image.
        image_with_props = ee.Image(image.copyProperties(image, ['system:time_start', 'system:time_end']))
        
        # Get the timestamp for metadata writing
        time_start_ms = image_with_props.get('system:time_start').getInfo()
        
        # Apply scale factor to get LST in Kelvin
        lst_kelvin = image_with_props.select('LST_Day_1km').multiply(0.02).rename('LST_Kelvin')
        
        # Check for valid data in the ROI before exporting
        stats = lst_kelvin.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=roi_geom,
            scale=1000,  # Native resolution of MODIS LST
            maxPixels=1e9
        ).getInfo()

        # If the dictionary is empty or the keys have null values, there's no valid data
        if not stats or stats.get('LST_Kelvin_min') is None:
            print(f"Skipping MODIS download for {date_str}: No valid data found in the ROI (likely all cloudy).")
            return
            
        print(f"LST Kelvin range for {date_str}: {stats}")
 
        print(f"Exporting MODIS LST data for {date_str}...")
        export_ee_image(
            image=lst_kelvin,
            bands=['LST_Kelvin'],
            region=region,
            out_path=out_path,
            scale=EXPORT_SCALE,
            crs=TARGET_CRS,
            timestamp_ms=time_start_ms,
            acquisition_type='Day'
        )
 
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            resample_to_match_reference(out_path, reference_tif_path)
            if verify_image(out_path):
                print(f"Verified: {os.path.basename(out_path)}")
            else:
                print(f"Deleting invalid file: {os.path.basename(out_path)}")
                os.remove(out_path)
        elif os.path.exists(out_path):
             print(f"Downloaded file {os.path.basename(out_path)} is empty. Deleting.")
             os.remove(out_path)

    except ee.EEException as e:
        print(f"Download failed for MODIS {date_str}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred for {date_str}: {e}")

# --- Main Execution Logic (Unchanged) ---

def main(input_folder, output_folder, specific_dates=None):
    overall_start_time = time.time()
    os.makedirs(output_folder, exist_ok=True)
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
    if specific_dates:
        dates_to_process = [datetime.strptime(d, '%Y-%m-%d') for d in specific_dates]
        print(f"Processing a specific list of {len(dates_to_process)} provided dates.")
    else:
        print("No specific dates provided. Deriving dates from reference filenames...")
        dates_to_process = get_dates_from_filenames(input_folder)
    if not dates_to_process:
        print("No dates to process. Exiting.")
        return
    print(f"Found {len(dates_to_process)} total dates to process for MODIS retrieval.")
    for i, target_date in enumerate(dates_to_process):
        date_str = target_date.strftime('%Y-%m-%d')
        print(f"--- Processing date {i+1}/{len(dates_to_process)}: {date_str} ---")
        get_modis_for_date(target_date, roi_geometry, roi_coords, output_folder, reference_tif)
    total_time = time.time() - overall_start_time
    print(f"\nMODIS retrieval complete. Total time: {total_time:.2f} seconds.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and align MODIS LST data to match a set of reference GeoTIFFs.")
    parser.add_argument("--input_folder", required=True, help="Folder containing the reference .tif files.")
    parser.add_argument("--output_folder", required=True, help="Folder where the downloaded MODIS data will be saved.")
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)