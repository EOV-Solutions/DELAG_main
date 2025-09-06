"""
Comprehensive Google Earth Engine Data Downloader

This script downloads satellite data from multiple sources (ERA5, ASTER, Landsat8/9)
for a specified date range and organizes them into individual zip files with 
task ID naming and JSON mapping for tracking.

Author: AI Assistant
Date: 2025
"""

import os
import time
import json
import uuid
import zipfile
import shutil
import requests
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

import ee
import rasterio
import numpy as np

# ========================================
# CONFIGURATION AND INITIALIZATION
# ========================================

# Global Configuration
TARGET_CRS = 'EPSG:4326'
DOWNLOAD_TIMEOUT = 600  # seconds
MAX_RETRIES = 3

# Dataset-specific native scales (meters)
# Using native resolutions to preserve original data quality
DATASET_SCALES = {
    'era5': 11000,      # ERA5 Land: ~11km native resolution (0.1° grid)
    'aster': 100,       # ASTER GED: 100m native resolution
    'landsat_l1': 100,  # Landsat L1 thermal bands: 100m native resolution (B10, B11)
    'landsat_l2': 30,   # Landsat L2 surface reflectance: 30m native resolution (SR_B*)
    'sentinel2': 10     # Sentinel-2: 10m native resolution (B2, B3, B4, B8)
}

# Initialize Google Earth Engine
try:
    ee.Initialize(project='ee-hadat-461702-p4')
    print("Google Earth Engine initialized successfully")
except Exception as e:
    print("Authenticating to Earth Engine...")
    ee.Authenticate()
    ee.Initialize(project='ee-hadat-461702-p4')
    print("Google Earth Engine authenticated and initialized")

# ========================================
# UTILITY FUNCTIONS
# ========================================

def generate_task_id() -> str:
    """Generate a unique task ID"""
    return str(uuid.uuid4())

def find_grid_feature(phien_hieu: str, grid_file_path: str) -> dict | None:
    """
    Finds a specific feature in a GeoJSON file based on its 'PhienHieu' property.

    Args:
        phien_hieu: The 'PhienHieu' identifier to search for.
        grid_file_path: Path to the GeoJSON grid file.

    Returns:
        The matching GeoJSON feature dictionary, or None if not found.
    """
    try:
        with open(grid_file_path, 'r') as f:
            grid_data = json.load(f)
        
        for feature in grid_data.get('features', []):
            if feature.get('properties', {}).get('PhienHieu') == phien_hieu:
                return feature
        
        print(f"Error: Failed to find feature with PhienHieu '{phien_hieu}'.")
        return None
    except FileNotFoundError:
        print(f"Error: Grid file not found at: {grid_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to read grid file: {grid_file_path}")
        return None

def create_output_directories(base_path: str) -> Dict[str, str]:
    """Create output directories for different dataset types"""
    dirs = {
        'zips': os.path.join(base_path, 'downloaded_zips'),
        'temp': os.path.join(base_path, 'temp_downloads'),
        'era5': os.path.join(base_path, 'temp_downloads', 'era5'),
        'aster': os.path.join(base_path, 'temp_downloads', 'aster'),
        'sentinel2': os.path.join(base_path, 'temp_downloads', 'sentinel2'),
        'landsat8_l1': os.path.join(base_path, 'temp_downloads', 'landsat8_l1'),
        'landsat8_l2': os.path.join(base_path, 'temp_downloads', 'landsat8_l2'),
        'landsat9_l1': os.path.join(base_path, 'temp_downloads', 'landsat9_l1'),
        'landsat9_l2': os.path.join(base_path, 'temp_downloads', 'landsat9_l2')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_roi_geometry(region_coords: List[List[float]]) -> ee.Geometry:
    """Convert coordinate list to Earth Engine Geometry"""
    return ee.Geometry.Polygon(region_coords, proj=TARGET_CRS, evenOdd=False)

def get_roi_geometry_from_geojson(geometry_data: dict) -> ee.Geometry:
    """Convert GeoJSON geometry to Earth Engine Geometry"""
    return ee.Geometry.Polygon(geometry_data['coordinates'], proj=TARGET_CRS, evenOdd=False)

def merge_tifs(tif_files: List[str], output_path: str, bands_order: List[str]) -> bool:
    """
    Merges a list of single-band GeoTIFFs into a single multi-band GeoTIFF.
    
    Args:
        tif_files: List of paths to single-band GeoTIFF files.
        output_path: Path to save the merged multi-band GeoTIFF.
        bands_order: A list of band names specifying the desired order in the output file.
    
    Returns:
        True if merging was successful, False otherwise.
    """
    try:
        # Create a mapping from band name to file path to ensure correct order
        band_file_map = {}
        for f_path in tif_files:
            basename = os.path.basename(f_path)
            for band_name in bands_order:
                if band_name in basename:
                    band_file_map[band_name] = f_path
                    break
        
        # Sort the file paths according to the desired band order
        sorted_files = [band_file_map[b] for b in bands_order if b in band_file_map]
        
        if len(sorted_files) < len(bands_order):
            print(f"  > Warning: Not all requested bands were found for merging. Proceeding with {len(sorted_files)} bands.")

        if not sorted_files:
            print("  ✗ Error: No valid band files found to merge.")
            return False

        # Read metadata from the first file to use as a template
        with rasterio.open(sorted_files[0]) as src0:
            profile = src0.profile.copy()

        # Update metadata for the multi-band output
        profile.update(
            count=len(sorted_files),
            driver='GTiff',
            compress='lzw'
        )

        # Write the multi-band raster
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, file_path in enumerate(sorted_files):
                with rasterio.open(file_path) as src:
                    dst.write(src.read(1), i + 1)
        
        print(f"  > Successfully merged {len(sorted_files)} bands into {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"  ✗ Error merging TIF files: {e}")
        return False

def download_ee_image(image: ee.Image, bands: List[str], region,
                     scale: int, output_dir: str, filename_prefix: str) -> List[Tuple[str, str]]:
    """
    Download an Earth Engine image. Tries multi-band first, falls back to single-band downloads.
    Returns a list of (source_file_path, generic_band_filename) tuples.
    """
    files_downloaded = []
    temp_extract_base_dir = os.path.join(output_dir, filename_prefix)
    os.makedirs(temp_extract_base_dir, exist_ok=True)

    try:
        if isinstance(region, ee.Geometry):
            region_geometry = region
            region_coords = region.coordinates().getInfo()[0]
        else:
            region_geometry = get_roi_geometry(region)
            region_coords = region
        
        clipped_image = image.clip(region_geometry).select(bands)
        
        band_info = clipped_image.bandNames().getInfo()
        if not band_info:
            print(f"Warning: No bands available for {filename_prefix} after clipping.")
            return []

        # --- Attempt multi-band download first ---
        try:
            print(f"  > Attempting multi-band download for {filename_prefix}...")
            url = clipped_image.getDownloadURL({
                'scale': scale, 'region': region_coords, 'fileFormat': 'GeoTIFF', 'crs': TARGET_CRS
            })
            
            temp_zip_path = os.path.join(output_dir, f"{filename_prefix}_multi.zip")
            response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()

            with open(temp_zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_extract_base_dir)
            os.remove(temp_zip_path)

            for file in os.listdir(temp_extract_base_dir):
                if file.endswith('.tif'):
                    source_path = os.path.join(temp_extract_base_dir, file)
                    files_downloaded.append((source_path, file))
            
            if files_downloaded:
                print(f"  > Successfully downloaded multi-band image for {filename_prefix}.")
                return files_downloaded
            else:
                raise Exception("Multi-band download successful but no TIFF files were extracted.")

        except Exception as e:
            if "Total request size" in str(e):
                print(f"  > Multi-band download failed for {filename_prefix} due to size limit. Falling back to single-band downloads.")
            else:
                print(f"  ✗ Critical error during multi-band download for {filename_prefix}: {e}")
                print("  > Attempting single-band download fallback...")

            # --- Fallback to single-band downloads ---
            shutil.rmtree(temp_extract_base_dir)
            os.makedirs(temp_extract_base_dir, exist_ok=True)
            files_downloaded.clear()
            
            for band_name in bands:
                print(f"    > Downloading band '{band_name}'...")
                try:
                    single_band_image = clipped_image.select([band_name])
                    url = single_band_image.getDownloadURL({
                        'scale': scale, 'region': region_coords, 'fileFormat': 'GeoTIFF', 'crs': TARGET_CRS
                    })
                    
                    temp_zip_path_band = os.path.join(output_dir, f"{filename_prefix}_{band_name}.zip")
                    response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
                    response.raise_for_status()

                    with open(temp_zip_path_band, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            f.write(chunk)
                    
                    with zipfile.ZipFile(temp_zip_path_band, 'r') as zip_ref:
                        extracted_files = [f for f in zip_ref.namelist() if f.endswith('.tif')]
                        if extracted_files:
                            zip_ref.extract(extracted_files[0], temp_extract_base_dir)
                            source_path = os.path.join(temp_extract_base_dir, extracted_files[0])
                            files_downloaded.append((source_path, extracted_files[0]))
                            print(f"    ✓ Successfully downloaded band '{band_name}'.")
                    os.remove(temp_zip_path_band)

                except Exception as band_e:
                    print(f"    ✗ Failed to download band '{band_name}' for {filename_prefix}: {band_e}")
    except Exception as e:
        print(f"  ✗ Critical error during download setup for {filename_prefix}: {e}")

    return files_downloaded

# ========================================
# DATASET-SPECIFIC DOWNLOAD FUNCTIONS
# ========================================

class DatasetDownloader:
    """Base class for dataset downloaders"""
    
    def __init__(self, output_dirs: Dict[str, str], region_geometry: ee.Geometry):
        self.output_dirs = output_dirs
        self.region_geometry = region_geometry

class ERA5Downloader(DatasetDownloader):
    """Download ERA5 reanalysis data"""
    
    def download_era5_for_date(self, target_date: datetime) -> List[Tuple[str, str]]:
        """Download all hourly ERA5 images for a specific date and return their file paths."""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            print(f"  - Processing ERA5 for {date_str}...")
            
            start_date = ee.Date(target_date.strftime('%Y-%m-%d'))
            end_date = start_date.advance(1, 'day')
            
            era5_hourly_collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region_geometry)
            
            image_list = era5_hourly_collection.toList(era5_hourly_collection.size())
            collection_size = image_list.size().getInfo()

            if collection_size == 0:
                print(f"    ✗ No ERA5 hourly data found for {date_str}")
                return []
            
            print(f"    - Found {collection_size} hourly images for {date_str}")
            
            files_to_return = []
            for i in range(collection_size):
                hourly_image = ee.Image(image_list.get(i))
                timestamp = hourly_image.get('system:time_start').getInfo()
                dt_object = datetime.fromtimestamp(timestamp / 1000)
                datetime_str = dt_object.strftime('%Y%m%d_%H%M%S')
                
                bands = ['skin_temperature', 'temperature_2m']
                
                downloaded_files = download_ee_image(
                    hourly_image,
                    bands,
                    self.region_geometry,
                    DATASET_SCALES['era5'],
                    self.output_dirs['era5'],
                    f"era5_{datetime_str}"
                )
                
                if downloaded_files:
                    for file_path, generic_filename in downloaded_files:
                        new_name = ""
                        if 'skin_temperature' in generic_filename:
                            new_name = f"skin_temperature_era5_{datetime_str}Z.tif"
                        elif 'temperature_2m' in generic_filename:
                            new_name = f"temperature_2m_era5_{datetime_str}Z.tif"
                        
                        if new_name:
                            files_to_return.append((file_path, new_name))
            
            return files_to_return
            
        except Exception as e:
            print(f"    ✗ ERA5 hourly download failed for {target_date.strftime('%Y-%m-%d')}: {e}")
            return []

class ASTERDownloader(DatasetDownloader):
    """Download ASTER Global Emissivity Dataset"""
    
    def download_aster_data(self) -> List[Tuple[str, str]]:
        """Download ASTER emissivity data (static dataset) and return file paths."""
        try:
            print(f"Processing ASTER GED (static dataset)...")
            
            # ASTER Global Emissivity Dataset (static)
            aster_ged = ee.Image('NASA/ASTER_GED/AG100_003')
            
            # Define bands to download, including band14
            bands = [
                'emissivity_band10', 'emissivity_band11', 
                'emissivity_band12', 'emissivity_band13', 
                'emissivity_band14', 'ndvi'
            ]
            
            # Download the image
            downloaded_files = download_ee_image(
                aster_ged,
                bands,
                self.region_geometry,
                DATASET_SCALES['aster'],
                self.output_dirs['aster'],
                f"aster_static_data"
            )
            
            if downloaded_files:
                files_to_return = []
                for file_path, generic_filename in downloaded_files:
                    # Rename files according to specification
                    new_name = ""
                    if 'emissivity_band10' in generic_filename: new_name = "ASTER_emissivity_band10.tif"
                    elif 'emissivity_band11' in generic_filename: new_name = "ASTER_emissivity_band11.tif"
                    elif 'emissivity_band12' in generic_filename: new_name = "ASTER_emissivity_band12.tif"
                    elif 'emissivity_band13' in generic_filename: new_name = "ASTER_emissivity_band13.tif"
                    elif 'emissivity_band14' in generic_filename: new_name = "ASTER_emissivity_band14.tif"
                    elif 'ndvi' in generic_filename: new_name = "ASTER_ndvi.tif"
                    else: new_name = generic_filename
                    
                    if new_name:
                        files_to_return.append((file_path, new_name))
                return files_to_return
            return []
            
        except Exception as e:
            print(f"ASTER download failed: {e}")
            return []

class LandsatDownloader(DatasetDownloader):
    """Download Landsat 8/9 data (both L1 and L2)"""
    
    def download_landsat_for_interval(self, start_date: datetime, end_date: datetime, satellite: str, 
                                     processing_level: str) -> List[Tuple[str, str]]:
        """Download all available Landsat images for a specific date range."""
        try:
            collection_name = self._get_collection_name(satellite, processing_level)
            if not collection_name: return []

            landsat_collection = ee.ImageCollection(collection_name) \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region_geometry) \
                .filter(ee.Filter.lt('CLOUD_COVER', 80))
            
            collection_size = landsat_collection.size().getInfo()
            if collection_size == 0:
                print(f"  - No Landsat {satellite} L{processing_level} images found in date range.")
                return []

            print(f"  - Found {collection_size} Landsat {satellite} L{processing_level} images. Downloading...")
            image_list = landsat_collection.toList(collection_size)
            
            files_to_return = []
            for i in range(collection_size):
                landsat_image = ee.Image(image_list.get(i))
                image_date = ee.Date(landsat_image.get('system:time_start'))
                date_str = image_date.format('YYYY-MM-dd').getInfo()

                bands = self._get_bands(processing_level)
                scale_key = f"landsat_{processing_level.lower()}"
                scale = DATASET_SCALES[scale_key]
                
                dataset_key = f"landsat{satellite.lower()}_{processing_level.lower()}"
                downloaded_files = download_ee_image(
                    landsat_image,
                    bands,
                    self.region_geometry,
                    scale,
                    self.output_dirs[dataset_key],
                    f"landsat{satellite}_{processing_level}_{date_str}"
                )
                
                if downloaded_files:
                    image_date_str_fmt = image_date.format('YYYYMMdd').getInfo()
                    scene_id = landsat_image.get('LANDSAT_PRODUCT_ID').getInfo() or f"scene_{i}"
                    for file_path, generic_filename in downloaded_files:
                        band_name = self._extract_band_name(generic_filename)
                        if band_name:
                            new_name = f"L{satellite}_{processing_level}_{band_name}_{image_date_str_fmt}_{scene_id}.tif"
                            files_to_return.append((file_path, new_name))
            return files_to_return
            
        except Exception as e:
            print(f"  ✗ Landsat {satellite} {processing_level} download failed: {e}")
            return []
    
    def _get_collection_name(self, satellite: str, processing_level: str) -> Optional[str]:
        """Get GEE collection name for Landsat satellite and processing level"""
        collections = {
            ('8', 'L1'): 'LANDSAT/LC08/C02/T1_TOA',
            ('8', 'L2'): 'LANDSAT/LC08/C02/T1_L2',
            ('9', 'L1'): 'LANDSAT/LC09/C02/T1_TOA',
            ('9', 'L2'): 'LANDSAT/LC09/C02/T1_L2'
        }
        return collections.get((satellite, processing_level))
    
    def _get_bands(self, processing_level: str) -> List[str]:
        """Get band list based on processing level"""
        if processing_level == 'L1':
            return ['B10', 'B11']
        elif processing_level == 'L2':
            return ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
        else:
            return []
    
    def _extract_band_name(self, filename: str) -> Optional[str]:
        """Extract band name from GEE downloaded filename"""
        for band in ['B10', 'B11', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']:
            if band in filename:
                return band
        return None

class Sentinel2Downloader(DatasetDownloader):
    """Download Sentinel-2 Surface Reflectance data"""
    
    def download_sentinel2_for_interval(self, start_date: datetime, end_date: datetime) -> List[Tuple[str, str]]:
        """Download all available Sentinel-2 images for a specific date range."""
        try:
            ee_start_date = ee.Date(start_date.strftime('%Y-%m-%d'))
            ee_end_date = ee.Date(end_date.strftime('%Y-%m-%d'))
            
            cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
            qa_band = 'cs'
            clear_threshold = 0.5
            
            sentinel2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                .filterDate(ee_start_date, ee_end_date) \
                .filterBounds(self.region_geometry) \
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 85)) \
                .linkCollection(cs_plus, [qa_band])
            
            collection_size = sentinel2_collection.size().getInfo()
            if collection_size == 0:
                print(f"  - No Sentinel-2 images found in date range.")
                return []

            print(f"  - Found {collection_size} Sentinel-2 images. Applying cloud mask and downloading...")
            
            sentinel_masked = sentinel2_collection.map(
                lambda img: img.updateMask(img.select(qa_band).gte(clear_threshold)).clip(self.region_geometry)
            )
            
            image_list = sentinel_masked.toList(collection_size)
            files_to_return = []

            for i in range(collection_size):
                sentinel2_image = ee.Image(image_list.get(i))
                image_date = ee.Date(sentinel2_image.get('system:time_start'))
                date_str = image_date.format('YYYY-MM-dd').getInfo()

                bands = ['B4', 'B3', 'B2', 'B8']
                
                downloaded_files = download_ee_image(
                    sentinel2_image,
                    bands,
                    self.region_geometry,
                    DATASET_SCALES['sentinel2'],
                    self.output_dirs['sentinel2'],
                    f"sentinel2_{date_str}"
                )
                
                if downloaded_files:
                    source_tif_paths = [fp for fp, fn in downloaded_files]

                    # Define paths and names for merging
                    image_date_str_fmt = image_date.format('YYYYMMdd').getInfo()
                    tile_id = sentinel2_image.get('MGRS_TILE').getInfo() or f"tile_{i}"
                    
                    merged_filename = f"S2_SR_merged_{image_date_str_fmt}_{tile_id}.tif"
                    merged_output_path = os.path.join(self.output_dirs['sentinel2'], merged_filename)

                    # Merge the downloaded single-band files into one multi-band file
                    if merge_tifs(source_tif_paths, merged_output_path, bands_order=bands):
                        # If merge successful, this is the file we add to the zip list
                        final_archive_name = f"S2_SR_{image_date_str_fmt}_{tile_id}.tif"
                        files_to_return.append((merged_output_path, final_archive_name))
                    else:
                        print(f"  ✗ Failed to merge bands for {date_str}, scene {tile_id}. Skipping this scene.")

            return files_to_return
            
        except Exception as e:
            print(f"  ✗ Sentinel-2 download failed: {e}")
            return []
    
    def _extract_band_name(self, filename: str) -> Optional[str]:
        """Extract band name from GEE downloaded filename"""
        for band in ['B4', 'B3', 'B2', 'B8']:
            if band in filename:
                return band
        return None

# ========================================
# MAIN DOWNLOAD ORCHESTRATOR
# ========================================

class GEEDataDownloader:
    """Main class to orchestrate all dataset downloads"""
    
    def __init__(self, output_base_path: str, region_geometry: ee.Geometry):
        self.output_base_path = output_base_path
        self.region_geometry = region_geometry
        self.output_dirs = create_output_directories(output_base_path)
        
        # Initialize downloaders
        self.era5_downloader = ERA5Downloader(self.output_dirs, region_geometry)
        self.aster_downloader = ASTERDownloader(self.output_dirs, region_geometry)
        self.sentinel2_downloader = Sentinel2Downloader(self.output_dirs, region_geometry)
        self.landsat_downloader = LandsatDownloader(self.output_dirs, region_geometry)

    def _create_dataset_zip(self, files_to_zip: List[Tuple[str, str]], task_id: str) -> Optional[str]:
        """Create a single zip file for a dataset from a list of files."""
        if not files_to_zip:
            return None
            
        zip_filename = f"{task_id}.zip"
        zip_path = os.path.join(self.output_dirs['zips'], zip_filename)
        
        print(f"  > Creating zip file: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for source_path, archive_name in files_to_zip:
                    if os.path.exists(source_path):
                        zipf.write(source_path, archive_name)
                    else:
                        print(f"  > Warning: Source file not found, skipping: {source_path}")
            return zip_path
        except Exception as e:
            print(f"  > Error creating zip file {zip_path}: {e}")
            return None
    
    def download_all_for_date_range(self, start_date: datetime, end_date: datetime,
                                   datasets: List[str] = None) -> Dict:
        """Download all datasets for a date range, creating one zip per dataset."""
        if datasets is None:
            datasets = ['era5', 'aster', 'sentinel2', 'landsat8_l1', 'landsat8_l2', 'landsat9_l1', 'landsat9_l2']
        
        print(f"Starting downloads for date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        dates_to_process = []
        current_date = start_date
        while current_date <= end_date:
            dates_to_process.append(current_date)
            current_date += timedelta(days=1)
        
        print(f"Processing {len(dates_to_process)} dates for {len(datasets)} datasets")

        self.global_task_mapping = {}

        for dataset in datasets:
            task_id = generate_task_id()
            print(f"\n=== Processing dataset: {dataset} | Task ID: {task_id} ===")
            
            files_to_zip = []
            archive_names = set() # To track for duplicates
            
            if dataset == 'aster':
                aster_files = self.aster_downloader.download_aster_data()
                if aster_files:
                    files_to_zip.extend(aster_files)
            elif dataset == 'sentinel2':
                s2_files = self.sentinel2_downloader.download_sentinel2_for_interval(start_date, end_date)
                if s2_files:
                    files_to_zip.extend(s2_files)
            elif dataset.startswith('landsat'):
                parts = dataset.split('_')
                satellite = parts[0][-1]
                level = parts[1].upper()
                landsat_files = self.landsat_downloader.download_landsat_for_interval(start_date, end_date, satellite, level)
                if landsat_files:
                    files_to_zip.extend(landsat_files)
            elif dataset == 'era5':
                for date in dates_to_process:
                    daily_files = self.era5_downloader.download_era5_for_date(date)
                    if daily_files:
                        files_to_zip.extend(daily_files)
            
            # Filter for unique archive names to prevent duplicates in zip
            unique_files_to_zip = []
            for file_path, archive_name in files_to_zip:
                if archive_name not in archive_names:
                    unique_files_to_zip.append((file_path, archive_name))
                    archive_names.add(archive_name)
                else:
                    print(f"  > Skipping duplicate file for archive: {archive_name}")

            if unique_files_to_zip:
                zip_path = self._create_dataset_zip(unique_files_to_zip, task_id)
                if zip_path:
                    self.global_task_mapping[task_id] = {
                        'dataset_type': dataset,
                        'date_range': {
                            'start': start_date.strftime('%Y-%m-%d'),
                            'end': end_date.strftime('%Y-%m-%d')
                        },
                        'zip_path': os.path.basename(zip_path),
                        'file_count': len(unique_files_to_zip)
                    }
                    print(f"  ✓ Dataset {dataset} completed.")
            else:
                print(f"  ✗ No data found for dataset {dataset} in the entire date range.")
                
        self._save_task_mapping(start_date, end_date)
        self._cleanup_temp_dirs()
        return self.global_task_mapping
    
    def _save_task_mapping(self, start_date: datetime, end_date: datetime):
        """Save task mapping to JSON file"""
        mapping_file = os.path.join(self.output_base_path, 'task_mapping.json')
        
        summary = {
            'total_tasks': len(self.global_task_mapping),
            'datasets': {},
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            },
            'generated_at': datetime.now().isoformat()
        }
        
        for task_id, info in self.global_task_mapping.items():
            dataset_type = info['dataset_type']
            if dataset_type not in summary['datasets']:
                summary['datasets'][dataset_type] = 0
            summary['datasets'][dataset_type] += 1
        
        full_mapping = {
            'summary': summary,
            'task_mapping': self.global_task_mapping
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(full_mapping, f, indent=2)
        
        print(f"\nTask mapping saved to: {mapping_file}")
        print(f"Total tasks completed: {summary['total_tasks']}")
        print("Dataset breakdown:")
        for dataset, count in summary['datasets'].items():
            print(f"  {dataset}: {count} tasks")
    
    def _cleanup_temp_dirs(self):
        """Remove temporary download directories"""
        temp_dir = self.output_dirs['temp']
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")

# ========================================
# COMMAND LINE INTERFACE
# ========================================

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object"""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")

def parse_region(region_str: str) -> List[List[float]]:
    """Parse region string to coordinate list (backward compatibility)"""
    try:
        # Expected format: "minLon,minLat,maxLon,maxLat"
        coords = [float(x) for x in region_str.split(',')]
        if len(coords) != 4:
            raise ValueError("Need exactly 4 coordinates")
        
        min_lon, min_lat, max_lon, max_lat = coords
        
        # Create polygon coordinates
        return [
            [min_lon, min_lat],
            [max_lon, min_lat], 
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat]
        ]
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(
            f"Invalid region format: {region_str}. Use 'minLon,minLat,maxLon,maxLat'"
        )

def validate_grid_file(grid_file_path: str) -> str:
    """Validate that grid file exists and is readable"""
    if not os.path.exists(grid_file_path):
        raise argparse.ArgumentTypeError(f"Grid file not found: {grid_file_path}")
    try:
        with open(grid_file_path, 'r') as f:
            json.load(f)
        return grid_file_path
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Invalid JSON file: {grid_file_path}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Download satellite data from Google Earth Engine for multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grid-based approach (recommended)
  python comprehensive_gee_downloader.py \\
    --start_date 2023-01-01 \\
    --end_date 2023-01-07 \\
    --grid_file "data/Grid_50K_MatchedDates.geojson" \\
    --phien_hieu "D-49-49-A" \\
    --output_dir "./downloads"
    
  # Region-based approach (legacy)
  python comprehensive_gee_downloader.py \\
    --start_date 2023-06-15 \\
    --end_date 2023-06-20 \\
    --region "105.0,10.0,106.0,11.0" \\
    --output_dir "./vietnam_data" \\
    --datasets era5 aster landsat8_l2
        """
    )
    
    parser.add_argument(
        '--start_date', 
        type=parse_date,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    
    parser.add_argument(
        '--end_date',
        type=parse_date, 
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    
    # Region specification (mutually exclusive)
    region_group = parser.add_mutually_exclusive_group(required=True)
    region_group.add_argument(
        '--region',
        type=parse_region,
        help='Region of interest as "minLon,minLat,maxLon,maxLat" (legacy approach)'
    )
    region_group.add_argument(
        '--phien_hieu',
        type=str,
        help='Grid identifier (PhienHieu) to download data for (requires --grid_file)'
    )
    
    parser.add_argument(
        '--grid_file',
        type=validate_grid_file,
        help='Path to GeoJSON grid file containing grid features with PhienHieu properties'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./gee_downloads',
        help='Output directory for downloaded data (default: ./gee_downloads)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['era5', 'aster', 'sentinel2', 'landsat8_l1', 'landsat8_l2', 'landsat9_l1', 'landsat9_l2'],
        default=['era5', 'aster', 'sentinel2', 'landsat8_l1', 'landsat8_l2', 'landsat9_l1', 'landsat9_l2'],
        help='Datasets to download (default: all)'
    )
    
    args = parser.parse_args()
    
    # Validate date range
    if args.start_date > args.end_date:
        print("Error: start_date must be before or equal to end_date")
        return 1
    
    # Validate grid file requirement for phien_hieu
    if args.phien_hieu and not args.grid_file:
        print("Error: --grid_file is required when using --phien_hieu")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get region geometry
    if args.phien_hieu:
        # Grid-based approach
        feature = find_grid_feature(args.phien_hieu, args.grid_file)
        if not feature:
            return 1
        
        roi_geometry = get_roi_geometry_from_geojson(feature['geometry'])
        region_description = f"Grid {args.phien_hieu} from {args.grid_file}"
        
        # Update output directory to include phien_hieu
        args.output_dir = os.path.join(args.output_dir, args.phien_hieu)
        os.makedirs(args.output_dir, exist_ok=True)
        
    else:
        # Legacy region-based approach
        roi_geometry = get_roi_geometry(args.region)
        region_description = f"Bounding box {args.region}"
    
    print("=== Google Earth Engine Comprehensive Data Downloader ===")
    print(f"Date range: {args.start_date.strftime('%Y-%m-%d')} to {args.end_date.strftime('%Y-%m-%d')}")
    print(f"Region: {region_description}")
    print(f"Output directory: {args.output_dir}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print()
    
    # Initialize downloader
    downloader = GEEDataDownloader(args.output_dir, roi_geometry)
    
    # Start downloads
    start_time = time.time()
    try:
        task_mapping = downloader.download_all_for_date_range(
            args.start_date, 
            args.end_date,
            args.datasets
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n=== Download Complete ===")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Tasks completed: {len(task_mapping)}")
        print(f"ZIP files saved to: {downloader.output_dirs['zips']}")
        print(f"Task mapping saved to: {os.path.join(args.output_dir, 'task_mapping.json')}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        return 1
    except Exception as e:
        print(f"Download failed with error: {e}")
        return 1

if __name__ == '__main__':
    exit(main())
