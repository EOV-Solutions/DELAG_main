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
    'landsat_l2': 30    # Landsat L2 surface reflectance: 30m native resolution (SR_B*)
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

def download_ee_image(image: ee.Image, bands: List[str], region, 
                     scale: int, output_dir: str, filename_prefix: str) -> Optional[str]:
    """
    Download an Earth Engine image and save individual bands as GeoTIFFs
    Returns the path to the output directory containing the band files
    
    Args:
        region: Can be either List[List[float]] for coordinates or ee.Geometry object
    """
    try:
        # Handle different region types
        if isinstance(region, ee.Geometry):
            region_geometry = region
            region_coords = region.coordinates().getInfo()[0]  # For download URL
        else:
            region_geometry = get_roi_geometry(region)
            region_coords = region
        
        # Clip image to region and select bands
        clipped_image = image.clip(region_geometry).select(bands)
        
        # Check if image has data
        band_info = clipped_image.bandNames().getInfo()
        if not band_info:
            print(f"Warning: No bands available for {filename_prefix}")
            return None
        
        # Get download URL
        url = clipped_image.getDownloadURL({
            'scale': scale,
            'region': region_coords,
            'fileFormat': 'GeoTIFF',
            'crs': TARGET_CRS
        })
        
        # Download the zip file
        print(f"Downloading {filename_prefix}...")
        response = requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        
        # Save to temporary zip file
        temp_zip_path = os.path.join(output_dir, f"{filename_prefix}.zip")
        with open(temp_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        
        # Extract the zip file
        extract_dir = os.path.join(output_dir, filename_prefix)
        os.makedirs(extract_dir, exist_ok=True)
        
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Clean up temporary zip
        os.remove(temp_zip_path)
        
        return extract_dir
        
    except Exception as e:
        print(f"Failed to download {filename_prefix}: {e}")
        return None

# ========================================
# DATASET-SPECIFIC DOWNLOAD FUNCTIONS
# ========================================

class DatasetDownloader:
    """Base class for dataset downloaders"""
    
    def __init__(self, output_dirs: Dict[str, str], region_geometry: ee.Geometry):
        self.output_dirs = output_dirs
        self.region_geometry = region_geometry
        self.task_mapping = {}
    
    def add_to_mapping(self, task_id: str, dataset_type: str, date: str, 
                      bands: List[str], metadata: Dict = None):
        """Add entry to task mapping"""
        self.task_mapping[task_id] = {
            'dataset_type': dataset_type,
            'date': date,
            'bands': bands,
            'metadata': metadata or {}
        }

class ERA5Downloader(DatasetDownloader):
    """Download ERA5 reanalysis data"""
    
    def download_era5_for_date(self, target_date: datetime, task_id: str) -> bool:
        """Download ERA5 data for a specific date"""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            print(f"Processing ERA5 for {date_str}...")
            
            # Define date range (ERA5 daily data)
            start_date = ee.Date(target_date.strftime('%Y-%m-%d'))
            end_date = start_date.advance(1, 'day')
            
            # Get ERA5 Land daily collection
            era5_collection = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY') \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region_geometry)
            
            if era5_collection.size().getInfo() == 0:
                print(f"No ERA5 data found for {date_str}")
                return False
            
            # Get the first (and likely only) image
            era5_image = ee.Image(era5_collection.first())
            
            # Define bands to download
            bands = ['skin_temperature', 'temperature_2m']
            
            # Download the image
            extract_dir = download_ee_image(
                era5_image, 
                bands, 
                self.region_geometry, 
                DATASET_SCALES['era5'],
                self.output_dirs['era5'],
                f"era5_{date_str}"
            )
            
            if extract_dir:
                # Create organized zip file
                zip_path = self._create_era5_zip(extract_dir, task_id, date_str)
                
                # Add to mapping
                self.add_to_mapping(
                    task_id, 'ERA5', date_str, bands,
                    {'source': 'ECMWF/ERA5_LAND/DAILY', 'zip_path': zip_path}
                )
                
                return True
            
            return False
            
        except Exception as e:
            print(f"ERA5 download failed for {target_date}: {e}")
            return False
    
    def _create_era5_zip(self, extract_dir: str, task_id: str, date_str: str) -> str:
        """Create organized ZIP file for ERA5 data"""
        zip_filename = f"downloaded_era5_{task_id}.zip"
        zip_path = os.path.join(self.output_dirs['zips'], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(extract_dir):
                if file.endswith('.tif'):
                    file_path = os.path.join(extract_dir, file)
                    
                    # Rename files according to specification
                    if 'skin_temperature' in file:
                        new_name = f"skin_temperature_era5_{date_str.replace('-', '')}_100000Z.tif"
                    elif 'temperature_2m' in file:
                        new_name = f"temperature_2m_era5_{date_str.replace('-', '')}_100000Z.tif"
                    else:
                        new_name = file
                    
                    zipf.write(file_path, new_name)
        
        return zip_path

class ASTERDownloader(DatasetDownloader):
    """Download ASTER Global Emissivity Dataset"""
    
    def download_aster_for_date(self, target_date: datetime, task_id: str) -> bool:
        """Download ASTER emissivity data (static dataset)"""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            print(f"Processing ASTER GED for {date_str}...")
            
            # ASTER Global Emissivity Dataset (static)
            aster_ged = ee.Image('NASA/ASTER_GED/AG100_003')
            
            # Define bands to download
            bands = [
                'emissivity_band10', 'emissivity_band11', 
                'emissivity_band12', 'emissivity_band13', 'ndvi'
            ]
            
            # Download the image
            extract_dir = download_ee_image(
                aster_ged,
                bands,
                self.region_geometry,
                DATASET_SCALES['aster'],
                self.output_dirs['aster'],
                f"aster_{date_str}"
            )
            
            if extract_dir:
                # Create organized zip file
                zip_path = self._create_aster_zip(extract_dir, task_id)
                
                # Add to mapping
                self.add_to_mapping(
                    task_id, 'ASTER', date_str, bands,
                    {'source': 'NASA/ASTER_GED/AG100_003', 'zip_path': zip_path}
                )
                
                return True
            
            return False
            
        except Exception as e:
            print(f"ASTER download failed for {target_date}: {e}")
            return False
    
    def _create_aster_zip(self, extract_dir: str, task_id: str) -> str:
        """Create organized ZIP file for ASTER data"""
        zip_filename = f"downloaded_aster_{task_id}.zip"
        zip_path = os.path.join(self.output_dirs['zips'], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(extract_dir):
                if file.endswith('.tif'):
                    file_path = os.path.join(extract_dir, file)
                    
                    # Rename files according to specification
                    if 'emissivity_band10' in file:
                        new_name = "ASTER_emissivity_band10.tif"
                    elif 'emissivity_band11' in file:
                        new_name = "ASTER_emissivity_band11.tif"
                    elif 'emissivity_band12' in file:
                        new_name = "ASTER_emissivity_band12.tif"
                    elif 'emissivity_band13' in file:
                        new_name = "ASTER_emissivity_band13.tif"
                    elif 'ndvi' in file:
                        new_name = "ASTER_ndvi.tif"
                    else:
                        new_name = file
                    
                    zipf.write(file_path, new_name)
        
        return zip_path

class LandsatDownloader(DatasetDownloader):
    """Download Landsat 8/9 data (both L1 and L2)"""
    
    def download_landsat_for_date(self, target_date: datetime, satellite: str, 
                                 processing_level: str, task_id: str) -> bool:
        """Download Landsat data for a specific date"""
        try:
            date_str = target_date.strftime('%Y-%m-%d')
            print(f"Processing Landsat {satellite} {processing_level} for {date_str}...")
            
            # Define collection based on satellite and processing level
            collection_name = self._get_collection_name(satellite, processing_level)
            if not collection_name:
                return False
            
            # Define date range (Landsat 16-day cycle)
            start_date = ee.Date(target_date.strftime('%Y-%m-%d')).advance(-8, 'day')
            end_date = ee.Date(target_date.strftime('%Y-%m-%d')).advance(8, 'day')
            
            # Get Landsat collection
            landsat_collection = ee.ImageCollection(collection_name) \
                .filterDate(start_date, end_date) \
                .filterBounds(self.region_geometry) \
                .filter(ee.Filter.lt('CLOUD_COVER', 80))
            
            if landsat_collection.size().getInfo() == 0:
                print(f"No Landsat {satellite} {processing_level} data found for {date_str}")
                return False
            
            # Get the best image (least cloudy, closest to target date)
            landsat_image = landsat_collection.sort('CLOUD_COVER').first()
            
            # Define bands based on processing level
            bands = self._get_bands(processing_level)
            
            # Get appropriate scale for processing level
            scale_key = f"landsat_{processing_level.lower()}"
            scale = DATASET_SCALES[scale_key]
            
            # Download the image
            dataset_key = f"landsat{satellite.lower()}_{processing_level.lower()}"
            extract_dir = download_ee_image(
                landsat_image,
                bands,
                self.region_geometry,
                scale,
                self.output_dirs[dataset_key],
                f"landsat{satellite.lower()}_{processing_level.lower()}_{date_str}"
            )
            
            if extract_dir:
                # Create organized zip file
                zip_path = self._create_landsat_zip(
                    extract_dir, task_id, satellite, processing_level, date_str
                )
                
                # Add to mapping
                self.add_to_mapping(
                    task_id, f'Landsat{satellite}_{processing_level}', date_str, bands,
                    {'source': collection_name, 'zip_path': zip_path}
                )
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Landsat {satellite} {processing_level} download failed for {target_date}: {e}")
            return False
    
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
            return ['B10', 'B11']  # Thermal bands
        elif processing_level == 'L2':
            return ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
        else:
            return []
    
    def _create_landsat_zip(self, extract_dir: str, task_id: str, satellite: str, 
                           processing_level: str, date_str: str) -> str:
        """Create organized ZIP file for Landsat data"""
        zip_filename = f"landsat{satellite.lower()}_{processing_level.lower()}_{task_id}.zip"
        zip_path = os.path.join(self.output_dirs['zips'], zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in os.listdir(extract_dir):
                if file.endswith('.tif'):
                    file_path = os.path.join(extract_dir, file)
                    
                    # Extract band name from filename
                    band_name = self._extract_band_name(file)
                    if band_name:
                        # Create new filename according to specification
                        new_name = f"L{satellite}_{processing_level}_{band_name}_{date_str.replace('-', '')}_00.tif"
                        zipf.write(file_path, new_name)
        
        return zip_path
    
    def _extract_band_name(self, filename: str) -> Optional[str]:
        """Extract band name from GEE downloaded filename"""
        # This is a simplified extraction - in practice, you might need more robust parsing
        for band in ['B10', 'B11', 'SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']:
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
        self.global_task_mapping = {}
        
        # Initialize downloaders
        self.era5_downloader = ERA5Downloader(self.output_dirs, region_geometry)
        self.aster_downloader = ASTERDownloader(self.output_dirs, region_geometry)
        self.landsat_downloader = LandsatDownloader(self.output_dirs, region_geometry)
    
    def download_all_for_date_range(self, start_date: datetime, end_date: datetime,
                                   datasets: List[str] = None) -> Dict:
        """Download all datasets for a date range"""
        if datasets is None:
            datasets = ['era5', 'aster', 'landsat8_l1', 'landsat8_l2', 'landsat9_l1', 'landsat9_l2']
        
        print(f"Starting downloads for date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Generate date list
        current_date = start_date
        dates_to_process = []
        while current_date <= end_date:
            dates_to_process.append(current_date)
            current_date += timedelta(days=1)
        
        print(f"Processing {len(dates_to_process)} dates for {len(datasets)} datasets")
        
        # Download each dataset for each date
        for date in dates_to_process:
            date_str = date.strftime('%Y-%m-%d')
            print(f"\n=== Processing date: {date_str} ===")
            
            for dataset in datasets:
                task_id = generate_task_id()
                success = False
                
                if dataset == 'era5':
                    success = self.era5_downloader.download_era5_for_date(date, task_id)
                    if success:
                        self.global_task_mapping.update(self.era5_downloader.task_mapping)
                
                elif dataset == 'aster':
                    success = self.aster_downloader.download_aster_for_date(date, task_id)
                    if success:
                        self.global_task_mapping.update(self.aster_downloader.task_mapping)
                
                elif dataset.startswith('landsat'):
                    # Parse dataset string: landsat8_l1, landsat8_l2, landsat9_l1, landsat9_l2
                    parts = dataset.split('_')
                    satellite = parts[0][-1]  # '8' or '9'
                    level = parts[1].upper()  # 'L1' or 'L2'
                    
                    success = self.landsat_downloader.download_landsat_for_date(
                        date, satellite, level, task_id
                    )
                    if success:
                        self.global_task_mapping.update(self.landsat_downloader.task_mapping)
                
                if success:
                    print(f"  ✓ {dataset} completed for {date_str}")
                else:
                    print(f"  ✗ {dataset} failed for {date_str}")
                
                # Small delay between downloads
                time.sleep(1)
        
        # Save task mapping to JSON
        self._save_task_mapping()
        
        # Cleanup temporary directories
        self._cleanup_temp_dirs()
        
        return self.global_task_mapping
    
    def _save_task_mapping(self):
        """Save task mapping to JSON file"""
        mapping_file = os.path.join(self.output_base_path, 'task_mapping.json')
        
        # Add summary statistics
        summary = {
            'total_tasks': len(self.global_task_mapping),
            'datasets': {},
            'date_range': {
                'start': None,
                'end': None
            },
            'generated_at': datetime.now().isoformat()
        }
        
        # Calculate dataset statistics
        for task_id, info in self.global_task_mapping.items():
            dataset_type = info['dataset_type']
            if dataset_type not in summary['datasets']:
                summary['datasets'][dataset_type] = 0
            summary['datasets'][dataset_type] += 1
            
            # Track date range
            date = info['date']
            if summary['date_range']['start'] is None or date < summary['date_range']['start']:
                summary['date_range']['start'] = date
            if summary['date_range']['end'] is None or date > summary['date_range']['end']:
                summary['date_range']['end'] = date
        
        # Create full mapping structure
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
        choices=['era5', 'aster', 'landsat8_l1', 'landsat8_l2', 'landsat9_l1', 'landsat9_l2'],
        default=['era5', 'aster', 'landsat8_l1', 'landsat8_l2', 'landsat9_l1', 'landsat9_l2'],
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
