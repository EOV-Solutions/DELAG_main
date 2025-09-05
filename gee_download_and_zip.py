#!/usr/bin/env python3
"""
GEE Download and ZIP Creator

Downloads data from Google Earth Engine using existing modules and creates
properly formatted ZIP files that match what the ETL_data_retrieval_module expects.

Usage:
    python gee_download_and_zip.py --roi_name "D-49-49-A" \
                                  --start_date "2023-01-01" \
                                  --end_date "2023-01-31" \
                                  --datasets era5 s2 lst
"""

import argparse
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import ee

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GEEZipCreator:
    """Downloads data from GEE and creates ETL-compatible ZIP files"""
    
    def __init__(self, output_folder: str = "./gee_etl_ready"):
        self.output_folder = output_folder
        self.zip_folder = os.path.join(output_folder, "zips")
        self.task_id_mapping = {}
        self.lst_reference_files = []  # Store LST files for ERA5 reference
        
        # Create directories
        os.makedirs(self.zip_folder, exist_ok=True)
        
    def initialize_gee(self):
        """Initialize Google Earth Engine"""
        try:
            ee.Initialize(project='ee-hadat-461702-p4')
            logging.info("✅ Google Earth Engine initialized")
        except Exception:
            logging.info("🔐 Authenticating to Google Earth Engine...")
            ee.Authenticate()
            ee.Initialize(project='ee-hadat-461702-p4')
            logging.info("✅ Google Earth Engine authenticated and initialized")
    
    def find_grid_feature(self, roi_name: str, grid_file: str) -> dict:
        """Find ROI feature in grid file"""
        try:
            with open(grid_file, 'r') as f:
                grid_data = json.load(f)
            
            for feature in grid_data.get('features', []):
                if feature.get('properties', {}).get('PhienHieu') == roi_name:
                    return feature
            
            raise ValueError(f"ROI '{roi_name}' not found in grid file")
        except Exception as e:
            logging.error(f"Error reading grid file: {e}")
            raise
    
    def download_and_process_era5(self, roi_name: str, roi_geometry_ee, start_date: str, end_date: str, reference_lst_files: List[str] = None) -> str:
        """Download ERA5 data and create ETL-compatible ZIP"""
        logging.info("🌡️  Downloading ERA5 data...")
        
        temp_folder = os.path.join(self.output_folder, "temp_era5")
        os.makedirs(temp_folder, exist_ok=True)
        
        try:
            # Import and use ERA5 module
            import sys
            sys.path.append('data_retrival_module')
            from era5_retriever import main as era5_main_retrieval
            
            # Create LST reference folder and copy reference files
            lst_ref_folder = os.path.join(temp_folder, roi_name, 'lst')
            os.makedirs(lst_ref_folder, exist_ok=True)
            
            # If we have reference LST files from previous processing, copy them
            if reference_lst_files:
                # Only copy valid GeoTIFF files
                valid_files = []
                for lst_file in reference_lst_files:
                    if os.path.exists(lst_file):
                        try:
                            # Test if file is a valid GeoTIFF
                            import rasterio
                            with rasterio.open(lst_file) as src:
                                _ = src.read(1, masked=True)  # Try to read first band
                            shutil.copy2(lst_file, lst_ref_folder)
                            valid_files.append(lst_file)
                            logging.info(f"  📄 Copied LST reference: {os.path.basename(lst_file)}")
                        except Exception as e:
                            logging.warning(f"  ⚠️  Skipping invalid reference file {os.path.basename(lst_file)}: {e}")
                
                if not valid_files:
                    logging.warning("  ⚠️  No valid LST reference files - ERA5 may not work properly")
                    return None
            else:
                # Create a dummy LST file for ERA5 reference if none available
                logging.warning("  ⚠️  No LST reference files - ERA5 may not work properly")
                # Skip ERA5 processing if no reference
                return None
            
            # Download ERA5 data
            era5_main_retrieval(
                input_folder=lst_ref_folder,
                output_folder=temp_folder
            )
            
            # Find and rename ERA5 files to match ETL expectations
            era5_files = []
            era5_dir = os.path.join(temp_folder, roi_name, 'era5')
            
            if os.path.exists(era5_dir):
                for filename in os.listdir(era5_dir):
                    if filename.endswith('.tif'):
                        old_path = os.path.join(era5_dir, filename)
                        
                        # Extract date from original filename and rename to ETL format
                        # ETL expects: skin_temperature_era5_YYYYMMDD_HHMMSSZ.tif
                        date_str = self._extract_date_from_era5_filename(filename)
                        if date_str:
                            new_filename = f"skin_temperature_era5_{date_str}_100000Z.tif"
                            new_path = os.path.join(era5_dir, new_filename)
                            shutil.move(old_path, new_path)
                            era5_files.append(new_path)
                            logging.info(f"  📄 Renamed: {filename} → {new_filename}")
            
            # Create ZIP file
            if era5_files:
                task_id = str(uuid.uuid4())
                zip_path = os.path.join(self.zip_folder, f"era5_{task_id}.zip")
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in era5_files:
                        # Add files in subdirectory structure that ETL can handle
                        arc_name = f"era5_{datetime.now().strftime('%Y%m%d_%H%M%S')}/{os.path.basename(file_path)}"
                        zip_file.write(file_path, arc_name)
                
                logging.info(f"📦 Created ERA5 ZIP: {os.path.basename(zip_path)} ({len(era5_files)} files)")
                
                # Add to task mapping (don't overwrite)
                if "era5" not in self.task_id_mapping:
                    self.task_id_mapping["era5"] = []
                self.task_id_mapping["era5"].append(task_id)
                return zip_path
                
        except Exception as e:
            logging.error(f"❌ ERA5 processing failed: {e}")
        finally:
            # Cleanup temp folder
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder, ignore_errors=True)
        
        return None
    
    def download_and_process_s2(self, roi_name: str, roi_geometry_ee, start_date: str, end_date: str) -> str:
        """Download raw S2 data and create ETL-compatible ZIP"""
        logging.info("🛰️  Downloading Sentinel-2 data...")
        
        temp_folder = os.path.join(self.output_folder, "temp_s2")
        os.makedirs(temp_folder, exist_ok=True)
        
        s2_files = []  # Initialize the variable
        
        try:
            # Download raw S2 data directly
            s2_files = self._download_s2_raw(roi_geometry_ee, start_date, end_date, temp_folder)
            
            # Create ZIP file
            if s2_files:
                task_id = str(uuid.uuid4())
                zip_path = os.path.join(self.zip_folder, f"s2_{task_id}.zip")
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in s2_files:
                        # Organize in date-based subdirectories
                        date_str = self._extract_date_from_s2_filename(os.path.basename(file_path))
                        if date_str:
                            arc_name = f"s2_{date_str.replace('-', '')}/{os.path.basename(file_path)}"
                        else:
                            arc_name = os.path.basename(file_path)
                        zip_file.write(file_path, arc_name)
                
                logging.info(f"📦 Created S2 ZIP: {os.path.basename(zip_path)} ({len(s2_files)} files)")
                
                # Add to task mapping (don't overwrite)
                if "s2" not in self.task_id_mapping:
                    self.task_id_mapping["s2"] = []
                self.task_id_mapping["s2"].append(task_id)
                return zip_path
                
        except Exception as e:
            logging.error(f"❌ S2 processing failed: {e}")
        finally:
            # Cleanup temp folder and extracted files
            if os.path.exists(temp_folder):
                shutil.rmtree(temp_folder, ignore_errors=True)
            for file_path in s2_files:
                if os.path.exists(file_path):
                    os.remove(file_path)
        
        return None
    
    def download_and_process_lst(self, roi_name: str, roi_geometry_ee, start_date: str, end_date: str) -> Dict[str, str]:
        """Download raw Landsat L1/L2 and ASTER data for LST processing"""
        logging.info("🌡️  Downloading raw Landsat and ASTER data...")
        
        temp_folder = os.path.join(self.output_folder, "temp_lst")
        os.makedirs(temp_folder, exist_ok=True)
        
        created_zips = {}
        lst_files_for_era5 = []  # Track LST files for ERA5 reference
        
        try:
            # Download raw Landsat and ASTER data separately
            l8_l1_files = self._download_landsat_l1(roi_geometry_ee, start_date, end_date, "L8", temp_folder)
            l8_l2_files = self._download_landsat_l2(roi_geometry_ee, start_date, end_date, "L8", temp_folder)
            l9_l1_files = self._download_landsat_l1(roi_geometry_ee, start_date, end_date, "L9", temp_folder)
            l9_l2_files = self._download_landsat_l2(roi_geometry_ee, start_date, end_date, "L9", temp_folder)
            aster_files = self._download_aster_ged(roi_geometry_ee, temp_folder)
            
            # Create ZIPs for each dataset type
            if l8_l1_files:
                task_id = str(uuid.uuid4())
                zip_path = os.path.join(self.zip_folder, f"landsat8_l1_{task_id}.zip")
                self._create_landsat_zip(l8_l1_files, zip_path, "L8_L1")
                created_zips["L8_L1"] = [task_id]
                
            if l8_l2_files:
                task_id = str(uuid.uuid4())
                zip_path = os.path.join(self.zip_folder, f"landsat8_l2_{task_id}.zip")
                self._create_landsat_zip(l8_l2_files, zip_path, "L8_L2")
                created_zips["L8_L2"] = [task_id]
                lst_files_for_era5.extend(l8_l2_files)  # Add to ERA5 reference
                
            if l9_l1_files:
                task_id = str(uuid.uuid4())
                zip_path = os.path.join(self.zip_folder, f"landsat9_l1_{task_id}.zip")
                self._create_landsat_zip(l9_l1_files, zip_path, "L9_L1")
                created_zips["L9_L1"] = [task_id]
                
            if l9_l2_files:
                task_id = str(uuid.uuid4())
                zip_path = os.path.join(self.zip_folder, f"landsat9_l2_{task_id}.zip")
                self._create_landsat_zip(l9_l2_files, zip_path, "L9_L2")
                created_zips["L9_L2"] = [task_id]
                lst_files_for_era5.extend(l9_l2_files)  # Add to ERA5 reference
                
            if aster_files:
                task_id = str(uuid.uuid4())
                zip_path = os.path.join(self.zip_folder, f"aster_{task_id}.zip")
                self._create_aster_zip(aster_files, zip_path)
                created_zips["aster"] = [task_id]
                
                # Add ASTER as separate dataset (not under lst)
                if "aster" not in self.task_id_mapping:
                    self.task_id_mapping["aster"] = []
                self.task_id_mapping["aster"].append(task_id)
                
                # Remove ASTER from created_zips to avoid duplication under lst
                created_zips.pop("aster", None)
            
            if created_zips:
                # Add to task mapping (don't overwrite)
                if "lst" not in self.task_id_mapping:
                    self.task_id_mapping["lst"] = {}
                self.task_id_mapping["lst"].update(created_zips)
                
            # Store LST files for ERA5 reference
            self.lst_reference_files = lst_files_for_era5
                
        except Exception as e:
            logging.error(f"❌ LST processing failed: {e}")
        finally:
            # Don't cleanup temp folder immediately - ERA5 might need LST reference files
            pass
        
        return created_zips
    
    def _extract_date_from_era5_filename(self, filename: str) -> str:
        """Extract date from ERA5 filename and convert to YYYYMMDD format"""
        # Try to find date patterns in ERA5 filenames
        import re
        
        # Look for YYYY-MM-DD pattern
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            date_str = date_match.group(1)
            return date_str.replace('-', '')
        
        # Look for YYYYMMDD pattern
        date_match = re.search(r'(\d{8})', filename)
        if date_match:
            return date_match.group(1)
        
        # Default to current date if can't extract
        return datetime.now().strftime('%Y%m%d')
    
    def _extract_date_from_s2_filename(self, filename: str) -> str:
        """Extract date from S2 filename"""
        import re
        
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            return date_match.group(1)
        
        return None
    
    def _extract_s2_bands(self, file_path: str, original_filename: str) -> List[str]:
        """Extract individual bands from multi-band S2 file"""
        import rasterio
        
        extracted_files = []
        
        try:
            with rasterio.open(file_path) as src:
                # Extract date from filename
                date_str = self._extract_date_from_s2_filename(original_filename)
                if not date_str:
                    date_str = datetime.now().strftime('%Y-%m-%d')
                
                bands = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR
                
                for i, band in enumerate(bands, 1):
                    if i <= src.count:
                        # Create individual band file
                        band_filename = f"S2_{band}_{date_str.replace('-', '')}.tif"
                        band_path = os.path.join(self.output_folder, "temp_bands", band_filename)
                        
                        os.makedirs(os.path.dirname(band_path), exist_ok=True)
                        
                        # Copy band data
                        profile = src.profile.copy()
                        profile.update(count=1)
                        
                        with rasterio.open(band_path, 'w', **profile) as dst:
                            dst.write(src.read(i), 1)
                        
                        extracted_files.append(band_path)
                        
        except Exception as e:
            logging.error(f"Failed to extract bands from {file_path}: {e}")
        
        return extracted_files
    
    def _download_landsat_l1(self, roi_geometry_ee, start_date: str, end_date: str, satellite: str, temp_folder: str) -> List[str]:
        """Download raw Landsat L1 (TOA) data"""
        import requests
        import tempfile
        
        # Define Landsat collections
        collections = {
            "L8": "LANDSAT/LC08/C02/T1_TOA",
            "L9": "LANDSAT/LC09/C02/T1_TOA"
        }
        
        if satellite not in collections:
            return []
        
        collection = ee.ImageCollection(collections[satellite]) \
            .filterDate(start_date, end_date) \
            .filterBounds(roi_geometry_ee) \
            .limit(5)  # Limit for testing
        
        # Get the required bands for L1 (TOA)
        l1_bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'QA_PIXEL']
        
        downloaded_files = []
        image_list = collection.toList(collection.size())
        size = collection.size().getInfo()
        
        for i in range(min(size, 2)):  # Process max 2 images for testing
            image = ee.Image(image_list.get(i))
            date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            for band in l1_bands:
                try:
                    if image.bandNames().contains(band).getInfo():
                        temp_dir = tempfile.mkdtemp()
                        band_image = image.select(band)
                        
                        params = {
                            'scale': 30,
                            'region': roi_geometry_ee,
                            'fileFormat': 'GeoTIFF',
                            'crs': 'EPSG:4326'
                        }
                        download_url = band_image.getDownloadURL(params)
                        response = requests.get(download_url, timeout=120)
                        
                        if response.status_code == 200:
                            file_path = os.path.join(temp_folder, f"{satellite}_L1_{band}_{date_str.replace('-', '')}_{i:02d}.tif")
                            with open(file_path, 'wb') as f:
                                f.write(response.content)
                            downloaded_files.append(file_path)
                            logging.info(f"  📄 Downloaded {satellite} L1 {band}: {date_str}")
                            
                except Exception as e:
                    logging.warning(f"Failed to download {satellite} L1 {band}: {e}")
        
        return downloaded_files
    
    def _download_landsat_l2(self, roi_geometry_ee, start_date: str, end_date: str, satellite: str, temp_folder: str) -> List[str]:
        """Download raw Landsat L2 (Surface Reflectance) data"""
        import requests
        import tempfile
        
        # Define Landsat collections
        collections = {
            "L8": "LANDSAT/LC08/C02/T1_L2",
            "L9": "LANDSAT/LC09/C02/T1_L2"
        }
        
        if satellite not in collections:
            return []
        
        collection = ee.ImageCollection(collections[satellite]) \
            .filterDate(start_date, end_date) \
            .filterBounds(roi_geometry_ee) \
            .limit(5)  # Limit for testing
        
        # Get the required bands for L2 (SR)
        l2_bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10', 'QA_PIXEL']
        
        downloaded_files = []
        image_list = collection.toList(collection.size())
        size = collection.size().getInfo()
        
        for i in range(min(size, 2)):  # Process max 2 images for testing
            image = ee.Image(image_list.get(i))
            date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            for band in l2_bands:
                try:
                    if image.bandNames().contains(band).getInfo():
                        temp_dir = tempfile.mkdtemp()
                        band_image = image.select(band)
                        
                        params = {
                            'scale': 30,
                            'region': roi_geometry_ee,
                            'fileFormat': 'GeoTIFF',
                            'crs': 'EPSG:4326'
                        }
                        download_url = band_image.getDownloadURL(params)
                        response = requests.get(download_url, timeout=120)
                        
                        if response.status_code == 200:
                            file_path = os.path.join(temp_folder, f"{satellite}_L2_{band}_{date_str.replace('-', '')}_{i:02d}.tif")
                            with open(file_path, 'wb') as f:
                                f.write(response.content)
                            downloaded_files.append(file_path)
                            logging.info(f"  📄 Downloaded {satellite} L2 {band}: {date_str}")
                            
                except Exception as e:
                    logging.warning(f"Failed to download {satellite} L2 {band}: {e}")
        
        return downloaded_files
    
    def _download_aster_ged(self, roi_geometry_ee, temp_folder: str) -> List[str]:
        """Download ASTER GED emissivity data"""
        import requests
        import tempfile
        
        # ASTER GED image (it's a single Image, not ImageCollection)
        aster_image = ee.Image("NASA/ASTER_GED/AG100_003")
        
        # Required ASTER bands
        aster_bands = ['emissivity_band10', 'emissivity_band11', 'emissivity_band12', 'emissivity_band13', 'emissivity_band14', 'ndvi']
        
        downloaded_files = []
        
        for band in aster_bands:
            try:
                if aster_image.bandNames().contains(band).getInfo():
                    temp_dir = tempfile.mkdtemp()
                    band_image = aster_image.select(band)
                    
                    params = {
                        'scale': 100,  # ASTER GED is 100m resolution
                        'region': roi_geometry_ee,
                        'fileFormat': 'GeoTIFF',
                        'crs': 'EPSG:4326'
                    }
                    download_url = band_image.getDownloadURL(params)
                    response = requests.get(download_url, timeout=120)
                    
                    if response.status_code == 200:
                        file_path = os.path.join(temp_folder, f"ASTER_{band}.tif")
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        downloaded_files.append(file_path)
                        logging.info(f"  📄 Downloaded ASTER {band}")
                        
            except Exception as e:
                logging.warning(f"Failed to download ASTER {band}: {e}")
        
        return downloaded_files
    
    def _download_s2_raw(self, roi_geometry_ee, start_date: str, end_date: str, temp_folder: str) -> List[str]:
        """Download raw Sentinel-2 data"""
        import requests
        import tempfile
        
        # S2 collection
        s2_collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
            .filterDate(start_date, end_date) \
            .filterBounds(roi_geometry_ee) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10)) \
            .limit(5)  # Limit for testing
        
        # Required S2 bands  
        s2_bands = ['B2', 'B3', 'B4', 'B8']  # Blue, Green, Red, NIR
        
        downloaded_files = []
        image_list = s2_collection.toList(s2_collection.size())
        size = s2_collection.size().getInfo()
        
        if size == 0:
            logging.warning("No S2 images found for the specified criteria")
            return []
        
        for i in range(min(size, 2)):  # Process max 2 images for testing
            image = ee.Image(image_list.get(i))
            date_str = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            
            for band in s2_bands:
                try:
                    if image.bandNames().contains(band).getInfo():
                        temp_dir = tempfile.mkdtemp()
                        band_image = image.select(band)
                        
                        params = {
                            'scale': 10,  # S2 is 10m resolution for visible bands
                            'region': roi_geometry_ee,
                            'fileFormat': 'GeoTIFF',
                            'crs': 'EPSG:4326'
                        }
                        download_url = band_image.getDownloadURL(params)
                        response = requests.get(download_url, timeout=120)
                        
                        if response.status_code == 200:
                            file_path = os.path.join(temp_folder, f"S2_{band}_{date_str.replace('-', '')}_{i:02d}.tif")
                            with open(file_path, 'wb') as f:
                                f.write(response.content)
                            downloaded_files.append(file_path)
                            logging.info(f"  📄 Downloaded S2 {band}: {date_str}")
                            
                except Exception as e:
                    logging.warning(f"Failed to download S2 {band}: {e}")
        
        return downloaded_files
    
    def _create_landsat_zip(self, files: List[str], zip_path: str, dataset_type: str):
        """Create Landsat ZIP file"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                # Keep original filenames for ETL processing
                zip_file.write(file_path, os.path.basename(file_path))
        
        logging.info(f"📦 Created {dataset_type} ZIP: {os.path.basename(zip_path)} ({len(files)} files)")
    
    def _create_aster_zip(self, files: List[str], zip_path: str):
        """Create ASTER ZIP file"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                # Keep original filenames for ETL processing
                zip_file.write(file_path, os.path.basename(file_path))
        
        logging.info(f"📦 Created ASTER ZIP: {os.path.basename(zip_path)} ({len(files)} files)")
    
    def _rename_aster_file(self, filename: str) -> str:
        """Rename ASTER file to match ETL expectations"""
        # Map original ASTER filenames to expected patterns
        if 'emissivity' in filename.lower():
            if 'band10' in filename or 'b10' in filename:
                return "ASTER_emissivity_band10.tif"
            elif 'band11' in filename or 'b11' in filename:
                return "ASTER_emissivity_band11.tif"
            elif 'band12' in filename or 'b12' in filename:
                return "ASTER_emissivity_band12.tif"
            elif 'band13' in filename or 'b13' in filename:
                return "ASTER_emissivity_band13.tif"
            elif 'band14' in filename or 'b14' in filename:
                return "ASTER_emissivity_band14.tif"
        elif 'ndvi' in filename.lower():
            return "ASTER_ndvi.tif"
        
        # Keep original name if no pattern match
        return filename
    
    def save_task_mapping(self, output_file: str):
        """Save task ID mapping to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.task_id_mapping, f, indent=2)
        
        logging.info(f"💾 Saved task ID mapping to {output_file}")
        
        print(f"\n📋 Generated Task ID Mapping for ETL Module:")
        print("=" * 60)
        print(json.dumps(self.task_id_mapping, indent=2))
        print("=" * 60)
        
        # Create usage instructions
        instructions = f"""
# ETL Processing Instructions

Use this command to process the data with the ETL module:

```bash
python -m ETL_data_retrieval_module.main \\
    --roi_name "your_roi_name" \\
    --task_ids "{output_file}" \\
    --datasets {' '.join(self.task_id_mapping.keys())} \\
    --output_folder "./processed_data"
```

# Manual Server Setup

If you want to set up a simple server for the ZIP files:

1. Start the upload server:
```bash
python simple_upload_server.py --port 8000 --storage_dir ./server_storage
```

2. Copy ZIP files to server storage:
"""
        
        for dataset, task_ids in self.task_id_mapping.items():
            if dataset == "lst":
                for category, task_id_list in task_ids.items():
                    if isinstance(task_id_list, list):
                        for task_id in task_id_list:
                            zip_name = f"{category.lower().replace('_', '')}_{task_id}.zip"
                            instructions += f"\ncp {self.zip_folder}/{zip_name} ./server_storage/{category.lower().replace('_', '')}/\n"
            else:
                if isinstance(task_ids, list):
                    for task_id in task_ids:
                        zip_name = f"{dataset}_{task_id}.zip"
                        instructions += f"cp {self.zip_folder}/{zip_name} ./server_storage/{dataset}/\n"
        
        readme_file = output_file.replace('.json', '_INSTRUCTIONS.md')
        with open(readme_file, 'w') as f:
            f.write(instructions)
        
        print(f"📖 Created instructions: {readme_file}")


def main():
    parser = argparse.ArgumentParser(description="Download data from GEE and create ETL-compatible ZIP files")
    parser.add_argument("--roi_name", required=True, help="Grid PhienHieu / ROI name")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--grid_file", default="data/Grid_50K_MatchedDates.geojson", help="Grid file path")
    parser.add_argument("--datasets", nargs="+", choices=["era5", "s2", "lst"], 
                       default=["era5", "s2", "lst"], help="Datasets to download and package")
    parser.add_argument("--output_folder", default="./gee_etl_ready", help="Local output folder")
    parser.add_argument("--task_mapping_file", default="etl_task_mapping.json", 
                       help="Output file for task ID mapping")

    args = parser.parse_args()

    # Initialize creator
    creator = GEEZipCreator(args.output_folder)
    
    try:
        # Initialize GEE
        creator.initialize_gee()
        
        # Find ROI geometry
        feature = creator.find_grid_feature(args.roi_name, args.grid_file)
        roi_geometry_data = feature['geometry']
        roi_geometry_ee = ee.Geometry.Polygon(roi_geometry_data['coordinates'])
        
        logging.info(f"🎯 Processing ROI: {args.roi_name}")
        logging.info(f"📅 Date range: {args.start_date} to {args.end_date}")
        logging.info(f"🗂️  Datasets: {args.datasets}")
        
        # Process datasets in order (LST first to provide reference for ERA5)
        if "lst" in args.datasets:
            creator.download_and_process_lst(args.roi_name, roi_geometry_ee, args.start_date, args.end_date)
        
        if "s2" in args.datasets:
            creator.download_and_process_s2(args.roi_name, roi_geometry_ee, args.start_date, args.end_date)
        
        if "era5" in args.datasets:
            # Use LST files as reference if available
            creator.download_and_process_era5(args.roi_name, roi_geometry_ee, args.start_date, args.end_date, creator.lst_reference_files)
        
        # Save task mapping
        creator.save_task_mapping(args.task_mapping_file)
        
        logging.info("🎉 GEE download and ZIP creation completed!")
        print(f"\n📁 ZIP files created in: {creator.zip_folder}")
        print(f"📋 Task mapping saved to: {args.task_mapping_file}")
        
        # Cleanup all temp folders at the end
        for temp_folder_name in ["temp_lst", "temp_era5", "temp_s2", "temp_bands"]:
            temp_path = os.path.join(args.output_folder, temp_folder_name)
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)
        
    except Exception as e:
        logging.error(f"💥 Process failed: {e}")
        raise


if __name__ == "__main__":
    main()
