#!/usr/bin/env python3
"""
Simple GEE to ZIP Converter

This script downloads data from Google Earth Engine using existing modules,
packages it into ZIP files with task IDs, and generates a JSON mapping.
You can then manually upload the ZIPs to your server.

Usage:
    python simple_gee_to_zip.py --roi_name "D-49-49-A" \
                                --start_date "2023-01-01" \
                                --end_date "2023-01-31"
"""

import argparse
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import ee

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SimpleGEEToZip:
    """Downloads data from GEE and packages into ZIP files with task IDs"""
    
    def __init__(self, output_folder: str = "./gee_zip_output"):
        self.output_folder = output_folder
        self.zip_folder = os.path.join(output_folder, "zips")
        self.task_id_mapping = {}
        
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
    
    def download_gee_data(self, roi_name: str, roi_geometry_ee, start_date: str, end_date: str, datasets: List[str]):
        """Download data from GEE using existing modules"""
        temp_download_folder = os.path.join(self.output_folder, "temp_downloads")
        os.makedirs(temp_download_folder, exist_ok=True)
        
        try:
            # Import GEE modules (add path)
            import sys
            sys.path.append('data_retrival_module')
            
            downloaded_data = {}
            
            if "era5" in datasets:
                logging.info("🌡️  Downloading ERA5 data...")
                try:
                    from era5_retriever import main as era5_main_retrieval
                    
                    # ERA5 requires LST files as reference, so we might need to create dummy ones
                    # or modify the function. For now, let's try with empty LST folder
                    lst_ref_folder = os.path.join(temp_download_folder, roi_name, 'lst')
                    os.makedirs(lst_ref_folder, exist_ok=True)
                    
                    era5_main_retrieval(
                        input_folder=lst_ref_folder,
                        output_folder=temp_download_folder
                    )
                    
                    # Find ERA5 files
                    era5_files = []
                    era5_dir = os.path.join(temp_download_folder, roi_name, 'era5')
                    if os.path.exists(era5_dir):
                        for f in os.listdir(era5_dir):
                            if f.endswith('.tif'):
                                era5_files.append(os.path.join(era5_dir, f))
                    
                    downloaded_data["era5"] = era5_files
                    logging.info(f"✅ Downloaded {len(era5_files)} ERA5 files")
                    
                except Exception as e:
                    logging.error(f"❌ ERA5 download failed: {e}")
                    downloaded_data["era5"] = []
            
            if "s2" in datasets:
                logging.info("🛰️  Downloading Sentinel-2 data...")
                try:
                    from s2_retrieval import main_s2_retrieval
                    
                    main_s2_retrieval(
                        roi_name=roi_name,
                        start_date=start_date,
                        end_date=end_date,
                        roi_geometry=roi_geometry_ee,
                        output_folder=temp_download_folder
                    )
                    
                    # Find S2 files
                    s2_files = []
                    s2_dir = os.path.join(temp_download_folder, roi_name, 's2_images')
                    if os.path.exists(s2_dir):
                        for f in os.listdir(s2_dir):
                            if f.endswith('.tif'):
                                s2_files.append(os.path.join(s2_dir, f))
                    
                    downloaded_data["s2"] = s2_files
                    logging.info(f"✅ Downloaded {len(s2_files)} S2 files")
                    
                except Exception as e:
                    logging.error(f"❌ S2 download failed: {e}")
                    downloaded_data["s2"] = []
            
            if "lst" in datasets:
                logging.info("🌡️  Downloading LST data...")
                try:
                    from lst_retrieval import lst_retrive
                    
                    lst_retrive(
                        start_date=start_date,
                        end_date=end_date,
                        roi_geometry=roi_geometry_ee,
                        roi_name=roi_name,
                        output_folder=temp_download_folder
                    )
                    
                    # Find LST files (organized by type)
                    lst_files = {
                        "L8_L1": [],
                        "L8_L2": [],
                        "L9_L1": [],
                        "L9_L2": [],
                        "aster": []
                    }
                    
                    # Check LST directory
                    lst_dir = os.path.join(temp_download_folder, roi_name, 'lst')
                    if os.path.exists(lst_dir):
                        for f in os.listdir(lst_dir):
                            if f.endswith('.tif'):
                                file_path = os.path.join(lst_dir, f)
                                if 'L8_lst' in f:
                                    lst_files["L8_L2"].append(file_path)
                                elif 'L9_lst' in f:
                                    lst_files["L9_L2"].append(file_path)
                    
                    # Check ASTER directory
                    aster_dir = os.path.join(temp_download_folder, roi_name, 'aster_ged')
                    if os.path.exists(aster_dir):
                        for f in os.listdir(aster_dir):
                            if f.endswith('.tif'):
                                lst_files["aster"].append(os.path.join(aster_dir, f))
                    
                    downloaded_data["lst"] = lst_files
                    total_files = sum(len(files) for files in lst_files.values())
                    logging.info(f"✅ Downloaded {total_files} LST/ASTER files")
                    
                except Exception as e:
                    logging.error(f"❌ LST download failed: {e}")
                    downloaded_data["lst"] = {"L8_L1": [], "L8_L2": [], "L9_L1": [], "L9_L2": [], "aster": []}
            
            return downloaded_data
            
        finally:
            # Clean up temp download folder
            if os.path.exists(temp_download_folder):
                shutil.rmtree(temp_download_folder, ignore_errors=True)
    
    def create_zip_file(self, files: List[str], dataset_name: str, roi_name: str) -> tuple:
        """Create ZIP file from list of files"""
        if not files:
            return None, None
        
        task_id = str(uuid.uuid4())
        zip_filename = f"{dataset_name}_{roi_name}_{task_id}.zip"
        zip_path = os.path.join(self.zip_folder, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                if os.path.exists(file_path):
                    # Organize files in subdirectories by date if possible
                    filename = os.path.basename(file_path)
                    
                    # Try to extract date from filename
                    if '_' in filename:
                        parts = filename.split('_')
                        date_part = None
                        for part in parts:
                            if len(part) == 10 and part.count('-') == 2:  # YYYY-MM-DD format
                                date_part = part
                                break
                        
                        if date_part:
                            # Create subdirectory structure: dataset_YYYYMMDD_HHMMSSZ/filename.tif
                            subdir = f"{dataset_name}_{date_part.replace('-', '')}_100000Z"
                            archive_name = f"{subdir}/{filename}"
                        else:
                            archive_name = filename
                    else:
                        archive_name = filename
                    
                    zip_file.write(file_path, archive_name)
        
        file_size = os.path.getsize(zip_path)
        logging.info(f"📦 Created ZIP: {zip_filename} ({len(files)} files, {file_size//1024} KB)")
        return zip_path, task_id
    
    def process_downloaded_data(self, downloaded_data: Dict, roi_name: str):
        """Process downloaded data into ZIP files and generate task mapping"""
        
        for dataset_name, data in downloaded_data.items():
            if dataset_name == "lst":
                # LST has nested structure
                if "lst" not in self.task_id_mapping:
                    self.task_id_mapping["lst"] = {}
                
                for category, files in data.items():
                    if files:
                        zip_path, task_id = self.create_zip_file(files, f"lst_{category}", roi_name)
                        if task_id:
                            self.task_id_mapping["lst"][category] = [task_id]
            else:
                # Simple datasets (era5, s2, aster)
                if data:  # Check if files exist
                    zip_path, task_id = self.create_zip_file(data, dataset_name, roi_name)
                    if task_id:
                        self.task_id_mapping[dataset_name] = [task_id]
    
    def save_task_mapping(self, output_file: str):
        """Save task ID mapping to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.task_id_mapping, f, indent=2)
        
        logging.info(f"💾 Saved task ID mapping to {output_file}")
        
        print(f"\n📋 Generated Task ID Mapping:")
        print("=" * 50)
        print(json.dumps(self.task_id_mapping, indent=2))
        print("=" * 50)
        
        # Also save a copy with clear instructions
        readme_content = f"""# Generated Task ID Mapping

This file contains the task IDs for data downloaded from GEE on {datetime.now().isoformat()}.

## ZIP Files Location
The ZIP files are located in: {self.zip_folder}

## Upload Instructions
1. Upload each ZIP file to your server using the corresponding endpoint:
   - ERA5: POST /v1/era5_download/{task_id}
   - S2: POST /v1/s2_download/{task_id}
   - Landsat 8 L1: POST /v1/landsat8_l1_download/{task_id}
   - Landsat 8 L2: POST /v1/landsat8_l2_download/{task_id}
   - Landsat 9 L1: POST /v1/landsat9_l1_download/{task_id}
   - Landsat 9 L2: POST /v1/landsat9_l2_download/{task_id}
   - ASTER: POST /v1/aster_download/{task_id}

2. Use the task_ids.json file with the ETL module:
   ```bash
   python -m ETL_data_retrieval_module.main \\
       --roi_name "your_roi" \\
       --task_ids "{output_file}" \\
       --datasets era5 s2 lst aster
   ```

## Task ID Mapping:
{json.dumps(self.task_id_mapping, indent=2)}
"""
        
        readme_file = output_file.replace('.json', '_README.md')
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"📖 Created instructions: {readme_file}")


def main():
    parser = argparse.ArgumentParser(description="Download data from GEE and create ZIP files with task IDs")
    parser.add_argument("--roi_name", required=True, help="Grid PhienHieu / ROI name")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--grid_file", default="data/Grid_50K_MatchedDates.geojson", help="Grid file path")
    parser.add_argument("--datasets", nargs="+", choices=["era5", "s2", "lst", "aster"], 
                       default=["era5", "s2", "lst"], help="Datasets to download and package")
    parser.add_argument("--output_folder", default="./gee_zip_output", help="Local output folder")
    parser.add_argument("--task_mapping_file", default="task_ids_from_gee.json", 
                       help="Output file for task ID mapping")

    args = parser.parse_args()

    # Initialize converter
    converter = SimpleGEEToZip(args.output_folder)
    
    try:
        # Initialize GEE
        converter.initialize_gee()
        
        # Find ROI geometry
        feature = converter.find_grid_feature(args.roi_name, args.grid_file)
        roi_geometry_data = feature['geometry']
        roi_geometry_ee = ee.Geometry.Polygon(roi_geometry_data['coordinates'])
        
        logging.info(f"🎯 Processing ROI: {args.roi_name}")
        logging.info(f"📅 Date range: {args.start_date} to {args.end_date}")
        logging.info(f"🗂️  Datasets: {args.datasets}")
        
        # Download data from GEE
        downloaded_data = converter.download_gee_data(
            args.roi_name, roi_geometry_ee, 
            args.start_date, args.end_date, 
            args.datasets
        )
        
        # Process into ZIP files
        converter.process_downloaded_data(downloaded_data, args.roi_name)
        
        # Save task mapping
        converter.save_task_mapping(args.task_mapping_file)
        
        logging.info("🎉 GEE to ZIP conversion completed successfully!")
        print(f"\n📁 ZIP files created in: {converter.zip_folder}")
        print(f"📋 Task mapping saved to: {args.task_mapping_file}")
        
    except Exception as e:
        logging.error(f"💥 Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()
