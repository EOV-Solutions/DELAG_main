#!/usr/bin/env python3
"""
GEE to Server Data Uploader

This script downloads data from Google Earth Engine using the existing data_retrival_module,
packages it into ZIP files, uploads to the server, and generates task ID mappings.

Usage:
    python gee_to_server_uploader.py --roi_name "D-49-49-A" \
                                    --start_date "2023-01-01" \
                                    --end_date "2023-01-31" \
                                    --server_url "http://localhost:8000" \
                                    --datasets era5 s2 lst aster
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import ee
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gee_to_server_upload.log')
    ]
)

class GEEToServerUploader:
    """Downloads data from GEE and uploads to server with task ID mapping"""
    
    def __init__(self, server_url: str, output_folder: str = "./gee_downloads"):
        self.server_url = server_url.rstrip('/')
        self.output_folder = output_folder
        self.task_id_mapping = {}
        
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
    
    def download_era5_data(self, roi_name: str, roi_geometry_ee, start_date: str, end_date: str) -> List[str]:
        """Download ERA5 data using existing module"""
        logging.info(f"🌡️  Downloading ERA5 data for {roi_name}")
        
        try:
            # Import the ERA5 retrieval module
            import sys
            sys.path.append('data_retrival_module')
            from era5_retriever import main as era5_main_retrieval
            
            era5_output_folder = os.path.join(self.output_folder, roi_name)
            os.makedirs(era5_output_folder, exist_ok=True)
            
            # Download ERA5 data
            era5_main_retrieval(
                input_folder=os.path.join(era5_output_folder, 'lst'),  # ERA5 looks for LST files here
                output_folder=era5_output_folder
            )
            
            # Find downloaded ERA5 files
            era5_files = []
            era5_dir = os.path.join(era5_output_folder, roi_name, 'era5')
            if os.path.exists(era5_dir):
                for f in os.listdir(era5_dir):
                    if f.endswith('.tif'):
                        era5_files.append(os.path.join(era5_dir, f))
            
            logging.info(f"✅ Downloaded {len(era5_files)} ERA5 files")
            return era5_files
            
        except Exception as e:
            logging.error(f"❌ Failed to download ERA5 data: {e}")
            return []
    
    def download_s2_data(self, roi_name: str, roi_geometry_ee, start_date: str, end_date: str) -> List[str]:
        """Download Sentinel-2 data using existing module"""
        logging.info(f"🛰️  Downloading Sentinel-2 data for {roi_name}")
        
        try:
            # Import the S2 retrieval module
            import sys
            sys.path.append('data_retrival_module')
            from s2_retrieval import main_s2_retrieval
            
            s2_output_folder = os.path.join(self.output_folder, roi_name)
            os.makedirs(s2_output_folder, exist_ok=True)
            
            # Download S2 data
            main_s2_retrieval(
                roi_name=roi_name,
                start_date=start_date,
                end_date=end_date,
                roi_geometry=roi_geometry_ee,
                output_folder=s2_output_folder
            )
            
            # Find downloaded S2 files
            s2_files = []
            s2_dir = os.path.join(s2_output_folder, roi_name, 's2_images')
            if os.path.exists(s2_dir):
                for f in os.listdir(s2_dir):
                    if f.endswith('.tif'):
                        s2_files.append(os.path.join(s2_dir, f))
            
            logging.info(f"✅ Downloaded {len(s2_files)} S2 files")
            return s2_files
            
        except Exception as e:
            logging.error(f"❌ Failed to download S2 data: {e}")
            return []
    
    def download_lst_data(self, roi_name: str, roi_geometry_ee, start_date: str, end_date: str) -> Dict[str, List[str]]:
        """Download LST data using existing module"""
        logging.info(f"🌡️  Downloading LST data for {roi_name}")
        
        try:
            # Import the LST retrieval module
            import sys
            sys.path.append('data_retrival_module')
            from lst_retrieval import lst_retrive
            
            lst_output_folder = os.path.join(self.output_folder, roi_name)
            os.makedirs(lst_output_folder, exist_ok=True)
            
            # Download LST data
            lst_retrive(
                start_date=start_date,
                end_date=end_date,
                roi_geometry=roi_geometry_ee,
                roi_name=roi_name,
                output_folder=lst_output_folder
            )
            
            # Find downloaded LST files (organized by satellite and processing level)
            lst_files = {
                "L8_L1": [],
                "L8_L2": [],
                "L9_L1": [],
                "L9_L2": [],
                "aster": []
            }
            
            lst_dir = os.path.join(lst_output_folder, roi_name, 'lst')
            if os.path.exists(lst_dir):
                for f in os.listdir(lst_dir):
                    if f.endswith('.tif'):
                        file_path = os.path.join(lst_dir, f)
                        if 'L8_lst' in f:
                            lst_files["L8_L2"].append(file_path)  # LST is processed L2 product
                        elif 'L9_lst' in f:
                            lst_files["L9_L2"].append(file_path)
            
            # Look for ASTER files
            aster_dir = os.path.join(lst_output_folder, roi_name, 'aster_ged')
            if os.path.exists(aster_dir):
                for f in os.listdir(aster_dir):
                    if f.endswith('.tif'):
                        lst_files["aster"].append(os.path.join(aster_dir, f))
            
            total_files = sum(len(files) for files in lst_files.values())
            logging.info(f"✅ Downloaded {total_files} LST/ASTER files")
            return lst_files
            
        except Exception as e:
            logging.error(f"❌ Failed to download LST data: {e}")
            return {"L8_L1": [], "L8_L2": [], "L9_L1": [], "L9_L2": [], "aster": []}
    
    def create_zip_file(self, files: List[str], dataset_name: str, roi_name: str) -> str:
        """Create ZIP file from list of files"""
        if not files:
            return None
        
        task_id = str(uuid.uuid4())
        zip_filename = f"{dataset_name}_{roi_name}_{task_id}.zip"
        zip_path = os.path.join(self.output_folder, "zips", zip_filename)
        
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in files:
                if os.path.exists(file_path):
                    # Add file to ZIP with just the filename (no path)
                    zip_file.write(file_path, os.path.basename(file_path))
        
        logging.info(f"📦 Created ZIP: {zip_path} ({len(files)} files)")
        return zip_path, task_id
    
    def upload_zip_to_server(self, zip_path: str, task_id: str, dataset_name: str) -> bool:
        """Upload ZIP file to server"""
        try:
            # Determine upload endpoint based on dataset
            upload_endpoint = f"{self.server_url}/v1/upload/{dataset_name}/{task_id}"
            
            with open(zip_path, 'rb') as f:
                files = {'file': (os.path.basename(zip_path), f, 'application/zip')}
                response = requests.post(upload_endpoint, files=files, timeout=300)
                response.raise_for_status()
            
            logging.info(f"☁️  Uploaded {os.path.basename(zip_path)} to server (task_id: {task_id})")
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to upload {zip_path}: {e}")
            return False
    
    def process_dataset(self, dataset_name: str, roi_name: str, roi_geometry_ee, start_date: str, end_date: str):
        """Process a single dataset: download, zip, upload"""
        logging.info(f"🔄 Processing {dataset_name} dataset...")
        
        if dataset_name == "era5":
            files = self.download_era5_data(roi_name, roi_geometry_ee, start_date, end_date)
            if files:
                zip_path, task_id = self.create_zip_file(files, dataset_name, roi_name)
                if self.upload_zip_to_server(zip_path, task_id, dataset_name):
                    if dataset_name not in self.task_id_mapping:
                        self.task_id_mapping[dataset_name] = []
                    self.task_id_mapping[dataset_name].append(task_id)
        
        elif dataset_name == "s2":
            files = self.download_s2_data(roi_name, roi_geometry_ee, start_date, end_date)
            if files:
                zip_path, task_id = self.create_zip_file(files, dataset_name, roi_name)
                if self.upload_zip_to_server(zip_path, task_id, dataset_name):
                    if dataset_name not in self.task_id_mapping:
                        self.task_id_mapping[dataset_name] = []
                    self.task_id_mapping[dataset_name].append(task_id)
        
        elif dataset_name == "lst":
            lst_files_dict = self.download_lst_data(roi_name, roi_geometry_ee, start_date, end_date)
            
            # Create separate task mapping for LST
            if "lst" not in self.task_id_mapping:
                self.task_id_mapping["lst"] = {}
            
            for category, files in lst_files_dict.items():
                if files:
                    zip_path, task_id = self.create_zip_file(files, f"lst_{category}", roi_name)
                    endpoint_name = category.lower().replace('_', '') if category != "aster" else "aster"
                    if self.upload_zip_to_server(zip_path, task_id, endpoint_name):
                        if category not in self.task_id_mapping["lst"]:
                            self.task_id_mapping["lst"][category] = []
                        self.task_id_mapping["lst"][category].append(task_id)
        
        elif dataset_name == "aster":
            # ASTER is handled as part of LST, but we can also handle it separately
            lst_files_dict = self.download_lst_data(roi_name, roi_geometry_ee, start_date, end_date)
            aster_files = lst_files_dict.get("aster", [])
            
            if aster_files:
                zip_path, task_id = self.create_zip_file(aster_files, dataset_name, roi_name)
                if self.upload_zip_to_server(zip_path, task_id, dataset_name):
                    if dataset_name not in self.task_id_mapping:
                        self.task_id_mapping[dataset_name] = []
                    self.task_id_mapping[dataset_name].append(task_id)
    
    def save_task_mapping(self, output_file: str):
        """Save task ID mapping to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.task_id_mapping, f, indent=2)
        
        logging.info(f"💾 Saved task ID mapping to {output_file}")
        print(f"\n📋 Task ID Mapping:")
        print(json.dumps(self.task_id_mapping, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Download data from GEE and upload to server with task IDs")
    parser.add_argument("--roi_name", required=True, help="Grid PhienHieu / ROI name")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--server_url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--grid_file", default="data/Grid_50K_MatchedDates.geojson", help="Grid file path")
    parser.add_argument("--datasets", nargs="+", choices=["era5", "s2", "lst", "aster"], 
                       default=["era5", "s2", "lst"], help="Datasets to process")
    parser.add_argument("--output_folder", default="./gee_server_upload", help="Local output folder")
    parser.add_argument("--task_mapping_file", default="task_ids_generated.json", 
                       help="Output file for task ID mapping")

    args = parser.parse_args()

    # Initialize uploader
    uploader = GEEToServerUploader(args.server_url, args.output_folder)
    
    try:
        # Initialize GEE
        uploader.initialize_gee()
        
        # Find ROI geometry
        feature = uploader.find_grid_feature(args.roi_name, args.grid_file)
        roi_geometry_data = feature['geometry']
        roi_geometry_ee = ee.Geometry.Polygon(roi_geometry_data['coordinates'])
        
        logging.info(f"🎯 Processing ROI: {args.roi_name}")
        logging.info(f"📅 Date range: {args.start_date} to {args.end_date}")
        logging.info(f"🗂️  Datasets: {args.datasets}")
        
        # Process each dataset
        for dataset in args.datasets:
            uploader.process_dataset(
                dataset, args.roi_name, roi_geometry_ee, 
                args.start_date, args.end_date
            )
        
        # Save task mapping
        uploader.save_task_mapping(args.task_mapping_file)
        
        logging.info("🎉 GEE to Server upload completed successfully!")
        
    except Exception as e:
        logging.error(f"💥 Upload failed: {e}")
        raise


if __name__ == "__main__":
    main()
