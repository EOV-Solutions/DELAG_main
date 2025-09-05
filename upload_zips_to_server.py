#!/usr/bin/env python3
"""
Upload ZIP Files to Server

This script uploads ZIP files created by simple_gee_to_zip.py to the server.

Usage:
    python upload_zips_to_server.py --zip_folder ./gee_zip_output/zips \
                                   --task_mapping ./task_ids_from_gee.json \
                                   --server_url http://localhost:8000
"""

import argparse
import json
import logging
import os
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def upload_zip_file(zip_path: str, task_id: str, dataset: str, server_url: str) -> bool:
    """Upload a single ZIP file to the server"""
    try:
        upload_url = f"{server_url}/v1/upload/{dataset}/{task_id}"
        
        with open(zip_path, 'rb') as f:
            files = {'file': (os.path.basename(zip_path), f, 'application/zip')}
            response = requests.post(upload_url, files=files, timeout=300)
            response.raise_for_status()
        
        logging.info(f"✅ Uploaded {os.path.basename(zip_path)} to {dataset}/{task_id}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to upload {zip_path}: {e}")
        return False

def find_zip_file(zip_folder: str, task_id: str, dataset: str, roi_name: str) -> str:
    """Find ZIP file matching the task ID and dataset"""
    # Look for files matching pattern: {dataset}_{roi_name}_{task_id}.zip
    pattern = f"{dataset}_{roi_name}_{task_id}.zip"
    zip_path = os.path.join(zip_folder, pattern)
    
    if os.path.exists(zip_path):
        return zip_path
    
    # Also try LST subcategory patterns
    if dataset.startswith("lst_"):
        lst_pattern = f"{dataset}_{roi_name}_{task_id}.zip"
        lst_zip_path = os.path.join(zip_folder, lst_pattern)
        if os.path.exists(lst_zip_path):
            return lst_zip_path
    
    # Fallback: search for any file containing the task_id
    for filename in os.listdir(zip_folder):
        if task_id in filename and filename.endswith('.zip'):
            return os.path.join(zip_folder, filename)
    
    return None

def upload_from_mapping(zip_folder: str, task_mapping: dict, server_url: str, roi_name: str):
    """Upload all ZIP files based on task mapping"""
    total_uploads = 0
    successful_uploads = 0
    
    for dataset, task_data in task_mapping.items():
        if dataset == "lst":
            # LST has nested structure
            for category, task_ids in task_data.items():
                if isinstance(task_ids, list):
                    for task_id in task_ids:
                        # Map LST categories to server endpoints
                        endpoint_mapping = {
                            "L8_L1": "landsat8_l1",
                            "L8_L2": "landsat8_l2", 
                            "L9_L1": "landsat9_l1",
                            "L9_L2": "landsat9_l2",
                            "aster": "aster"
                        }
                        
                        server_dataset = endpoint_mapping.get(category, category.lower())
                        zip_file = find_zip_file(zip_folder, task_id, f"lst_{category}", roi_name)
                        
                        if zip_file:
                            total_uploads += 1
                            if upload_zip_file(zip_file, task_id, server_dataset, server_url):
                                successful_uploads += 1
                        else:
                            logging.warning(f"⚠️  ZIP file not found for LST {category} task {task_id}")
        else:
            # Simple datasets (era5, s2, aster)
            if isinstance(task_data, list):
                for task_id in task_data:
                    zip_file = find_zip_file(zip_folder, task_id, dataset, roi_name)
                    
                    if zip_file:
                        total_uploads += 1
                        if upload_zip_file(zip_file, task_id, dataset, server_url):
                            successful_uploads += 1
                    else:
                        logging.warning(f"⚠️  ZIP file not found for {dataset} task {task_id}")
    
    logging.info(f"📊 Upload summary: {successful_uploads}/{total_uploads} files uploaded successfully")
    return successful_uploads, total_uploads

def main():
    parser = argparse.ArgumentParser(description="Upload ZIP files to server using task mapping")
    parser.add_argument("--zip_folder", required=True, help="Folder containing ZIP files")
    parser.add_argument("--task_mapping", required=True, help="JSON file with task ID mapping")
    parser.add_argument("--server_url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("--roi_name", help="ROI name (auto-detected from filenames if not provided)")
    
    args = parser.parse_args()
    
    try:
        # Load task mapping
        with open(args.task_mapping, 'r') as f:
            task_mapping = json.load(f)
        
        logging.info(f"📋 Loaded task mapping from {args.task_mapping}")
        
        # Auto-detect ROI name if not provided
        roi_name = args.roi_name
        if not roi_name:
            # Try to extract ROI name from ZIP filenames
            for filename in os.listdir(args.zip_folder):
                if filename.endswith('.zip'):
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        # Assume format: dataset_roi_taskid.zip
                        roi_name = parts[1]
                        break
            
            if roi_name:
                logging.info(f"🎯 Auto-detected ROI name: {roi_name}")
            else:
                logging.error("❌ Could not detect ROI name. Please provide --roi_name")
                return
        
        # Check server status
        try:
            status_response = requests.get(f"{args.server_url}/v1/status", timeout=10)
            status_response.raise_for_status()
            logging.info(f"✅ Server is running at {args.server_url}")
        except Exception as e:
            logging.warning(f"⚠️  Could not check server status: {e}")
        
        # Upload files
        logging.info(f"🚀 Starting upload to {args.server_url}")
        successful, total = upload_from_mapping(
            args.zip_folder, task_mapping, args.server_url, roi_name
        )
        
        if successful == total:
            logging.info("🎉 All files uploaded successfully!")
        else:
            logging.warning(f"⚠️  {total - successful} files failed to upload")
        
    except Exception as e:
        logging.error(f"💥 Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()
