#!/usr/bin/env python3
"""
Test GEE ZIPs with ETL Module

This script tests that ZIP files created by gee_download_and_zip.py 
work correctly with the ETL_data_retrieval_module.

Usage:
    python test_gee_zips_with_etl.py --task_mapping etl_task_mapping.json --roi_name "D-49-49-A"
"""

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def setup_mock_server(zip_folder: str, task_mapping: dict, server_storage: str):
    """Set up mock server storage structure with ZIP files"""
    logging.info("📁 Setting up mock server storage...")
    
    # Create server storage directories
    datasets = ["era5", "s2", "landsat8_l1", "landsat8_l2", "landsat9_l1", "landsat9_l2", "aster"]
    for dataset in datasets:
        os.makedirs(os.path.join(server_storage, dataset), exist_ok=True)
    
    # Copy ZIP files to correct server locations
    for dataset, task_data in task_mapping.items():
        if dataset == "lst":
            # LST has nested structure
            for category, task_ids in task_data.items():
                if isinstance(task_ids, list):
                    for task_id in task_ids:
                        # Map LST categories to server directories
                        endpoint_mapping = {
                            "L8_L1": "landsat8_l1",
                            "L8_L2": "landsat8_l2",
                            "L9_L1": "landsat9_l1", 
                            "L9_L2": "landsat9_l2",
                            "aster": "aster"
                        }
                        
                        server_dir = endpoint_mapping.get(category, category.lower())
                        
                        # Find corresponding ZIP file
                        zip_pattern = f"{category.lower().replace('_', '')}_{task_id}.zip"
                        source_zip = os.path.join(zip_folder, zip_pattern)
                        
                        # Also try landsat pattern
                        if not os.path.exists(source_zip):
                            zip_pattern = f"landsat{category[1].lower()}_{category[3:].lower()}_{task_id}.zip"
                            source_zip = os.path.join(zip_folder, zip_pattern)
                        
                        if os.path.exists(source_zip):
                            dest_zip = os.path.join(server_storage, server_dir, f"{task_id}.zip")
                            shutil.copy2(source_zip, dest_zip)
                            logging.info(f"  📦 Copied {os.path.basename(source_zip)} → {server_dir}/{task_id}.zip")
                        else:
                            logging.warning(f"  ⚠️  ZIP not found: {zip_pattern}")
        else:
            # Simple datasets (era5, s2, aster)
            if isinstance(task_data, list):
                for task_id in task_data:
                    zip_pattern = f"{dataset}_{task_id}.zip"
                    source_zip = os.path.join(zip_folder, zip_pattern)
                    
                    if os.path.exists(source_zip):
                        dest_zip = os.path.join(server_storage, dataset, f"{task_id}.zip")
                        shutil.copy2(source_zip, dest_zip)
                        logging.info(f"  📦 Copied {os.path.basename(source_zip)} → {dataset}/{task_id}.zip")
                    else:
                        logging.warning(f"  ⚠️  ZIP not found: {zip_pattern}")

def start_simple_server(server_storage: str, port: int = 8001):
    """Start simple Flask server for testing"""
    import subprocess
    import time
    
    try:
        # Start the simple upload server
        server_cmd = [
            "python", "simple_upload_server.py",
            "--port", str(port),
            "--storage_dir", server_storage,
            "--host", "127.0.0.1"
        ]
        
        logging.info(f"🚀 Starting test server on port {port}...")
        process = subprocess.Popen(server_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        time.sleep(3)
        
        # Check if server is running
        import requests
        try:
            response = requests.get(f"http://127.0.0.1:{port}/v1/status", timeout=5)
            if response.status_code == 200:
                logging.info("✅ Test server started successfully")
                return process
            else:
                raise Exception(f"Server returned status {response.status_code}")
        except Exception as e:
            process.terminate()
            raise Exception(f"Failed to start test server: {e}")
            
    except Exception as e:
        logging.error(f"❌ Could not start server: {e}")
        return None

def test_etl_processing(task_mapping_file: str, roi_name: str, server_port: int = 8001):
    """Test ETL module with generated task mapping"""
    import subprocess
    
    logging.info("⚙️  Testing ETL module processing...")
    
    # Determine which datasets to test based on task mapping
    with open(task_mapping_file, 'r') as f:
        task_mapping = json.load(f)
    
    datasets = list(task_mapping.keys())
    
    # Run ETL module
    output_folder = "./test_etl_output"
    
    etl_cmd = [
        "python", "-m", "ETL_data_retrieval_module.main",
        "--roi_name", roi_name,
        "--task_ids", task_mapping_file,
        "--datasets"] + datasets + [
        "--output_folder", output_folder,
        "--api_base_url", f"http://127.0.0.1:{server_port}"
    ]
    
    logging.info(f"🔄 Running ETL command: {' '.join(etl_cmd)}")
    
    try:
        result = subprocess.run(etl_cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logging.info("✅ ETL processing completed successfully!")
            
            # Check output files
            roi_output_dir = os.path.join(output_folder, roi_name)
            if os.path.exists(roi_output_dir):
                logging.info("📊 Generated output files:")
                for root, dirs, files in os.walk(roi_output_dir):
                    for file in files:
                        if file.endswith('.tif'):
                            rel_path = os.path.relpath(os.path.join(root, file), output_folder)
                            logging.info(f"  📄 {rel_path}")
                
                return True
            else:
                logging.error(f"❌ No output files found in {roi_output_dir}")
                return False
        else:
            logging.error(f"❌ ETL processing failed:")
            logging.error(f"STDOUT: {result.stdout}")
            logging.error(f"STDERR: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.error("❌ ETL processing timed out")
        return False
    except Exception as e:
        logging.error(f"❌ ETL processing error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test GEE ZIP files with ETL module")
    parser.add_argument("--task_mapping", required=True, help="Task mapping JSON file")
    parser.add_argument("--roi_name", required=True, help="ROI name for testing")
    parser.add_argument("--zip_folder", help="Folder containing ZIP files (auto-detected if not provided)")
    parser.add_argument("--server_port", type=int, default=8001, help="Test server port")
    parser.add_argument("--keep_server", action="store_true", help="Keep server running after test")
    
    args = parser.parse_args()
    
    # Auto-detect zip folder
    zip_folder = args.zip_folder
    if not zip_folder:
        # Look for common zip folder locations
        possible_folders = [
            "./gee_etl_ready/zips",
            "./gee_zip_output/zips", 
            "./zips"
        ]
        
        for folder in possible_folders:
            if os.path.exists(folder):
                zip_folder = folder
                break
        
        if not zip_folder:
            logging.error("❌ Could not find ZIP folder. Please specify --zip_folder")
            return
    
    logging.info(f"🧪 Testing ZIP files from: {zip_folder}")
    
    # Load task mapping
    try:
        with open(args.task_mapping, 'r') as f:
            task_mapping = json.load(f)
        logging.info(f"📋 Loaded task mapping: {list(task_mapping.keys())}")
    except Exception as e:
        logging.error(f"❌ Failed to load task mapping: {e}")
        return
    
    # Create temporary server storage
    server_storage = "./test_server_storage"
    if os.path.exists(server_storage):
        shutil.rmtree(server_storage)
    os.makedirs(server_storage)
    
    server_process = None
    
    try:
        # Set up mock server storage
        setup_mock_server(zip_folder, task_mapping, server_storage)
        
        # Start test server
        server_process = start_simple_server(server_storage, args.server_port)
        if not server_process:
            return
        
        # Test ETL processing
        success = test_etl_processing(args.task_mapping, args.roi_name, args.server_port)
        
        if success:
            logging.info("🎉 All tests passed! Your ZIP files are compatible with the ETL module.")
        else:
            logging.error("💥 Tests failed. Check the ZIP file format or ETL module compatibility.")
    
    except Exception as e:
        logging.error(f"💥 Test failed: {e}")
    
    finally:
        # Clean up
        if server_process and not args.keep_server:
            logging.info("🛑 Stopping test server...")
            server_process.terminate()
            server_process.wait()
        
        if not args.keep_server and os.path.exists(server_storage):
            shutil.rmtree(server_storage, ignore_errors=True)
        
        if args.keep_server:
            logging.info(f"🖥️  Server still running on port {args.server_port}")
            logging.info(f"📁 Server storage: {server_storage}")

if __name__ == "__main__":
    main()
