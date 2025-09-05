#!/usr/bin/env python3
"""
Test Mock Data with ETL Module

This script tests the ETL_data_retrieval_module using the mock data generated 
by generate_mock_server_data.py. It creates a simple HTTP server to serve 
the mock data and then runs the ETL module against it.

Usage:
    python test_mock_data_with_etl.py --mock_data_dir "mock_server_data" \
                                      --roi_name "49-49-A-c-2-1" \
                                      --test_datasets era5 s2 lst
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

logging.basicConfig(level=logging.INFO)

class MockAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the mock API server."""
    
    def __init__(self, mock_data_dir, *args, **kwargs):
        self.mock_data_dir = Path(mock_data_dir)
        self.summary_manifest = self._load_summary_manifest()
        super().__init__(*args, **kwargs)
    
    def _load_summary_manifest(self):
        """Load the summary manifest from mock data directory."""
        manifest_path = self.mock_data_dir / "summary_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return {"datasets": {}}
    
    def do_POST(self):
        """Handle POST requests (search endpoints)."""
        try:
            parsed_url = urlparse(self.path)
            
            # Determine dataset type from path
            dataset_type = None
            if '/era5_search' in self.path:
                dataset_type = 'era5'
            elif '/s2_search' in self.path:
                dataset_type = 's2'
            elif '/landsat8_l1_search' in self.path:
                dataset_type = 'landsat8_l1'
            elif '/landsat8_l2_search' in self.path:
                dataset_type = 'landsat8_l2'
            elif '/landsat9_l1_search' in self.path:
                dataset_type = 'landsat9_l1'
            elif '/landsat9_l2_search' in self.path:
                dataset_type = 'landsat9_l2'
            elif '/aster_search' in self.path:
                dataset_type = 'aster'
            
            if dataset_type and dataset_type in self.summary_manifest["datasets"]:
                # Return the folder ID as task_id
                folder_id = self.summary_manifest["datasets"][dataset_type]["folder_id"]
                response = {"task_id": folder_id}
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
                logging.info(f"Search request for {dataset_type}: returned task_id {folder_id}")
            else:
                self.send_response(404)
                self.end_headers()
                logging.warning(f"Unknown search endpoint: {self.path}")
                
        except Exception as e:
            logging.error(f"Error handling POST request: {e}")
            self.send_response(500)
            self.end_headers()
    
    def do_GET(self):
        """Handle GET requests (download endpoints)."""
        try:
            if '_download/' in self.path:
                # Extract task_id from path like /v1/era5_download/uuid-here
                path_parts = self.path.split('/')
                task_id = path_parts[-1]
                
                logging.info(f"Download request received: {self.path}")
                logging.info(f"Extracted task_id: {task_id}")
                
                # Find the dataset folder with this task_id
                dataset_folder = None
                for dataset_type, info in self.summary_manifest["datasets"].items():
                    if info["folder_id"] == task_id:
                        dataset_folder = self.mock_data_dir / info["folder_path"]
                        break
                
                if dataset_folder and dataset_folder.exists():
                    # Create a ZIP file of the dataset folder
                    zip_path = self._create_dataset_zip(dataset_folder, task_id)
                    
                    # Send the ZIP file
                    self.send_response(200)
                    self.send_header('Content-type', 'application/zip')
                    self.send_header('Content-Disposition', f'attachment; filename="{task_id}.zip"')
                    self.end_headers()
                    
                    with open(zip_path, 'rb') as f:
                        shutil.copyfileobj(f, self.wfile)
                    
                    # Clean up temporary ZIP
                    os.unlink(zip_path)
                    
                    logging.info(f"Download request for task_id {task_id}: sent ZIP file")
                else:
                    self.send_response(404)
                    self.end_headers()
                    logging.warning(f"Task ID not found: {task_id}")
            else:
                self.send_response(404)
                self.end_headers()
                
        except Exception as e:
            logging.error(f"Error handling GET request: {e}")
            self.send_response(500)
            self.end_headers()
    
    def _create_dataset_zip(self, dataset_folder, task_id):
        """Create a ZIP file of the dataset folder."""
        import zipfile
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in dataset_folder.rglob("*"):
                    if file_path.is_file() and file_path.suffix == '.tif':
                        # Add file to ZIP with relative path
                        arcname = file_path.relative_to(dataset_folder)
                        zipf.write(file_path, arcname)
            
            return temp_zip.name
    
    def log_message(self, format, *args):
        """Override to control logging."""
        pass  # Suppress default HTTP server logs


def create_mock_server_handler(mock_data_dir):
    """Create a mock server handler with the mock data directory."""
    def handler(*args, **kwargs):
        return MockAPIHandler(mock_data_dir, *args, **kwargs)
    return handler


def start_mock_server(mock_data_dir, port=8000):
    """Start the mock API server."""
    handler_class = create_mock_server_handler(mock_data_dir)
    server = HTTPServer(('localhost', port), handler_class)
    
    def run_server():
        logging.info(f"Mock API server started on http://localhost:{port}")
        server.serve_forever()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(1)
    
    return server


def test_etl_with_mock_data(mock_data_dir, roi_name, datasets, server_port=8000):
    """Test the ETL module with mock data."""
    logging.info("🧪 Testing ETL module with mock data...")
    
    # Start mock server
    server = start_mock_server(mock_data_dir, server_port)
    
    try:
        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_output:
            # Run ETL module
            cmd = [
                "python", "-m", "ETL_data_retrieval_module.main",
                "--roi_name", roi_name,
                "--start_date", "2023-01-01",
                "--end_date", "2023-01-31",
                "--output_folder", temp_output,
                "--api_base_url", f"http://localhost:{server_port}",
                "--datasets"
            ] + datasets
            
            logging.info(f"Running ETL command: {' '.join(cmd)}")
            
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode == 0:
                logging.info("✅ ETL module ran successfully!")
                logging.info("Output:")
                print(result.stdout)
                
                # Check generated files
                output_dir = Path(temp_output) / roi_name
                if output_dir.exists():
                    logging.info("Generated files:")
                    for file_path in output_dir.rglob("*.tif"):
                        relative_path = file_path.relative_to(output_dir)
                        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                        logging.info(f"  📄 {relative_path} ({file_size:.2f} MB)")
                else:
                    logging.warning(f"Output directory not found: {output_dir}")
            else:
                logging.error("❌ ETL module failed!")
                logging.error("Error output:")
                print(result.stderr)
                
    finally:
        # Stop server
        server.shutdown()
        logging.info("Mock server stopped")


def main():
    parser = argparse.ArgumentParser(description="Test ETL module with mock data")
    parser.add_argument("--mock_data_dir", required=True, help="Directory containing mock data")
    parser.add_argument("--roi_name", required=True, help="ROI name to test")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["era5", "s2", "lst", "aster"], 
                       default=["era5"], 
                       help="Datasets to test")
    parser.add_argument("--server_port", type=int, default=8000, help="Mock server port")
    
    args = parser.parse_args()
    
    # Verify mock data directory exists
    mock_data_path = Path(args.mock_data_dir)
    if not mock_data_path.exists():
        logging.error(f"Mock data directory not found: {mock_data_path}")
        return
    
    summary_manifest = mock_data_path / "summary_manifest.json"
    if not summary_manifest.exists():
        logging.error(f"Summary manifest not found: {summary_manifest}")
        return
    
    # Test ETL with mock data
    test_etl_with_mock_data(
        mock_data_dir=args.mock_data_dir,
        roi_name=args.roi_name,
        datasets=args.datasets,
        server_port=args.server_port
    )


if __name__ == "__main__":
    main()
