#!/usr/bin/env python3
"""
Simple Upload Server for ZIP Files

A basic Flask server that receives ZIP file uploads and stores them 
for later retrieval by the ETL module.

Usage:
    python simple_upload_server.py --port 8000 --storage_dir ./server_storage
"""

import argparse
import os
import shutil
from pathlib import Path

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Global storage directory
STORAGE_DIR = "./server_storage"

def init_storage_dir(storage_dir: str):
    """Initialize storage directory structure"""
    global STORAGE_DIR
    STORAGE_DIR = storage_dir
    
    # Create dataset directories
    datasets = ["era5", "s2", "landsat8_l1", "landsat8_l2", "landsat9_l1", "landsat9_l2", "aster"]
    for dataset in datasets:
        os.makedirs(os.path.join(STORAGE_DIR, dataset), exist_ok=True)
    
    print(f"📁 Storage initialized at: {STORAGE_DIR}")

@app.route('/v1/upload/<dataset>/<task_id>', methods=['POST'])
def upload_file(dataset, task_id):
    """Upload endpoint for ZIP files"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not file.filename.endswith('.zip'):
            return jsonify({"error": "Only ZIP files allowed"}), 400
        
        # Save file with task_id as filename
        filename = f"{task_id}.zip"
        dataset_dir = os.path.join(STORAGE_DIR, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_path = os.path.join(dataset_dir, filename)
        file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        print(f"📤 Uploaded: {dataset}/{task_id} ({file_size//1024} KB)")
        
        return jsonify({
            "message": "File uploaded successfully",
            "task_id": task_id,
            "dataset": dataset,
            "size_bytes": file_size
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/<dataset>_download/<task_id>', methods=['GET'])
def download_file(dataset, task_id):
    """Download endpoint for ZIP files (matches ETL module expectations)"""
    try:
        # Handle endpoint mappings
        dataset_mapping = {
            "era5": "era5",
            "s2": "s2", 
            "landsat8_l1": "landsat8_l1",
            "landsat8_l2": "landsat8_l2",
            "landsat9_l1": "landsat9_l1", 
            "landsat9_l2": "landsat9_l2",
            "aster": "aster"
        }
        
        storage_dataset = dataset_mapping.get(dataset, dataset)
        file_path = os.path.join(STORAGE_DIR, storage_dataset, f"{task_id}.zip")
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found for task_id: {task_id}"}), 404
        
        print(f"📥 Downloaded: {dataset}/{task_id}")
        return send_file(file_path, as_attachment=True, download_name=f"{task_id}.zip")
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/status', methods=['GET'])
def status():
    """Server status endpoint"""
    try:
        # Count files in each dataset
        datasets = {}
        for dataset_dir in os.listdir(STORAGE_DIR):
            dataset_path = os.path.join(STORAGE_DIR, dataset_dir)
            if os.path.isdir(dataset_path):
                file_count = len([f for f in os.listdir(dataset_path) if f.endswith('.zip')])
                datasets[dataset_dir] = file_count
        
        return jsonify({
            "status": "running",
            "storage_dir": STORAGE_DIR,
            "datasets": datasets
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/v1/list/<dataset>', methods=['GET'])
def list_files(dataset):
    """List available files for a dataset"""
    try:
        dataset_path = os.path.join(STORAGE_DIR, dataset)
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset {dataset} not found"}), 404
        
        files = []
        for filename in os.listdir(dataset_path):
            if filename.endswith('.zip'):
                file_path = os.path.join(dataset_path, filename)
                task_id = filename.replace('.zip', '')
                file_size = os.path.getsize(file_path)
                files.append({
                    "task_id": task_id,
                    "filename": filename,
                    "size_bytes": file_size
                })
        
        return jsonify({
            "dataset": dataset,
            "files": files,
            "count": len(files)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    """Simple index page"""
    return """
    <h1>Simple Upload Server</h1>
    <p>Endpoints:</p>
    <ul>
        <li><b>POST /v1/upload/&lt;dataset&gt;/&lt;task_id&gt;</b> - Upload ZIP file</li>
        <li><b>GET /v1/&lt;dataset&gt;_download/&lt;task_id&gt;</b> - Download ZIP file</li>
        <li><b>GET /v1/status</b> - Server status</li>
        <li><b>GET /v1/list/&lt;dataset&gt;</b> - List files in dataset</li>
    </ul>
    <p>Supported datasets: era5, s2, landsat8_l1, landsat8_l2, landsat9_l1, landsat9_l2, aster</p>
    """

def main():
    parser = argparse.ArgumentParser(description="Simple upload server for ZIP files")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--storage_dir", default="./server_storage", help="Storage directory for uploaded files")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    # Initialize storage
    init_storage_dir(args.storage_dir)
    
    print(f"🚀 Starting upload server on {args.host}:{args.port}")
    print(f"📁 Storage directory: {args.storage_dir}")
    print("🌐 Server endpoints:")
    print(f"   - Upload: POST http://{args.host}:{args.port}/v1/upload/<dataset>/<task_id>")
    print(f"   - Download: GET http://{args.host}:{args.port}/v1/<dataset>_download/<task_id>")
    print(f"   - Status: GET http://{args.host}:{args.port}/v1/status")
    
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
