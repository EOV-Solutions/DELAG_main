#!/usr/bin/env python3
"""
Simple test server to serve ZIP files for ETL module testing
"""
import os
import shutil
from flask import Flask, send_file, abort

app = Flask(__name__)

# Create storage directories
STORAGE_DIR = "./test_server_storage"
for dataset in ["era5", "s2", "landsat8_l1", "landsat8_l2", "landsat9_l1", "landsat9_l2", "aster"]:
    os.makedirs(os.path.join(STORAGE_DIR, dataset), exist_ok=True)

@app.route('/v1/<dataset>_download/<task_id>')
def download_data(dataset, task_id):
    """Serve ZIP files for download"""
    file_path = os.path.join(STORAGE_DIR, dataset, f"{dataset}_{task_id}.zip")
    
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        print(f"File not found: {file_path}")
        abort(404)

if __name__ == '__main__':
    # Copy ZIP files to storage directory
    print("Setting up test server storage...")
    
    zip_source = "./test_fixed_gee/zips"
    if os.path.exists(zip_source):
        for zip_file in os.listdir(zip_source):
            if zip_file.endswith('.zip'):
                src_path = os.path.join(zip_source, zip_file)
                
                # Determine dataset from filename
                if zip_file.startswith('landsat8_l1_'):
                    dst_path = os.path.join(STORAGE_DIR, "landsat8_l1", zip_file)
                elif zip_file.startswith('landsat8_l2_'):
                    dst_path = os.path.join(STORAGE_DIR, "landsat8_l2", zip_file)
                elif zip_file.startswith('landsat9_l1_'):
                    dst_path = os.path.join(STORAGE_DIR, "landsat9_l1", zip_file)
                elif zip_file.startswith('landsat9_l2_'):
                    dst_path = os.path.join(STORAGE_DIR, "landsat9_l2", zip_file)
                elif zip_file.startswith('aster_'):
                    dst_path = os.path.join(STORAGE_DIR, "aster", zip_file)
                else:
                    continue
                
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {zip_file} -> {os.path.basename(dst_path)}")
    
    print("\nStarting server on http://localhost:8000")
    print("Available endpoints:")
    print("- /v1/landsat8_l1_download/<task_id>")
    print("- /v1/landsat8_l2_download/<task_id>")
    print("- /v1/landsat9_l1_download/<task_id>")
    print("- /v1/landsat9_l2_download/<task_id>")
    print("- /v1/aster_download/<task_id>")
    
    app.run(host='0.0.0.0', port=8000, debug=True)
