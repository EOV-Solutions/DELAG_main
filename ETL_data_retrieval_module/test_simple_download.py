#!/usr/bin/env python3
"""
Simple Download Test Script

Tests the basic download workflow using folder IDs directly without search endpoints.
This simulates having 4 test folders on the server and downloading them.

Usage:
    python test_simple_download.py --folder_ids id1,id2,id3,id4 --api_base_url http://localhost:8000
"""

import argparse
import os
import glob
import shutil
from datetime import datetime
from typing import List

import requests
import tempfile
import zipfile


def test_direct_download(folder_id: str, api_base_url: str, dataset_name: str) -> str:
    """
    Test direct download from server using folder ID as task_id.
    Returns path to extracted directory.
    """
    print(f"📡 Testing download for {dataset_name} (folder: {folder_id})")
    
    # Use config to determine download endpoint
    from .config import config
    endpoint_base = config.get_download_endpoint(dataset_name)
    if endpoint_base:
        download_url = f"{api_base_url}{endpoint_base}/{folder_id}"
    else:
        # fallback generic endpoint
        download_url = f"{api_base_url}/v1/download/{folder_id}"
    
    try:
        # Download ZIP
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Save to temp file
        temp_dir = tempfile.mkdtemp(prefix=f"test_{dataset_name}_")
        zip_path = os.path.join(temp_dir, f"{folder_id}.zip")
        
        total_size = 0
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"  ✓ Downloaded {total_size} bytes")
        
        # Extract ZIP
        extract_dir = os.path.join(temp_dir, "extracted")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Count extracted files
        tif_files = glob.glob(os.path.join(extract_dir, '**', '*.tif'), recursive=True)
        all_files = glob.glob(os.path.join(extract_dir, '**', '*'), recursive=True)
        all_files = [f for f in all_files if os.path.isfile(f)]
        
        print(f"  ✓ Extracted {len(all_files)} files ({len(tif_files)} TIF files)")
        
        # Show file structure
        print(f"  📁 Contents:")
        for root, dirs, files in os.walk(extract_dir):
            level = root.replace(extract_dir, '').count(os.sep)
            indent = '    ' + '  ' * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = '    ' + '  ' * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
        
        # Clean up ZIP file
        os.remove(zip_path)
        
        return extract_dir
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Download failed: {e}")
        return None
    except Exception as e:
        print(f"  ❌ Extraction failed: {e}")
        return None


def create_mock_output_files(extract_dir: str, dataset_name: str, output_base: str, roi_name: str, test_date: str) -> bool:
    """
    Create mock output files in the expected format using downloaded test data.
    """
    if not extract_dir or not os.path.exists(extract_dir):
        return False
    
    # Find TIF files in extracted data
    tif_files = glob.glob(os.path.join(extract_dir, '**', '*.tif'), recursive=True)
    if not tif_files:
        print(f"  ❌ No TIF files found in {dataset_name} data")
        return False
    
    # Create appropriate output directory and file
    if dataset_name == "era5":
        output_dir = os.path.join(output_base, roi_name, "era5")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"ERA5_data_{test_date}.tif")
        
        # Copy first TIF file as ERA5 skin temperature
        shutil.copy2(tif_files[0], output_file)
        print(f"  ✅ Created ERA5 file: {output_file}")
        
    elif dataset_name == "s2":
        output_dir = os.path.join(output_base, roi_name, "s2_images")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"S2_8days_{test_date}.tif")
        
        # Use first TIF as S2 composite (or merge multiple if available)
        if len(tif_files) >= 4:
            # Mock 4-band merge - copy first file and rename
            shutil.copy2(tif_files[0], output_file)
            print(f"  ✅ Created S2 composite: {output_file} (using {len(tif_files)} available files)")
        else:
            shutil.copy2(tif_files[0], output_file)
            print(f"  ✅ Created S2 file: {output_file}")
            
    elif dataset_name in ["landsat8", "landsat9"]:
        output_dir = os.path.join(output_base, roi_name, "lst")
        os.makedirs(output_dir, exist_ok=True)
        satellite = "L8" if "8" in dataset_name else "L9"
        output_file = os.path.join(output_dir, f"{satellite}_lst16days_{test_date}.tif")
        
        # Copy first TIF as LST
        shutil.copy2(tif_files[0], output_file)
        print(f"  ✅ Created {satellite} LST: {output_file}")
    
    else:
        # Generic copy
        output_dir = os.path.join(output_base, roi_name, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset_name}_{test_date}.tif")
        shutil.copy2(tif_files[0], output_file)
        print(f"  ✅ Created {dataset_name} file: {output_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test ETL download workflow with folder IDs")
    parser.add_argument("--folder_ids", required=True,
                       help="Comma-separated list of folder IDs on server (e.g., abc123,def456,ghi789,jkl012)")
    parser.add_argument("--api_base_url", default="http://localhost:8000", help="Server API base URL")
    parser.add_argument("--output_folder", default="test_retrieved_data", help="Output folder for test results")
    parser.add_argument("--roi_name", default="test_roi", help="Test ROI name")
    parser.add_argument("--keep_temp", action="store_true", help="Keep temporary downloaded files for inspection")
    
    args = parser.parse_args()
    
    # Parse folder IDs
    folder_ids = [fid.strip() for fid in args.folder_ids.split(',')]
    if len(folder_ids) < 2:
        raise ValueError("Need at least 2 folder IDs for testing")
    
    print(f"🚀 Testing ETL download workflow")
    print(f"📂 Test folder IDs: {folder_ids}")
    print(f"🌐 API base URL: {args.api_base_url}")
    print(f"📁 Output folder: {args.output_folder}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Test date
    test_date = "2024-01-15"
    
    # Map folder IDs to datasets
    dataset_mapping = [
        ("era5", folder_ids[0]),
        ("s2", folder_ids[1]),
    ]
    
    # Add Landsat datasets if more folder IDs available
    if len(folder_ids) > 2:
        dataset_mapping.append(("landsat8", folder_ids[2]))
    if len(folder_ids) > 3:
        dataset_mapping.append(("landsat9", folder_ids[3]))
    
    temp_dirs = []
    success_count = 0
    
    try:
        for dataset_name, folder_id in dataset_mapping:
            print(f"\n🧪 Testing {dataset_name} with folder ID: {folder_id}")
            
            # Test download
            extract_dir = test_direct_download(folder_id, args.api_base_url, dataset_name)
            
            if extract_dir:
                temp_dirs.append(extract_dir)
                
                # Create mock output file
                if create_mock_output_files(extract_dir, dataset_name, args.output_folder, args.roi_name, test_date):
                    success_count += 1
            
            print("-" * 40)
        
        # Show final results
        print(f"\n🎉 Test Summary:")
        print(f"   Datasets tested: {len(dataset_mapping)}")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {len(dataset_mapping) - success_count}")
        
        # Show output structure
        roi_dir = os.path.join(args.output_folder, args.roi_name)
        if os.path.exists(roi_dir):
            print(f"\n📁 Final output structure:")
            for root, dirs, files in os.walk(roi_dir):
                level = root.replace(roi_dir, '').count(os.sep)
                indent = '  ' * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = '  ' * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"{subindent}{file} ({file_size:,} bytes)")
        
        print(f"\n✅ Test completed! Output saved to: {roi_dir}")
        
        # Test the module import path as well
        print(f"\n🔧 Testing module import compatibility...")
        try:
            from . import ServerClient
            print("  ✓ ServerClient import successful")
        except ImportError as e:
            print(f"  ⚠️ Module import issue: {e}")
        
        return 0 if success_count > 0 else 1
        
    finally:
        # Cleanup temp directories unless requested to keep
        if not args.keep_temp:
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n🧹 Cleaned up {len(temp_dirs)} temporary directories")
        else:
            print(f"\n📁 Temporary directories preserved:")
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    print(f"  {temp_dir}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
