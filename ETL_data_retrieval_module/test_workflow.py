#!/usr/bin/env python3
"""
ETL Workflow Test Script

Tests the complete ETL data retrieval workflow using mock data folders on the server.
This allows testing without implementing the actual satellite search endpoints.

Usage:
    python test_workflow.py --test_folder_ids folder1,folder2,folder3,folder4
"""

import argparse
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List

from .server_client import ServerClient
from .utils import find_grid_feature, bbox_from_feature
from .config import config


class MockServerClient(ServerClient):
    """
    Extended ServerClient that bypasses search endpoints and uses predefined folder IDs
    for testing the download and processing workflow.
    """
    
    def __init__(self, test_folder_mapping: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.test_folder_mapping = test_folder_mapping
        
    def create_era5_task(self, bbox, datetime_range_iso, variables=None, utc_hours=None, limit=None):
        """Mock ERA5 task creation - returns predefined folder ID"""
        return self.test_folder_mapping.get('era5', 'test_era5_folder')
    
    def create_s2_task(self, bbox, datetime_range_iso, extra=None):
        """Mock S2 task creation - returns predefined folder ID"""
        return self.test_folder_mapping.get('s2', 'test_s2_folder')
    
    def post_json(self, path: str, payload: Dict) -> Dict:
        """Mock API calls for Landsat and ASTER endpoints"""
        if 'landsat8l1_search' in path:
            return {"task_id": self.test_folder_mapping.get('landsat8l1', 'test_l8l1_folder')}
        elif 'landsat8l2_search' in path:
            return {"task_id": self.test_folder_mapping.get('landsat8l2', 'test_l8l2_folder')}
        elif 'landsat9l1_search' in path:
            return {"task_id": self.test_folder_mapping.get('landsat9l1', 'test_l9l1_folder')}
        elif 'landsat9l2_search' in path:
            return {"task_id": self.test_folder_mapping.get('landsat9l2', 'test_l9l2_folder')}
        elif 'aster_search' in path:
            return {"task_id": self.test_folder_mapping.get('aster', 'test_aster_folder')}
        else:
            # Fallback to original implementation for other endpoints
            return super().post_json(path, payload)


def create_mock_roi_feature(roi_name: str, bbox: List[float]) -> Dict:
    """Create a mock GeoJSON feature for testing"""
    return {
        'type': 'Feature',
        'properties': {'PhienHieu': roi_name},
        'geometry': {
            'type': 'Polygon',
            'coordinates': [[
                [bbox[0], bbox[1]],  # SW
                [bbox[2], bbox[1]],  # SE
                [bbox[2], bbox[3]],  # NE
                [bbox[0], bbox[3]],  # NW
                [bbox[0], bbox[1]]   # Close
            ]]
        }
    }


def test_era5_workflow(client: MockServerClient, roi_name: str, output_base: str, test_dates: List[str]):
    """Test ERA5 retrieval workflow"""
    print("🧪 Testing ERA5 workflow...")
    
    from .era5_from_server import _ensure_dirs, _write_single_band_copy
    import glob
    import rasterio
    import numpy as np
    
    era5_dir = _ensure_dirs(output_base, roi_name)
    
    for date_str in test_dates:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        out_name = os.path.join(era5_dir, f"ERA5_data_{date_str}.tif")
        
        if os.path.exists(out_name):
            print(f"  ✓ ERA5 file already exists: {out_name}")
            continue
        
        try:
            # Mock task creation
            task_id = client.create_era5_task(
                bbox=[105.0, 20.0, 106.0, 21.0],
                datetime_range_iso=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z",
                variables=["skin_temperature"]
            )
            
            # Download test data
            temp_dir = client.download_and_extract("era5", task_id)
            
            # Find TIF files
            tif_files = glob.glob(os.path.join(temp_dir, '**', '*.tif'), recursive=True)
            if tif_files:
                # Use first available TIF as skin temperature data
                chosen_file = tif_files[0]
                _write_single_band_copy(chosen_file, out_name, target_date, "Hourly_10:00")
                print(f"  ✅ Created ERA5 file: {out_name}")
            else:
                print(f"  ❌ No TIF files found in test data for {date_str}")
                
        except Exception as e:
            print(f"  ❌ ERA5 test failed for {date_str}: {e}")


def test_s2_workflow(client: MockServerClient, roi_name: str, output_base: str, test_dates: List[str]):
    """Test S2 retrieval workflow"""
    print("🧪 Testing S2 workflow...")
    
    from .s2_from_server import _ensure_dirs, _merge_bands_to_multiband
    import glob
    
    s2_dir = _ensure_dirs(output_base, roi_name)
    
    for date_str in test_dates:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        out_name = os.path.join(s2_dir, f"S2_8days_{date_str}.tif")
        
        if os.path.exists(out_name):
            print(f"  ✓ S2 file already exists: {out_name}")
            continue
        
        try:
            # Mock task creation
            task_id = client.create_s2_task(
                bbox=[105.0, 20.0, 106.0, 21.0],
                datetime_range_iso=f"{date_str}T00:00:00Z/{date_str}T23:59:59Z",
                extra={"cloud_cover": 85.0, "bands": ["B2", "B3", "B4", "B8"]}
            )
            
            # Download test data
            temp_dir = client.download_and_extract("s2", task_id)
            
            # Find band files (look for B2, B3, B4, B8 patterns)
            tif_files = glob.glob(os.path.join(temp_dir, '**', '*.tif'), recursive=True)
            
            # Mock band detection - use first 4 TIFs as bands
            if len(tif_files) >= 4:
                band_files = tif_files[:4]
                
                # Rename to match expected band patterns for testing
                mock_bands = []
                for i, (original_file, band_name) in enumerate(zip(band_files, ['B2', 'B3', 'B4', 'B8'])):
                    mock_path = os.path.join(temp_dir, f"test_{band_name}.tif")
                    shutil.copy2(original_file, mock_path)
                    mock_bands.append(mock_path)
                
                _merge_bands_to_multiband(mock_bands, out_name)
                print(f"  ✅ Created S2 composite: {out_name}")
            else:
                print(f"  ❌ Not enough TIF files for S2 composite ({len(tif_files)} found, need 4)")
                
        except Exception as e:
            print(f"  ❌ S2 test failed for {date_str}: {e}")


def test_lst_workflow(client: MockServerClient, roi_name: str, output_base: str, test_dates: List[str]):
    """Test LST retrieval workflow (simplified for testing)"""
    print("🧪 Testing LST workflow...")
    
    from .lst_from_server import _ensure_dirs
    import glob
    import rasterio
    import numpy as np
    
    lst_dir = _ensure_dirs(output_base, roi_name)
    
    # Mock ASTER download
    try:
        aster_task_id = client.post_json("/v1/aster_search", {
            "bbox": [105.0, 20.0, 106.0, 21.0],
            "datetime": f"{test_dates[0]}T00:00:00Z/{test_dates[-1]}T23:59:59Z",
            "bands": ["emissivity_band10"]
        })["task_id"]
        
        aster_temp_dir = client.download_and_extract("aster", aster_task_id)
        print(f"  ✓ Downloaded ASTER test data")
    except Exception as e:
        print(f"  ⚠️ ASTER download failed: {e}")
        aster_temp_dir = None
    
    # Process each satellite and date
    for satellite in ["L8", "L9"]:
        for date_str in test_dates:
            output_file = os.path.join(lst_dir, f"{satellite}_lst16days_{date_str}.tif")
            
            if os.path.exists(output_file):
                print(f"  ✓ LST file already exists: {output_file}")
                continue
            
            try:
                # Mock L1 and L2 downloads
                l1_endpoint = f"landsat{satellite[1].lower()}l1"
                l2_endpoint = f"landsat{satellite[1].lower()}l2"
                
                l1_task_id = client.post_json(f"/v1/{l1_endpoint}_search", {
                    "bbox": [105.0, 20.0, 106.0, 21.0],
                    "datetime": f"{date_str}T00:00:00Z/{date_str}T23:59:59Z"
                })["task_id"]
                
                l2_task_id = client.post_json(f"/v1/{l2_endpoint}_search", {
                    "bbox": [105.0, 20.0, 106.0, 21.0],
                    "datetime": f"{date_str}T00:00:00Z/{date_str}T23:59:59Z"
                })["task_id"]
                
                l1_temp_dir = client.download_and_extract(l1_endpoint, l1_task_id)
                l2_temp_dir = client.download_and_extract(l2_endpoint, l2_task_id)
                
                # Find any TIF file to use as mock LST
                all_tifs = (glob.glob(os.path.join(l1_temp_dir, '**', '*.tif'), recursive=True) +
                           glob.glob(os.path.join(l2_temp_dir, '**', '*.tif'), recursive=True))
                
                if all_tifs:
                    # Copy first TIF as mock LST result
                    with rasterio.open(all_tifs[0]) as src:
                        profile = src.profile.copy()
                        data = src.read(1).astype(np.float32)
                        
                        # Mock LST data (convert to reasonable temperature range)
                        data = np.where(data > 0, 290 + (data % 50), np.nan)
                        
                        profile.update(dtype='float32', count=1, nodata=np.nan)
                        
                        with rasterio.open(output_file, 'w', **profile) as dst:
                            dst.write(data, 1)
                            dst.update_tags(
                                ACQUISITION_TYPE=f'{satellite}_LST',
                                PROCESSING_LEVEL='LST_L3'
                            )
                    
                    print(f"  ✅ Created {satellite} LST: {output_file}")
                else:
                    print(f"  ❌ No TIF files found for {satellite} LST on {date_str}")
                    
            except Exception as e:
                print(f"  ❌ {satellite} LST test failed for {date_str}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test ETL workflow with mock server data")
    parser.add_argument("--test_folder_ids", required=True, 
                       help="Comma-separated list of test folder IDs on server (era5,s2,l8l1,l8l2)")
    parser.add_argument("--roi_name", default="test_roi", help="Test ROI name")
    parser.add_argument("--api_base_url", default="http://localhost:8000", help="Server API URL")
    parser.add_argument("--output_folder", default="test_retrieved_data", help="Test output folder")
    parser.add_argument("--datasets", nargs="+", choices=["era5", "s2", "lst"], 
                       default=["era5", "s2"], help="Datasets to test")
    
    args = parser.parse_args()
    
    # Parse test folder IDs
    folder_ids = args.test_folder_ids.split(',')
    if len(folder_ids) < 2:
        raise ValueError("Need at least 2 test folder IDs (era5,s2 minimum)")
    
    # Map datasets to folder IDs
    test_mapping = {
        'era5': folder_ids[0],
        's2': folder_ids[1] if len(folder_ids) > 1 else folder_ids[0],
        'landsat8l1': folder_ids[2] if len(folder_ids) > 2 else folder_ids[0],
        'landsat8l2': folder_ids[3] if len(folder_ids) > 3 else folder_ids[1],
        'landsat9l1': folder_ids[2] if len(folder_ids) > 2 else folder_ids[0],
        'landsat9l2': folder_ids[3] if len(folder_ids) > 3 else folder_ids[1],
        'aster': folder_ids[0]
    }
    
    print(f"🚀 Starting ETL workflow test")
    print(f"📂 Test folder mapping: {test_mapping}")
    print(f"🎯 Target datasets: {args.datasets}")
    print(f"📍 ROI: {args.roi_name}")
    print(f"🌐 API: {args.api_base_url}")
    
    # Create mock client
    client = MockServerClient(test_mapping, api_base_url=args.api_base_url, timeout=120)
    
    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Generate test dates (3 dates, 16 days apart)
    start_date = datetime(2024, 1, 1)
    test_dates = []
    for i in range(3):
        test_date = start_date + timedelta(days=i*16)
        test_dates.append(test_date.strftime('%Y-%m-%d'))
    
    print(f"📅 Test dates: {test_dates}")
    
    # Run tests
    try:
        if "era5" in args.datasets:
            test_era5_workflow(client, args.roi_name, args.output_folder, test_dates)
        
        if "s2" in args.datasets:
            test_s2_workflow(client, args.roi_name, args.output_folder, test_dates)
        
        if "lst" in args.datasets:
            test_lst_workflow(client, args.roi_name, args.output_folder, test_dates)
        
        print(f"\n🎉 Test completed! Check output in: {args.output_folder}")
        
        # Show output structure
        roi_dir = os.path.join(args.output_folder, args.roi_name)
        if os.path.exists(roi_dir):
            print(f"\n📁 Output structure:")
            for root, dirs, files in os.walk(roi_dir):
                level = root.replace(roi_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
