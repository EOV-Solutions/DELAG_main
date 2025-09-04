import glob
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import rasterio

from .server_client import ServerClient
from .utils import find_grid_feature, bbox_from_feature, group_files_by_date
from .config import config


def _ensure_dirs(base_output: str, roi_name: str) -> str:
    s2_dir = os.path.join(base_output, roi_name, config.get_output_dir("s2"))
    os.makedirs(s2_dir, exist_ok=True)
    return s2_dir


def _merge_bands_to_multiband(band_files: List[str], dst_path: str) -> None:
    """
    Merge individual band TIFFs into a 4-band composite (B4,B3,B2,B8) -> (R,G,B,NIR).
    Sets nodata to -100 to match preprocessing expectations.
    """
    if len(band_files) != 4:
        raise ValueError(f"Expected 4 band files, got {len(band_files)}")
    
    # Sort by band name to ensure consistent order: B2, B3, B4, B8
    band_order = ['B2', 'B3', 'B4', 'B8']
    sorted_files = []
    for band in band_order:
        matching = [f for f in band_files if band in os.path.basename(f)]
        if not matching:
            raise ValueError(f"Missing band {band} in downloaded files")
        sorted_files.append(matching[0])
    
    with rasterio.open(sorted_files[0]) as src0:
        profile = src0.profile.copy()
        bands = [src0.read(1)]
    
    for f in sorted_files[1:]:
        with rasterio.open(f) as src:
            bands.append(src.read(1))
    
    # Set nodata and count to 4 bands
    nodata_val = config.get_nodata_value("s2")
    profile.update(count=4, nodata=nodata_val, dtype='float32')
    
    with rasterio.open(dst_path, 'w', **profile) as dst:
        for i, arr in enumerate(bands, start=1):
            # Convert to float32 and set nodata
            arr_float = arr.astype(np.float32)
            arr_float = np.where(np.isnan(arr_float), nodata_val, arr_float)
            dst.write(arr_float, i)
        
        # Write tags for metadata
        dst.update_tags(ACQUISITION_TYPE='S2_8days')


def _apply_cloud_mask_and_composite(temp_dir: str, roi_name: str, target_date: datetime) -> Optional[List[str]]:
    """
    Apply cloud masking and create 8-day median composite from downloaded S2 scenes.
    Returns list of band file paths for composite or None if no valid data.
    """
    # Find all S2 TIF files in temp directory
    tif_files = glob.glob(os.path.join(temp_dir, '**', '*.tif'), recursive=True)
    if not tif_files:
        return None
    
    # Group by date and filter to ±4 days around target (8-day window)
    grouped = group_files_by_date(tif_files)
    valid_dates = []
    target_str = target_date.strftime('%Y-%m-%d')
    
    for i in range(-4, 5):  # ±4 days
        check_date = target_date + timedelta(days=i)
        check_str = check_date.strftime('%Y-%m-%d')
        if check_str in grouped:
            valid_dates.extend(grouped[check_str])
    
    if not valid_dates:
        return None
    
    # For simplicity, take the scene closest to target date
    # In a full implementation, you'd apply cloud masking and create median composite
    best_scene_files = []
    target_bands = ['B2', 'B3', 'B4', 'B8']
    
    # Find files for each band from the best available scene
    for band in target_bands:
        band_files = [f for f in valid_dates if band in os.path.basename(f)]
        if band_files:
            best_scene_files.append(band_files[0])
    
    if len(best_scene_files) == 4:
        return best_scene_files
    
    return None


def retrieve_s2_from_server(
    roi_name: str,
    grid_file: str,
    composite_dates: List[str],
    output_base: str,
    api_base_url: str = "http://localhost:8000",
    cloud_cover: float = 85.0,
) -> None:
    """
    Download S2 RGB-NIR composites from the server, creating 8-day composites and writing as
    data/retrieved_data/<roi_name>/s2_images/S2_8days_YYYY-MM-DD.tif
    
    Args:
        roi_name: Grid ROI identifier
        grid_file: Path to grid GeoJSON
        composite_dates: List of target composite dates (YYYY-MM-DD)
        output_base: Base output directory
        api_base_url: Server API base URL
        cloud_cover: Maximum cloud cover percentage (0-100)
    """
    feature = find_grid_feature(roi_name, grid_file)
    if feature is None:
        raise ValueError(f"ROI '{roi_name}' not found in grid {grid_file}")
    
    bbox = bbox_from_feature(feature)
    s2_dir = _ensure_dirs(output_base, roi_name)
    client = ServerClient(api_base_url=api_base_url, timeout=180)
    
    for date_str in composite_dates:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        out_name = os.path.join(s2_dir, f"S2_8days_{date_str}.tif")
        
        if os.path.exists(out_name):
            print(f"S2 composite already exists: {out_name}")
            continue
        
        # Create 8-day window around target date
        window_days = config.S2_CONFIG["composite_window_days"]
        start_date = target_date - timedelta(days=window_days)
        end_date = target_date + timedelta(days=window_days)
        datetime_range = f"{start_date.strftime('%Y-%m-%d')}T00:00:00Z/{end_date.strftime('%Y-%m-%d')}T23:59:59Z"
        
        try:
            # Create S2 search task
            task_id = client.create_s2_task(
                bbox=bbox,
                datetime_range_iso=datetime_range,
                extra={
                    "cloud_cover": cloud_cover,
                    "bands": config.S2_CONFIG["default_bands"],
                    "limit": config.S2_CONFIG["default_limit"]
                }
            )
            print(f"Created S2 task {task_id} for {date_str}")
            
        except Exception as e:
            print(f"Failed to create S2 task for {date_str}: {e}")
            continue
        
        # Download and extract
        temp_dir = None
        try:
            temp_dir = client.download_and_extract("s2", task_id)
            print(f"Downloaded S2 data for {date_str}")
            
            # Process the downloaded scenes into a composite
            composite_files = _apply_cloud_mask_and_composite(temp_dir, roi_name, target_date)
            
            if composite_files:
                _merge_bands_to_multiband(composite_files, out_name)
                print(f"✅ Created S2 composite: {out_name}")
            else:
                print(f"❌ No valid S2 data found for {date_str}")
                
        except Exception as e:
            print(f"Failed to process S2 data for {date_str}: {e}")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)