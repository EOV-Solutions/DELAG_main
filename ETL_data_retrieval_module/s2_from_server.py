import glob
import os
import shutil
import tempfile
from datetime import datetime
from typing import List, Optional

import numpy as np
import rasterio

from .server_client import ServerClient
from .utils import extract_datetime_from_filename, group_files_by_date
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


def _find_best_scene_bands(tif_files: List[str], target_date_str: str = None) -> Optional[List[str]]:
    """
    Find the best set of 4 bands (B2, B3, B4, B8) from downloaded S2 scenes.
    Returns list of band file paths or None if not all bands found.
    """
    if not tif_files:
        return None
    
    target_bands = ['B2', 'B3', 'B4', 'B8']
    
    # Group files by date to find complete scenes
    grouped = group_files_by_date(tif_files)
    
    # If we have a target date preference, try that first
    if target_date_str and target_date_str in grouped:
        date_files = grouped[target_date_str]
        scene_bands = []
        for band in target_bands:
            band_files = [f for f in date_files if f'_{band}_' in os.path.basename(f) or f'_{band}.' in os.path.basename(f)]
            if band_files:
                scene_bands.append(band_files[0])
        
        if len(scene_bands) == 4:
            return scene_bands
    
    # Otherwise, find any complete scene
    for date_str, date_files in grouped.items():
        scene_bands = []
        for band in target_bands:
            band_files = [f for f in date_files if f'_{band}_' in os.path.basename(f) or f'_{band}.' in os.path.basename(f)]
            if band_files:
                scene_bands.append(band_files[0])
        
        if len(scene_bands) == 4:
            return scene_bands
    
    return None


def retrieve_s2_from_server(
    roi_name: str,
    task_ids: List[str],
    output_base: str,
    api_base_url: str = "http://localhost:8000",
) -> None:
    """
    Download S2 RGB-NIR composites from server using predefined task IDs, writing as
    data/retrieved_data/<roi_name>/s2_images/S2_8days_YYYY-MM-DD.tif
    
    Args:
        roi_name: ROI identifier for output folder structure
        task_ids: List of task IDs to download
        output_base: Base output directory
        api_base_url: Server API base URL
    """
    s2_dir = _ensure_dirs(output_base, roi_name)
    client = ServerClient(api_base_url=api_base_url, timeout=180)
    
    for task_id in task_ids:
        print(f"Processing S2 task: {task_id}")
        
        temp_dir = None
        try:
            # Download and extract task data
            temp_dir = client.download_and_extract("s2", task_id)
            
            # Find all TIF files in extracted directory
            tif_files = glob.glob(os.path.join(temp_dir, '**', '*.tif'), recursive=True)
            if not tif_files:
                print(f"No TIFFs found for S2 task {task_id}")
                continue
            
            # Group files by date to process each date separately
            grouped = group_files_by_date(tif_files)
            
            for date_str, date_files in grouped.items():
                out_name = os.path.join(s2_dir, f"S2_8days_{date_str}.tif")
                if os.path.exists(out_name):
                    print(f"S2 composite already exists: {out_name}")
                    continue
                
                # Find complete set of bands for this date
                composite_files = _find_best_scene_bands(date_files, date_str)
                
                if composite_files:
                    _merge_bands_to_multiband(composite_files, out_name)
                    print(f"✅ Created S2 composite: {out_name}")
                else:
                    print(f"❌ No complete band set found for S2 date {date_str}")
                    
        except Exception as e:
            print(f"Failed to process S2 task {task_id}: {e}")
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)