"""
ASTER GED Data Retrieval from Server

Retrieves ASTER Global Emissivity Dataset (GED) data for LST calculations and other applications.
"""

import os
import glob
from datetime import datetime
from typing import List, Optional

import rasterio
import numpy as np

from .server_client import ServerClient
from .utils import extract_datetime_from_filename
from .config import config


def _ensure_dirs(base_output: str, roi_name: str) -> str:
    aster_dir = os.path.join(base_output, roi_name, config.get_output_dir("aster"))
    os.makedirs(aster_dir, exist_ok=True)
    return aster_dir


def _write_aster_band(src_path: str, dst_path: str, band_name: str, acquisition_date: datetime) -> None:
    """
    Write a single ASTER band with proper metadata and scaling.
    
    Args:
        src_path: Source GeoTIFF file
        dst_path: Destination file path
        band_name: ASTER band identifier (e.g., 'emissivity_band10')
        acquisition_date: Date for metadata
    """
    with rasterio.open(src_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        
        # Apply appropriate scaling based on band type
        if 'emissivity' in band_name:
            # ASTER emissivity values are typically scaled by 0.001
            data = data * config.ASTER_CONFIG.get("emissivity_scaling", 0.001)
        elif 'ndvi' in band_name:
            # ASTER NDVI values are typically scaled by 0.01
            data = data * config.ASTER_CONFIG.get("ndvi_scaling", 0.01)
        
        # Set nodata value
        nodata_val = config.get_nodata_value("aster")
        profile.update(dtype='float32', nodata=nodata_val)
        
        # Replace any invalid values with nodata
        if np.isnan(nodata_val):
            data = np.where((data < -1) | (data > 2), np.nan, data)  # Reasonable emissivity/NDVI range
        else:
            data = np.where((data < -1) | (data > 2), nodata_val, data)
        
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data, 1)
            
            # Write metadata tags
            dst.update_tags(
                ACQUISITION_TYPE='ASTER_GED',
                BAND_NAME=band_name,
                ACQUISITION_DATE=acquisition_date.strftime('%Y-%m-%d'),
                PROCESSING_LEVEL='L3',
                SCALING_APPLIED='YES' if 'emissivity' in band_name or 'ndvi' in band_name else 'NO'
            )


def retrieve_aster_from_server(
    roi_name: str,
    task_ids: List[str],
    output_base: str,
    api_base_url: str,
) -> None:
    """
    Retrieve ASTER GED data from server using predefined task IDs.
    
    Args:
        roi_name: ROI identifier for output folder structure
        task_ids: List of task IDs to download
        output_base: Base output directory
        api_base_url: Server API base URL
    """
    print(f"🌍 Retrieving ASTER GED data for {roi_name}")
    
    # Set up output directory
    aster_dir = _ensure_dirs(output_base, roi_name)
    
    # Create client
    client = ServerClient(api_base_url=api_base_url)
    
    for task_id in task_ids:
        print(f"Processing ASTER task: {task_id}")
        
        try:
            # Download and extract
            temp_dir = client.download_and_extract("aster", task_id)
            
            # Find all TIF files
            tif_files = glob.glob(os.path.join(temp_dir, '**', '*.tif'), recursive=True)
            print(f"✓ Found {len(tif_files)} TIF files")
            
            if not tif_files:
                print("❌ No ASTER TIF files found in downloaded data")
                continue
            
            # Process each TIF file
            for src_file in tif_files:
                fname = os.path.basename(src_file).lower()
                
                # Determine band type from filename
                band_name = None
                if 'emissivity_band10' in fname:
                    band_name = 'emissivity_band10'
                elif 'emissivity_band11' in fname:
                    band_name = 'emissivity_band11'
                elif 'emissivity_band12' in fname:
                    band_name = 'emissivity_band12'
                elif 'emissivity_band13' in fname:
                    band_name = 'emissivity_band13'
                elif 'emissivity_band14' in fname:
                    band_name = 'emissivity_band14'
                elif 'ndvi' in fname:
                    band_name = 'ndvi'
                
                if not band_name:
                    print(f"  ? Skipping unknown ASTER file: {fname}")
                    continue
                
                # Create output filename
                output_file = os.path.join(aster_dir, f"ASTER_{band_name}.tif")
                
                if os.path.exists(output_file):
                    print(f"  ✓ ASTER {band_name} already exists: {output_file}")
                    continue
                
                try:
                    # Extract date from filename if possible, fallback to current date
                    extracted_datetime = extract_datetime_from_filename(fname)
                    acquisition_date = extracted_datetime if extracted_datetime else datetime.now()
                    
                    _write_aster_band(src_file, output_file, band_name, acquisition_date)
                    print(f"  ✅ Created ASTER {band_name}: {output_file}")
                    
                except Exception as e:
                    print(f"  ❌ Failed to process ASTER {band_name}: {e}")
                    continue
            
            print(f"✓ ASTER task {task_id} completed")
            
        except Exception as e:
            print(f"❌ ASTER task {task_id} failed: {e}")
            continue


if __name__ == "__main__":
    # Example usage for testing
    retrieve_aster_from_server(
        roi_name="D-49-49-A-c-2",
        grid_file="data/Grid_50K_MatchedDates.geojson",
        start_date="2020-01-01",
        end_date="2020-12-31",
        output_base="data/retrieved_data",
        api_base_url="http://localhost:8000"
    )
