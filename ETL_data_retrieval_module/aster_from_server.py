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
from .utils import find_grid_feature, bbox_from_feature
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
    grid_file: str,
    start_date: str,
    end_date: str,
    output_base: str,
    api_base_url: str,
    bands: Optional[List[str]] = None,
) -> None:
    """
    Retrieve ASTER GED data from server for a given ROI and date range.
    
    Args:
        roi_name: Grid PhienHieu identifier
        grid_file: Path to GeoJSON grid file
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format  
        output_base: Base output directory
        api_base_url: Server API base URL
        bands: List of ASTER bands to retrieve (default from config)
    """
    print(f"🌍 Retrieving ASTER GED data for {roi_name} ({start_date} to {end_date})")
    
    # Find ROI geometry
    feature = find_grid_feature(roi_name, grid_file)
    if not feature:
        raise ValueError(f"ROI '{roi_name}' not found in grid file {grid_file}")
    
    bbox = bbox_from_feature(feature)
    
    # Set up output directory
    aster_dir = _ensure_dirs(output_base, roi_name)
    
    # Use default bands if not specified
    if bands is None:
        bands = config.ASTER_CONFIG["default_bands"]
    
    print(f"📡 Requesting ASTER bands: {bands}")
    print(f"📍 Bounding box: {bbox}")
    
    # Create client
    client = ServerClient(api_base_url=api_base_url)
    
    # ASTER GED is static data, so we use the full date range for availability
    datetime_range = f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    
    try:
        # Create ASTER search task
        task_id = client.create_aster_task(
            bbox=bbox,
            datetime_range_iso=datetime_range,
            bands=bands
        )
        print(f"✓ Created ASTER task: {task_id}")
        
        # Download and extract
        temp_dir = client.download_and_extract("aster", task_id)
        print(f"✓ Downloaded ASTER data to: {temp_dir}")
        
        # Find all TIF files
        tif_files = glob.glob(os.path.join(temp_dir, '**', '*.tif'), recursive=True)
        print(f"✓ Found {len(tif_files)} TIF files")
        
        if not tif_files:
            print("❌ No ASTER TIF files found in downloaded data")
            return
        
        # Organize files by band
        aster_files = {}
        for f in tif_files:
            fname = os.path.basename(f).lower()
            
            # Match band patterns
            for band in bands:
                band_lower = band.lower()
                if band_lower in fname:
                    aster_files[band] = f
                    break
        
        print(f"✓ Matched {len(aster_files)} bands: {list(aster_files.keys())}")
        
        # Process each band
        acquisition_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        for band_name, src_file in aster_files.items():
            # Create output filename
            safe_band_name = band_name.replace('_', '').replace(' ', '')
            output_file = os.path.join(aster_dir, f"ASTER_{safe_band_name}_{start_date}.tif")
            
            if os.path.exists(output_file):
                print(f"  ✓ ASTER {band_name} already exists: {output_file}")
                continue
            
            try:
                _write_aster_band(src_file, output_file, band_name, acquisition_date)
                print(f"  ✅ Created ASTER {band_name}: {output_file}")
                
            except Exception as e:
                print(f"  ❌ Failed to process ASTER {band_name}: {e}")
                continue
        
        print(f"🎉 ASTER GED retrieval completed for {roi_name}")
        
    except Exception as e:
        print(f"❌ ASTER retrieval failed: {e}")
        raise


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
