import glob
import os
from datetime import datetime
from typing import List

import rasterio
import numpy as np

from .server_client import ServerClient
from .utils import extract_datetime_from_filename, group_files_by_date
from .config import config


def _ensure_dirs(base_output: str, roi_name: str) -> str:
    era5_dir = os.path.join(base_output, roi_name, config.get_output_dir("era5"))
    os.makedirs(era5_dir, exist_ok=True)
    return era5_dir


def _write_single_band_copy(src_path: str, dst_path: str, extracted_datetime: datetime = None) -> None:
    """
    Copy the first band from src_path to dst_path preserving georeferencing,
    to match expected single-band ERA5 skin temperature files.
    """
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1, nodata=config.get_nodata_value("era5"))
        data = src.read(1)
    
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(data, 1)
        # Write metadata tags
        tags = {}
        if extracted_datetime:
            tags['DATETIME'] = extracted_datetime.strftime('%Y:%m:%d %H:%M:%S')
            tags['ACQUISITION_TYPE'] = f"Hourly_{extracted_datetime.hour:02d}:00"
        if tags:
            dst.update_tags(**tags)


def retrieve_era5_from_server(
    roi_name: str,
    task_ids: List[str],
    output_base: str,
    api_base_url: str = "http://localhost:8000",
) -> None:
    """
    Download ERA5 data from server using predefined task IDs, saving to
    data/retrieved_data/<roi_name>/era5/ERA5_data_YYYY-MM-DD.tif
    
    Args:
        roi_name: ROI identifier for output folder structure
        task_ids: List of task IDs to download
        output_base: Base output directory
        api_base_url: Server API base URL
    """
    era5_dir = _ensure_dirs(output_base, roi_name)
    client = ServerClient(api_base_url=api_base_url, timeout=120)

    for task_id in task_ids:
        print(f"Processing ERA5 task: {task_id}")
        
        try:
            # Download and extract task data
            extracted_dir = client.download_and_extract("era5", task_id)
            
            # Find all TIF files in extracted directory
            tif_paths = glob.glob(os.path.join(extracted_dir, '**', '*.tif'), recursive=True)
            if not tif_paths:
                print(f"No TIFFs found for ERA5 task {task_id}")
                continue

            # Group files by date extracted from filename
            grouped = group_files_by_date(tif_paths)
            
            # Process each date found in the extracted data
            for date_str, date_files in grouped.items():
                out_name = os.path.join(era5_dir, f"ERA5_data_{date_str}.tif")
                if os.path.exists(out_name):
                    print(f"ERA5 file already exists: {out_name}")
                    continue
                
                # Prefer skin_temperature files if present
                skin_files = [f for f in date_files if 'skin' in os.path.basename(f).lower()]
                chosen_file = skin_files[0] if skin_files else date_files[0]
                
                # Extract datetime from chosen filename for metadata
                extracted_datetime = extract_datetime_from_filename(os.path.basename(chosen_file))
                
                # Copy to final output location
                _write_single_band_copy(chosen_file, out_name, extracted_datetime)
                print(f"✅ Created ERA5 file: {out_name}")
                
        except Exception as e:
            print(f"Failed to process ERA5 task {task_id}: {e}")
            continue


