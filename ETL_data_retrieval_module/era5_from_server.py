import glob
import os
from datetime import datetime, timedelta
from typing import List, Optional

import rasterio
import numpy as np

from .server_client import ServerClient
from .utils import (
    find_grid_feature,
    bbox_from_feature,
    group_files_by_date,
    choose_file_closest_to_hour,
)
from .config import config


def _ensure_dirs(base_output: str, roi_name: str) -> str:
    era5_dir = os.path.join(base_output, roi_name, config.get_output_dir("era5"))
    os.makedirs(era5_dir, exist_ok=True)
    return era5_dir


def _infer_reference_and_hour(lst_dir: str, target_date: datetime) -> int:
    """
    Infer the target hour to select ERA5 product by peeking at LST acquisition time.
    If unavailable, default to 10 UTC.
    """
    try:
        candidates = [
            os.path.join(lst_dir, f)
            for f in os.listdir(lst_dir)
            if f.endswith('.tif') and f.split('_')[-1].startswith(target_date.strftime('%Y-%m-%d'))
        ]
        if not candidates:
            return 10
        # Read DATETIME from first candidate
        with rasterio.open(candidates[0]) as src:
            dt_str = src.tags().get('DATETIME')
        if dt_str and ':' in dt_str:
            # expected like YYYY:MM:DD HH:MM:SS
            try:
                date_part, time_part = dt_str.split(' ')
                time_parts = time_part.split(':')
                hour = int(time_parts[0])
                return hour
            except Exception:
                return 10
    except Exception:
        return 10
    return 10


def _write_single_band_copy(src_path: str, dst_path: str, datetime_for_tags: Optional[datetime] = None, acquisition_type: Optional[str] = None) -> None:
    """
    Copy the first band from src_path to dst_path preserving georeferencing,
    to match expected single-band ERA5 skin temperature files.
    """
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        profile.update(count=1)
        data = src.read(1)
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(data, 1)
        # Write legacy-like tags for downstream consumers
        tags = {}
        if datetime_for_tags is not None:
            tags['DATETIME'] = datetime_for_tags.strftime('%Y:%m:%d %H:%M:%S')
        if acquisition_type is not None:
            tags['ACQUISITION_TYPE'] = acquisition_type
        if tags:
            dst.update_tags(**tags)


def retrieve_era5_from_server(
    roi_name: str,
    grid_file: str,
    start_date: str,
    end_date: str,
    output_base: str,
    api_base_url: str = "http://localhost:8000",
    variables: Optional[List[str]] = None,
) -> None:
    """
    Download ERA5 from the server for dates aligned to LST cadence, saving to
    data/retrieved_data/<roi_name>/era5/ERA5_data_YYYY-MM-DD.tif
    """
    feature = find_grid_feature(roi_name, grid_file)
    if feature is None:
        raise ValueError(f"ROI '{roi_name}' not found in grid {grid_file}")

    bbox = bbox_from_feature(feature)
    era5_dir = _ensure_dirs(output_base, roi_name)

    # Determine target dates from existing LST files (16-day cadence)
    lst_dir = os.path.join(output_base, roi_name, 'lst')
    target_dates: List[datetime] = []
    if os.path.isdir(lst_dir):
        lst_files = [f for f in os.listdir(lst_dir) if f.endswith('.tif')]
        for f in lst_files:
            try:
                d = f.split('_')[-1].replace('.tif', '')
                target_dates.append(datetime.strptime(d, '%Y-%m-%d'))
            except Exception:
                continue
        target_dates = sorted(set(target_dates))

    if not target_dates:
        # Fallback to simple 16-day grid from start_date to end_date
        sd = datetime.strptime(start_date, '%Y-%m-%d')
        ed = datetime.strptime(end_date, '%Y-%m-%d')
        cur = sd
        while cur <= ed:
            target_dates.append(cur)
            cur = cur + timedelta(days=16)

    client = ServerClient(api_base_url=api_base_url, timeout=120)

    for dt in target_dates:
        out_name = os.path.join(era5_dir, f"ERA5_data_{dt.strftime('%Y-%m-%d')}.tif")
        if os.path.exists(out_name):
            continue

        # 1) Create task for a narrow datetime window on the day
        start_iso = f"{dt.strftime('%Y-%m-%d')}T00:00:00Z"
        end_iso = f"{(dt + timedelta(days=1)).strftime('%Y-%m-%d')}T00:00:00Z"
        datetime_range = f"{start_iso}/{end_iso}"

        utc_hr = _infer_reference_and_hour(lst_dir, dt)

        try:
            task_id = client.create_era5_task(
                bbox=bbox,
                datetime_range_iso=datetime_range,
                variables=variables or config.ERA5_CONFIG["default_variables"],
                utc_hours=[utc_hr],
                limit=config.ERA5_CONFIG["default_limit"],
            )
        except Exception as e:
            print(f"Failed to create ERA5 task for {dt.date()}: {e}")
            continue

        # 2) Download and extract
        try:
            extracted_dir = client.download_and_extract("era5", task_id)
        except Exception as e:
            print(f"Failed to download ERA5 task {task_id} for {dt.date()}: {e}")
            continue

        # 3) Choose best file for target hour and save as expected filename
        tif_paths = glob.glob(os.path.join(extracted_dir, '**', '*.tif'), recursive=True)
        if not tif_paths:
            print(f"No TIFFs found for ERA5 task {task_id} ({dt.date()})")
            continue

        # Prefer skin_temperature files if present
        grouped = group_files_by_date(tif_paths)
        files_today_all = grouped.get(dt.strftime('%Y-%m-%d'), tif_paths)
        files_skin = [p for p in files_today_all if 'skin' in os.path.basename(p).lower()]
        files_today = files_skin if files_skin else files_today_all
        chosen = choose_file_closest_to_hour(files_today, target_hour=utc_hr)
        if not chosen:
            chosen = files_today[0]

        try:
            # Ensure single-band output to match legacy expectations
            acquisition_type = f"Hourly_{utc_hr:02d}:00"
            _write_single_band_copy(chosen, out_name, datetime_for_tags=dt, acquisition_type=acquisition_type)
            # Set nodata to NaN if absent
            with rasterio.open(out_name, 'r+') as dst:
                if dst.nodata is None:
                    dst.nodata = np.nan
        except Exception as e:
            print(f"Error writing ERA5 output for {dt.date()}: {e}")
            continue


