import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def find_grid_feature(phien_hieu: str, grid_file_path: str) -> Optional[Dict]:
    """
    Find a specific feature in a GeoJSON file based on its 'PhienHieu' property.
    Returns the feature dict or None.
    """
    try:
        with open(grid_file_path, 'r') as f:
            grid_data = json.load(f)
        for feature in grid_data.get('features', []):
            if feature.get('properties', {}).get('PhienHieu') == phien_hieu:
                return feature
        return None
    except Exception:
        return None


def bbox_from_feature(feature: Dict) -> List[float]:
    """
    Compute bounding box [minx, miny, maxx, maxy] from a GeoJSON Polygon feature.
    Assumes coordinates are in EPSG:4326.
    """
    coords = feature['geometry']['coordinates'][0]
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def extract_datetime_from_filename(filename: str) -> Optional[datetime]:
    """
    Extract a datetime from typical server-provided filenames.
    Supports patterns like:
      - era5_YYYYMMDD_HHMMSSZ.tif (e.g., skin_temperature_era5_20240601_040000Z.tif)
      - YYYYMMDDTHHMMSS
      - ISO format strings
    Returns a timezone-naive datetime in UTC, or None if not found.
    """
    import re

    era5_pattern = r'era5_(\d{8})_(\d{6})Z\.tif'
    iso_pattern = r'(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})'
    compact_pattern = r'(\d{8})T(\d{6})'

    def try_parse(date_part: str, time_part: str, fmt: str) -> Optional[datetime]:
        try:
            return datetime.strptime(f"{date_part}T{time_part}", fmt)
        except ValueError:
            return None

    m = re.search(era5_pattern, filename)
    if m:
        return try_parse(m.group(1), m.group(2), '%Y%m%dT%H%M%S')

    m = re.search(iso_pattern, filename)
    if m:
        return try_parse(m.group(1), f"{m.group(2)}{m.group(3)}{m.group(4)}", '%Y-%m-%dT%H%M%S')

    m = re.search(compact_pattern, filename)
    if m:
        return try_parse(m.group(1), m.group(2), '%Y%m%dT%H%M%S')

    return None


def group_files_by_date(filepaths: List[str]) -> Dict[str, List[str]]:
    """
    Group file paths by date string (YYYY-MM-DD) according to extracted datetime.
    Files with no parsable datetime are ignored.
    """
    groups: Dict[str, List[str]] = {}
    for fp in filepaths:
        dt = extract_datetime_from_filename(os.path.basename(fp))
        if not dt:
            continue
        date_key = dt.strftime('%Y-%m-%d')
        groups.setdefault(date_key, []).append(fp)
    return groups


def choose_file_closest_to_hour(filepaths: List[str], target_hour: int = 10) -> Optional[str]:
    """
    From a list of filepaths, choose the one whose filename datetime is
    closest to the target_hour (0-23). Returns None if none parse.
    """
    best_fp = None
    best_delta = None
    for fp in filepaths:
        dt = extract_datetime_from_filename(os.path.basename(fp))
        if not dt:
            continue
        delta = abs(dt.hour - target_hour)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_fp = fp
    return best_fp


