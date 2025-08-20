import argparse
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, Optional, List

import openeo
import rasterio
from rasterio.warp import reproject, Resampling
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class RetrievalConfig:
    provider_url: str
    collection_id: str
    output_root: str
    roi_name: str
    start_date: str
    end_date: str
    grid_file: str
    units: str


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _dates_from_lst_folder(lst_folder: str) -> List[datetime]:
    dates: List[datetime] = []
    try:
        for name in os.listdir(lst_folder):
            if not name.lower().endswith('.tif'):
                continue
            try:
                date_str = name.split('_')[-1].replace('.tif', '')
                dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
            except Exception:
                continue
    except FileNotFoundError:
        logging.error("LST folder not found: %s", lst_folder)
    return sorted(list(set(dates)))


def find_grid_feature(roi_name: str, grid_file_path: str) -> Optional[Dict]:
    """
    Finds a specific feature in a GeoJSON file based on its 'PhienHieu'/'Phien_Hieu' property.

    This mirrors the logic used elsewhere in the project to keep behavior consistent.

    Args:
        roi_name: The 'PhienHieu' identifier to search for.
        grid_file_path: Path to the GeoJSON grid file.

    Returns:
        The matching GeoJSON feature dictionary, or None if not found.
    """
    try:
        with open(grid_file_path, 'r') as f:
            grid_data = json.load(f)

        for feature in grid_data.get('features', []):
            props = feature.get('properties', {})
            candidate = props.get('PhienHieu') or props.get('Phien_Hieu')
            if candidate == roi_name:
                return feature

        logging.error(f"Failed to find feature with PhienHieu/Phien_Hieu '{roi_name}'.")
        return None
    except FileNotFoundError:
        logging.critical(f"Grid file not found at: {grid_file_path}")
        return None
    except json.JSONDecodeError:
        logging.critical(f"Failed to read grid file: {grid_file_path}")
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_connection(provider_url: str) -> openeo.Connection:
    connection = openeo.connect(provider_url)
    # Will open device code flow in the terminal if needed
    connection.authenticate_oidc()
    return connection


def build_spatial_extent_from_geojson_geometry(geometry: Dict) -> Dict:
    """
    Returns a spatial extent dict suitable for openEO from a GeoJSON geometry.
    Most backends accept the GeoJSON geometry directly as the spatial extent.
    """
    return geometry


def download_daily_slstr_lst(
    connection: openeo.Connection,
    collection_id: str,
    spatial_extent: Dict,
    date: datetime,
    destination_path: str,
    to_celsius: bool,
    resample_to_30m_flag: bool = True,
) -> None:
    """
    Downloads a daily LST composite for the given date and spatial extent with cloud masking.

    Strategy:
    - Load one-day temporal slice with LST and confidence_in bands
    - Apply cloud mask using confidence_in band (mask pixels where confidence_in == 0)
    - Optionally convert from Kelvin to Celsius
    - Aggregate over the day (mean)
    - Optionally resample to 30m resolution on the back-end
    - Save as GeoTIFF to destination_path
    """
    t_start = date.strftime("%Y-%m-%d")
    t_end = (date + timedelta(days=1)).strftime("%Y-%m-%d")

    # Load the collection with LST and confidence_in bands
    cube = connection.load_collection(
        collection_id=collection_id,
        spatial_extent=spatial_extent,
        temporal_extent=[t_start, t_end],
        bands=["LST", "confidence_in"]
    )

    # Apply cloud mask: mask pixels where confidence_in == 0 (cloudy or invalid)
    cloud_mask = cube.band("confidence_in") == 0
    cube = cube.mask(cloud_mask, replacement=None)

    # Keep only the LST band after masking
    cube = cube.filter_bands(["LST"])

    # --- IMPORTANT: Apply scale factor and offset to convert from DN to Kelvin ---
    # According to product spec, LST is scaled: LST(K) = DN * 0.0020000001 + 290
    cube = cube * 0.0020000001 + 290

    # Optionally convert Kelvin to Celsius: x - 273.15
    if to_celsius:
        cube = cube - 273.15

    # Aggregate to a single slice within the day
    cube = cube.aggregate_temporal_period(period="day", reducer="mean")

    # Resample to 30m resolution on the back-end if requested.
    # We assume the collection's CRS is geographic (e.g., WGS84) and use an
    # approximate resolution in degrees. 30m is approx. 0.00027 degrees.
    if resample_to_30m_flag:
        cube = cube.resample_spatial(resolution=0.00027, method="bilinear")

    # Save/download
    cube.download(destination_path, format="GTIFF")


def run(config: RetrievalConfig) -> None:
    # 1) Connect to openEO
    logging.info(f"Connecting to openEO provider: {config.provider_url}")
    connection = create_connection(config.provider_url)

    # 2) Locate ROI feature
    logging.info(f"Reading grid from: {config.grid_file}")
    feature = find_grid_feature(config.roi_name, config.grid_file)
    if not feature:
        raise SystemExit(1)
    roi_name = feature.get('properties', {}).get('PhienHieu') or feature.get('properties', {}).get('Phien_Hieu') or config.roi_name
    geometry = feature['geometry']
    spatial_extent = build_spatial_extent_from_geojson_geometry(geometry)

    # 3) Prepare output structure
    output_dir = os.path.join(config.output_root, roi_name, "s3_slstr")
    ensure_dir(output_dir)

    # 4) Determine dates
    should_convert_to_celsius = config.units == "celsius"

    if config.start_date and config.end_date:
        start_dt = _parse_date(config.start_date)
        end_dt = _parse_date(config.end_date)
        dates: List[datetime] = list(_daterange(start_dt, end_dt))
    else:
        raise SystemExit("start_date and end_date are required when not using dates-based main.")

    logging.info(
        f"Starting Sentinel-3 SLSTR L2 LST retrieval for '{roi_name}' for {len(dates)} dates."
    )

    for day in tqdm(dates, desc="Downloading daily SLSTR LST"):
        out_file = os.path.join(
            output_dir,
            f"S3_SLSTR_LST_{day.strftime('%Y-%m-%d')}.tif",
        )
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            logging.info(f"Skipping download for {day.strftime('%Y-%m-%d')}: file already exists.")
            continue

        try:
            download_daily_slstr_lst(
                connection=connection,
                collection_id=config.collection_id,
                spatial_extent=spatial_extent,
                date=day,
                destination_path=out_file,
                to_celsius=should_convert_to_celsius,
                resample_to_30m_flag=True,
            )
        except Exception as exc:
            logging.warning(
                f"Failed to download {day.strftime('%Y-%m-%d')} for '{roi_name}': {exc}"
            )

    logging.info(f"Finished. Output written under: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve Sentinel-3 SLSTR Level-2 LST via openEO for a given grid (PhienHieu) and date range, or from a list of dates in an existing LST folder. "
            "Saves daily GeoTIFFs in specified units."
        )
    )
    parser.add_argument(
        "--roi_name",
        type=str,
        required=True,
        help="The 'PhienHieu'/'Phien_Hieu' identifier of the grid to process.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Start date in YYYY-MM-DD (used if --lst_folder is not provided).",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="End date in YYYY-MM-DD (used if --lst_folder is not provided).",
    )
    parser.add_argument(
        "--lst_folder",
        type=str,
        required=False,
        help="Path to an existing LST folder containing files named ..._YYYY-MM-DD.tif; dates will be derived from filenames.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/mnt/hdd12tb/code/nhatvm/DELAG_main/data/retrieved_data",
        help="Root folder to save results: <output>/<PhienHieu>/s3_slstr/*.tif",
    )
    parser.add_argument(
        "--grid_file",
        type=str,
        default="data/Grid_50K_MatchedDates.geojson",
        help="Path to the grid GeoJSON.",
    )
    parser.add_argument(
        "--provider_url",
        type=str,
        default="https://openeofed.dataspace.copernicus.eu",
        help="openEO back-end URL.",
    )
    parser.add_argument(
        "--collection_id",
        type=str,
        default="SENTINEL3_SLSTR_L2_LST",
        help=(
            "Collection ID for SLSTR L2 LST on the selected openEO back-end. "
            "Adjust if your back-end uses a different identifier."
        ),
    )
    parser.add_argument(
        "--units",
        type=str,
        default="kelvin",
        choices=["celsius", "kelvin"],
        help="Output units for the LST data. Default is 'kelvin'."
    )
    return parser


# --- Dates-based main (for integration with orchestrator) ---

def main_from_dates(roi_name: str, dates: List[str], output_folder: str, grid_file: str,
                    provider_url: str = "https://openeofed.dataspace.copernicus.eu",
                    collection_id: str = "SENTINEL3_SLSTR_L2_LST",
                    units: str = "kelvin") -> None:
    """
    Crawl SLSTR L2 LST for an explicit list of dates (YYYY-MM-DD strings) for one ROI.
    Designed to mirror the pattern used by ERA5 retrieval (driven by LST file dates).
    """
    # Connect
    logging.info(f"Connecting to openEO provider: {provider_url}")
    connection = create_connection(provider_url)

    # ROI feature
    feature = find_grid_feature(roi_name, grid_file)
    if not feature:
        raise RuntimeError(f"ROI '{roi_name}' not found in grid: {grid_file}")

    roi_name_final = feature.get('properties', {}).get('PhienHieu') or feature.get('properties', {}).get('Phien_Hieu') or roi_name
    geometry = feature['geometry']
    spatial_extent = build_spatial_extent_from_geojson_geometry(geometry)

    # Output
    output_dir = os.path.join(output_folder, roi_name_final, "s3_slstr")
    ensure_dir(output_dir)

    # Dates
    date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
    to_celsius = (units == "celsius")

    logging.info(f"Requesting SLSTR L2 LST for {len(date_objs)} dates (units={units})")

    for day in tqdm(date_objs, desc="Downloading daily SLSTR LST"):
        out_file = os.path.join(output_dir, f"S3_SLSTR_LST_{day.strftime('%Y-%m-%d')}.tif")
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            logging.info(f"Skipping download for {day.strftime('%Y-%m-%d')}: file already exists.")
            continue
        try:
            download_daily_slstr_lst(
                connection=connection,
                collection_id=collection_id,
                spatial_extent=spatial_extent,
                date=day,
                destination_path=out_file,
                to_celsius=to_celsius,
                resample_to_30m_flag=True,
            )
        except Exception as exc:
            logging.warning("Failed to download %s: %s", day.strftime('%Y-%m-%d'), exc)


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    # Allow two modes: range mode (start/end) or folder-driven mode (lst_folder)
    if not args.lst_folder and (not args.start_date or not args.end_date):
        raise SystemExit("Provide either --lst_folder or both --start_date and --end_date")

    cfg = RetrievalConfig(
        provider_url=args.provider_url,
        collection_id=args.collection_id,
        output_root=args.output_folder,
        roi_name=args.roi_name,
        start_date=args.start_date or "",
        end_date=args.end_date or "",
        grid_file=args.grid_file,
        units=args.units,
    )

    if args.lst_folder:
        # Derive dates from existing LST folder filenames and call the dates-based main
        date_list = [d.strftime('%Y-%m-%d') for d in _dates_from_lst_folder(args.lst_folder)]
        if not date_list:
            raise SystemExit(f"No valid dates found in lst_folder: {args.lst_folder}")
        main_from_dates(
            roi_name=cfg.roi_name,
            dates=date_list,
            output_folder=cfg.output_root,
            grid_file=cfg.grid_file,
            provider_url=cfg.provider_url,
            collection_id=cfg.collection_id,
            units=cfg.units,
        )
    else:
        run(cfg)