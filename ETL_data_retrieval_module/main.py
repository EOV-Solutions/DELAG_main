import argparse
import os
from datetime import datetime, timedelta
from typing import List

from .era5_from_server import retrieve_era5_from_server
from .lst_from_server import retrieve_lst_from_server
from .s2_from_server import retrieve_s2_from_server
from .aster_from_server import retrieve_aster_from_server
from .utils import find_grid_feature
from .config import config


def main():
    parser = argparse.ArgumentParser(description="ETL retrieval from server (replacing GEE).")
    parser.add_argument("--roi_name", required=True, help="Grid PhienHieu / ROI name")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--output_folder",
        default="/mnt/hdd12tb/code/nhatvm/DELAG_main/data/retrieved_data",
        help="Base output folder matching legacy structure",
    )
    parser.add_argument("--grid_file", default="data/Grid_50K_MatchedDates.geojson")
    parser.add_argument("--api_base_url", default=config.DEFAULT_API_BASE_URL)
    parser.add_argument("--datasets", nargs="+", choices=["lst", "era5", "s2", "aster"], default=["era5"],
                        help="Datasets to retrieve from server")

    args = parser.parse_args()

    roi_name = args.roi_name
    # Validate ROI exists in grid
    if not find_grid_feature(roi_name, args.grid_file):
        raise SystemExit(f"ROI '{roi_name}' not found in grid file {args.grid_file}")

    os.makedirs(os.path.join(args.output_folder, roi_name), exist_ok=True)

    if "lst" in args.datasets:
        retrieve_lst_from_server(
            roi_name=roi_name,
            grid_file=args.grid_file,
            start_date=args.start_date,
            end_date=args.end_date,
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
        )

    if "era5" in args.datasets:
        retrieve_era5_from_server(
            roi_name=roi_name,
            grid_file=args.grid_file,
            start_date=args.start_date,
            end_date=args.end_date,
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
            variables=config.ERA5_CONFIG["default_variables"],
        )

    if "s2" in args.datasets:
        # Derive composite dates from LST files if present; else use 16-day intervals
        lst_dir = os.path.join(args.output_folder, roi_name, 'lst')
        composite_dates: List[str] = []
        
        if os.path.isdir(lst_dir):
            # Extract dates from existing LST files
            lst_files = [f for f in os.listdir(lst_dir) if f.endswith('.tif')]
            for f in lst_files:
                try:
                    date_part = f.split('_')[-1].replace('.tif', '')
                    datetime.strptime(date_part, '%Y-%m-%d')  # Validate format
                    if date_part not in composite_dates:
                        composite_dates.append(date_part)
                except Exception:
                    continue
            composite_dates = sorted(composite_dates)
        
        if not composite_dates:
            # Fallback to 16-day intervals between start and end dates
            start_dt = datetime.strptime(args.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(args.end_date, '%Y-%m-%d')
            current_dt = start_dt
            while current_dt <= end_dt:
                composite_dates.append(current_dt.strftime('%Y-%m-%d'))
                current_dt = current_dt + timedelta(days=16)
        
        retrieve_s2_from_server(
            roi_name=roi_name,
            grid_file=args.grid_file,
            composite_dates=composite_dates,
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
        )

    if "aster" in args.datasets:
        retrieve_aster_from_server(
            roi_name=roi_name,
            grid_file=args.grid_file,
            start_date=args.start_date,
            end_date=args.end_date,
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
            bands=config.ASTER_CONFIG["default_bands"],
        )


if __name__ == "__main__":
    main()


