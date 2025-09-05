import argparse
import json
import os
from typing import Dict, List, Union

from .era5_from_server import retrieve_era5_from_server
from .lst_from_server import retrieve_lst_from_server
from .s2_from_server import retrieve_s2_from_server
from .aster_from_server import retrieve_aster_from_server
from .config import config


def main():
    parser = argparse.ArgumentParser(description="ETL retrieval from server using predefined task IDs.")
    parser.add_argument("--roi_name", required=True, help="Grid PhienHieu / ROI name for output folder structure")
    parser.add_argument("--task_ids", required=True, help="JSON string or file path containing task ID dictionary")
    parser.add_argument(
        "--output_folder",
        default="/mnt/hdd12tb/code/nhatvm/DELAG_main/data/retrieved_data",
        help="Base output folder matching legacy structure",
    )
    parser.add_argument("--api_base_url", default=config.DEFAULT_API_BASE_URL)
    parser.add_argument("--datasets", nargs="+", choices=["lst", "era5", "s2", "aster"], default=["era5"],
                        help="Datasets to retrieve from server")

    args = parser.parse_args()

    # Parse task_ids (JSON string or file path)
    try:
        if os.path.isfile(args.task_ids):
            with open(args.task_ids, 'r') as f:
                task_ids = json.load(f)
        else:
            task_ids = json.loads(args.task_ids)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise SystemExit(f"Invalid task_ids format: {e}")

    roi_name = args.roi_name
    os.makedirs(os.path.join(args.output_folder, roi_name), exist_ok=True)

    # Process each dataset using predefined task IDs
    if "era5" in args.datasets and "era5" in task_ids:
        retrieve_era5_from_server(
            roi_name=roi_name,
            task_ids=task_ids["era5"],
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
        )

    if "s2" in args.datasets and "s2" in task_ids:
        retrieve_s2_from_server(
            roi_name=roi_name,
            task_ids=task_ids["s2"],
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
        )

    if "lst" in args.datasets and "lst" in task_ids:
        retrieve_lst_from_server(
            roi_name=roi_name,
            task_ids=task_ids["lst"],
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
        )

    if "aster" in args.datasets and "aster" in task_ids:
        retrieve_aster_from_server(
            roi_name=roi_name,
            task_ids=task_ids["aster"],
            output_base=args.output_folder,
            api_base_url=args.api_base_url,
        )


if __name__ == "__main__":
    main()


