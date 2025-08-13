import argparse
import json
import logging
import os
from datetime import datetime, timedelta
import time

import ee

# Import the retrieval functions from your existing modules
from lst_retrieval import lst_retrive
from s2_retrieval import main_s2_retrieval
from era5_retriever import main as era5_main_retrieval
from modis_retriever import main as modis_main_retrieval
from slstr_lst_retriever import main_from_dates as slstr_main_retrieval

# Configure logging for the orchestrator
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_orchestrator.log')
    ]
)

def find_grid_feature(phien_hieu: str, grid_file_path: str) -> dict | None:
    """
    Finds a specific feature in a GeoJSON file based on its 'PhienHieu' property.

    Args:
        phien_hieu: The 'PhienHieu' identifier to search for.
        grid_file_path: Path to the GeoJSON grid file.

    Returns:
        The matching GeoJSON feature dictionary, or None if not found.
    """
    try:
        with open(grid_file_path, 'r') as f:
            grid_data = json.load(f)
        
        for feature in grid_data.get('features', []):
            if feature.get('properties', {}).get('PhienHieu') == phien_hieu:
                return feature
        
        logging.error(f"Failed to find feature with PhienHieu '{phien_hieu}'.")
        return None
    except FileNotFoundError:
        logging.critical(f"Grid file not found at: {grid_file_path}")
        return None
    except json.JSONDecodeError:
        logging.critical(f"Failed to read grid file: {grid_file_path}")
        return None

def main():
    """Main function to orchestrate the LST and S2 data retrieval."""
    overall_start_time = time.time()
    parser = argparse.ArgumentParser(description="Orchestrate LST and Sentinel-2 data retrieval for a specific grid and time interval.")
    parser.add_argument("--roi_name", type=str, required=True, help="The 'PhienHieu' identifier of the grid to process.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for data retrieval in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, required=True, help="End date for data retrieval in YYYY-MM-DD format.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/mnt/hdd12tb/code/nhatvm/DELAG_main/data/retrieved_data",
        help="The main folder to save the downloaded data."
    )
    parser.add_argument(
        "--grid_file",
        type=str,
        default="data/Grid_50K_MatchedDates.geojson",
        help="Path to the GeoJSON grid file."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'test'],
        default='train',
        help="Set retrieval mode. 'train' for LST, ERA5, S2. 'test' for all datasets."
    )
    args = parser.parse_args()

    # --- 1. Initialize Earth Engine ---
    try:
        ee.Initialize(project='ee-hadat-461702-p4')
    except Exception:
        logging.info("Authenticating to Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project='ee-hadat-461702-p4')

    # --- 2. Find the requested grid feature ---
    feature = find_grid_feature(args.roi_name, args.grid_file)
    if not feature:
        return # Stop execution if feature is not found

    roi_name = feature['properties']['PhienHieu']
    roi_geometry_data = feature['geometry']
    roi_geometry_ee = ee.Geometry.Polygon(roi_geometry_data['coordinates'])

    # --- 3. Run LST Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting LST data retrieval for ROI '{roi_name}'...")
    try:
        start_date_lst = args.start_date
        end_date_lst = args.end_date
        logging.info(f"LST retrieval period: {start_date_lst} to {end_date_lst}")
        
        # Call the LST retrieval function
        lst_retrive(start_date_lst, end_date_lst, roi_geometry_ee, roi_name, args.output_folder)
        logging.info(f"LST data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during LST retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 4. Run ERA5 Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting ERA5 data retrieval for ROI '{roi_name}'...")
    try:
        lst_output_folder = os.path.join(args.output_folder, roi_name, "lst")
        era5_output_folder = os.path.join(args.output_folder, roi_name, "era5")

        if not os.path.exists(lst_output_folder) or not any(fname.lower().endswith('.tif') for fname in os.listdir(lst_output_folder)):
            logging.warning(f"Skipping ERA5 retrieval: LST output folder for reference grid is empty or does not exist at '{lst_output_folder}'.")
        else:
            # Find the first L8 and L9 dates to use as pivots
            l8_files = [f for f in os.listdir(lst_output_folder) if f.startswith("L8_") and f.lower().endswith('.tif')]
            l9_files = [f for f in os.listdir(lst_output_folder) if f.startswith("L9_") and f.lower().endswith('.tif')]
            
            l8_dates = []
            for f in l8_files:
                try:
                    date_str = f.split('_')[-1].replace('.tif', '')
                    l8_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                except (ValueError, IndexError):
                    logging.warning(f"Could not parse date from L8 filename: {f}")
                    continue
            
            l9_dates = []
            for f in l9_files:
                try:
                    date_str = f.split('_')[-1].replace('.tif', '')
                    l9_dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                except (ValueError, IndexError):
                    logging.warning(f"Could not parse date from L9 filename: {f}")
                    continue
            # Define L9 pivot date
            l9_pivot_date = datetime(2022, 1, 1)  # Starting date of L9 satellite
            
            # Extract min/max dates for L8 and L9
            min_l8_date = None
            max_l8_date = None
            min_l9_date = None
            max_l9_date = None
            
            if l8_dates:
                min_l8_date = min(l8_dates)
                max_l8_date = max(l8_dates)
                logging.info(f"L8 date range: {min_l8_date.strftime('%Y-%m-%d')} to {max_l8_date.strftime('%Y-%m-%d')}")
            
            if l9_dates:
                min_l9_date = min(l9_dates)
                max_l9_date = max(l9_dates)
                logging.info(f"L9 date range: {min_l9_date.strftime('%Y-%m-%d')} to {max_l9_date.strftime('%Y-%m-%d')}")
            
            # Generate target dates for ERA5 based on new algorithm
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            
            era5_target_dates = set()
            
            # Case 1: Both L8 and L9 dates available
            if min_l8_date and min_l9_date:
                logging.info("Both L8 and L9 dates available. Creating merged time series.")
                
                # Create 16-day interval series for L8
                current_date = min_l8_date
                while current_date <= max_l8_date and current_date <= end_date:
                    era5_target_dates.add(current_date.strftime("%Y-%m-%d"))
                    current_date += timedelta(days=16)
                
                # Create 16-day interval series for L9
                current_date = min_l9_date
                while current_date <= max_l9_date and current_date <= end_date:
                    era5_target_dates.add(current_date.strftime("%Y-%m-%d"))
                    current_date += timedelta(days=16)
            
            # Case 2: Only one satellite has dates
            elif min_l8_date or min_l9_date:
                available_min_date = min_l8_date if min_l8_date else min_l9_date
                available_max_date = max_l8_date if min_l8_date else max_l9_date
                satellite_name = "L8" if min_l8_date else "L9"
                
                logging.info(f"Only {satellite_name} dates available. Creating time series and finding nearest to L9 pivot.")
                
                # Create 16-day interval series for available satellite
                available_dates = []
                current_date = available_min_date
                while current_date <= available_max_date and current_date <= end_date:
                    available_dates.append(current_date)
                    current_date += timedelta(days=16)
                
                # Find nearest day to L9 pivot date
                nearest_day = min(available_dates, key=lambda x: abs((x - l9_pivot_date).days))
                logging.info(f"Nearest {satellite_name} date to L9 pivot: {nearest_day.strftime('%Y-%m-%d')}")
                
                # Create list of (nearest_day ± 8 days)
                candidate_dates = []
                for i in range(-8, 9, 8):
                    candidate_date = nearest_day + timedelta(days=i)
                    candidate_dates.append(candidate_date)
                
                # Filter to dates > L9 pivot date
                filtered_dates = [d for d in candidate_dates if d > l9_pivot_date and d <= end_date and d >= start_date]
                
                if filtered_dates:
                    # Find nearest to L9 pivot from filtered list
                    min_date_remaining = min(filtered_dates, key=lambda x: abs((x - l9_pivot_date).days))
                    logging.info(f"Selected min date for remaining satellite: {min_date_remaining.strftime('%Y-%m-%d')}")
                    
                    # Create 16-day interval series from min_date_remaining to end_date
                    current_date = min_date_remaining
                    while current_date <= end_date:
                        era5_target_dates.add(current_date.strftime("%Y-%m-%d"))
                        current_date += timedelta(days=16)
                
                # Add the available satellite dates
                for date in available_dates:
                    era5_target_dates.add(date.strftime("%Y-%m-%d"))
            
            # Case 3: No LST dates available
            else:
                logging.warning("No LST dates available. Returning empty target dates.")
                era5_target_dates = set()

            logging.info(f"Generated {len(era5_target_dates)} target dates for ERA5 retrieval.")

            # Pass the generated dates to the retrieval function
            era5_main_retrieval(
                input_folder=lst_output_folder, 
                output_folder=era5_output_folder,
                specific_dates=sorted(list(era5_target_dates))
            )

            # Final check for missing ERA5 files based on actual LST files
            logging.info("Running final check for missing ERA5 files based on LST downloads...")
            lst_tif_files = [f for f in os.listdir(lst_output_folder) if f.lower().endswith('.tif')]
            missing_dates = []
            for tif_file in lst_tif_files:
                try:
                    date_str = tif_file.split('_')[-1].replace('.tif', '')
                    datetime.strptime(date_str, "%Y-%m-%d") # Validate format
                    era5_expected_file = os.path.join(era5_output_folder, f"ERA5_data_{date_str}.tif")
                    if not os.path.exists(era5_expected_file):
                        missing_dates.append(date_str)
                except (ValueError, IndexError):
                    continue
            
            if missing_dates:
                logging.warning(f"Found {len(missing_dates)} LST files with no corresponding ERA5 file. Retrieving missing data...")
                era5_main_retrieval(
                    input_folder=lst_output_folder,
                    output_folder=era5_output_folder,
                    specific_dates=sorted(missing_dates)
                )
            else:
                logging.info("Final check complete. No missing ERA5 files found.")

            logging.info(f"ERA5 data retrieval finished for ROI '{roi_name}'.")

    except Exception as e:
        logging.critical(f"An error occurred during ERA5 retrieval for '{roi_name}': {e}", exc_info=True)

    # --- Get target dates from LST files for other retrievals ---
    lst_output_folder = os.path.join(args.output_folder, roi_name, "lst")
    target_dates_str = set()
    if not os.path.exists(lst_output_folder) or not any(fname.lower().endswith('.tif') for fname in os.listdir(lst_output_folder)):
        logging.warning(f"LST output folder is empty or does not exist at '{lst_output_folder}'. Skipping dependent retrievals.")
    else:
        lst_tif_files = [f for f in os.listdir(lst_output_folder) if f.lower().endswith('.tif')]
        for tif_file in lst_tif_files:
            try:
                date_str = tif_file.split('_')[-1].replace('.tif', '')
                datetime.strptime(date_str, "%Y-%m-%d") # Validate format
                target_dates_str.add(date_str)
            except (ValueError, IndexError):
                logging.warning(f"Could not parse date from LST filename: {tif_file}")

    if target_dates_str:
        # --- Sentinel-2 Data Retrieval (train and test modes) ---
        logging.info("="*50)
        logging.info(f"Starting Sentinel-2 data retrieval for ROI '{roi_name}'...")
        try:
            target_composite_dates_s2 = [ee.Date(d) for d in sorted(list(target_dates_str))]
            logging.info(f"Requesting S2 composites for {len(target_composite_dates_s2)} dates based on LST files.")
            main_s2_retrieval(target_composite_dates_s2, roi_geometry_ee, roi_name, args.output_folder)
            logging.info(f"Sentinel-2 data retrieval finished for ROI '{roi_name}'.")
        except Exception as e:
            logging.critical(f"An error occurred during Sentinel-2 retrieval for '{roi_name}': {e}", exc_info=True)

        # --- Test Mode Only Retrievals ---
        if args.mode == 'test':
            # --- MODIS Data Retrieval ---
            logging.info("="*50)
            logging.info(f"Starting MODIS data retrieval for ROI '{roi_name}'...")
            try:
                modis_output_folder = os.path.join(args.output_folder, roi_name, "modis")
                logging.info(f"Generated {len(target_dates_str)} target dates for MODIS retrieval based on LST files.")
                modis_main_retrieval(
                    input_folder=lst_output_folder,
                    output_folder=modis_output_folder,
                    specific_dates=sorted(list(target_dates_str))
                )
                logging.info(f"MODIS data retrieval finished for ROI '{roi_name}'.")
            except Exception as e:
                logging.critical(f"An error occurred during MODIS retrieval for '{roi_name}': {e}", exc_info=True)

            # --- S3 SLSTR LST Data Retrieval ---
            logging.info("="*50)
            logging.info(f"Starting S3 SLSTR LST data retrieval for ROI '{roi_name}'...")
            try:
                logging.info(f"Generated {len(target_dates_str)} target dates for S3 SLSTR retrieval based on LST files.")
                slstr_main_retrieval(
                    roi_name=roi_name,
                    dates=sorted(list(target_dates_str)),
                    output_folder=args.output_folder,
                    grid_file=args.grid_file,
                    units='kelvin' # Fetching in Kelvin
                )
                logging.info(f"S3 SLSTR LST data retrieval finished for ROI '{roi_name}'.")
            except Exception as e:
                logging.critical(f"An error occurred during S3 SLSTR retrieval for '{roi_name}': {e}", exc_info=True)
    else:
        logging.warning("No valid dates found from LST files. Skipping S2, MODIS, and S3_SLSTR retrieval.")


    overall_end_time = time.time()
    logging.info("="*50)
    logging.info(f"Data retrieval orchestration finished. Total time: {timedelta(seconds=overall_end_time - overall_start_time)}")

if __name__ == '__main__':
    main() 