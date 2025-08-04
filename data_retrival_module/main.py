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
            
            if l8_dates:
                l8_pivot_date = min(l8_dates)
                logging.info(f"Dynamically determined L8 pivot date: {l8_pivot_date.strftime('%Y-%m-%d')}")
            else:
                l8_pivot_date = datetime(2013, 4, 11)  # Fallback if no L8 images found
                logging.warning(f"No L8 files found in {lst_output_folder}. Falling back to default L8 pivot date: {l8_pivot_date.strftime('%Y-%m-%d')}")
                
            if l9_dates:
                l9_pivot_date = min(l9_dates)
                logging.info(f"Dynamically determined L9 pivot date: {l9_pivot_date.strftime('%Y-%m-%d')}")
            else:
                l9_pivot_date = datetime(2022, 1, 1)  # Fallback if no L9 images found
                logging.warning(f"No L9 files found in {lst_output_folder}. Falling back to default L9 pivot date: {l9_pivot_date.strftime('%Y-%m-%d')}")

            # Generate target dates for ERA5 based on L8/L9 logic
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
            
            era5_target_dates = set()

            start_crawling_date = min(l9_pivot_date, l8_pivot_date)
            
            if l9_pivot_date <= l8_pivot_date:
                while start_crawling_date > start_date+timedelta(days=7):
                    start_crawling_date -= timedelta(days=8)
            else:
                while start_crawling_date > start_date+timedelta(days=15):
                    start_crawling_date -= timedelta(days=16)
            
            # 16-day interval before L9 pivot
            current_date_l8 = start_crawling_date
            while current_date_l8 < l9_pivot_date and current_date_l8 <= end_date:
                era5_target_dates.add(current_date_l8.strftime("%Y-%m-%d"))
                current_date_l8 += timedelta(days=16)

            # 8-day interval after L9 pivot
            current_date_l9 = max(start_crawling_date, l9_pivot_date)
            while current_date_l9 <= end_date:
                era5_target_dates.add(current_date_l9.strftime("%Y-%m-%d"))
                current_date_l9 += timedelta(days=8)

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

    # --- 6. Run MODIS Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting MODIS data retrieval for ROI '{roi_name}'...")
    try:
        lst_output_folder = os.path.join(args.output_folder, roi_name, "lst")
        modis_output_folder = os.path.join(args.output_folder, roi_name, "modis")

        if not os.path.exists(lst_output_folder) or not any(fname.lower().endswith('.tif') for fname in os.listdir(lst_output_folder)):
            logging.warning(f"Skipping MODIS retrieval: LST output folder for reference grid is empty or does not exist at '{lst_output_folder}'.")
        else:
            # Get unique dates from the downloaded LST filenames for MODIS retrieval
            lst_tif_files = [f for f in os.listdir(lst_output_folder) if f.lower().endswith('.tif')]
            modis_target_dates_str = set()
            for tif_file in lst_tif_files:
                try:
                    # Filename format is "{satellite}_lst16days_{YYYY-MM-DD}.tif"
                    date_str = tif_file.split('_')[-1].replace('.tif', '')
                    # Validate date format before adding
                    datetime.strptime(date_str, "%Y-%m-%d")
                    modis_target_dates_str.add(date_str)
                except (ValueError, IndexError):
                    logging.warning(f"Could not parse date from LST filename for MODIS: {tif_file}")

            if not modis_target_dates_str:
                logging.warning("No valid dates found from LST files. Skipping MODIS retrieval.")
            else:
                logging.info(f"Generated {len(modis_target_dates_str)} target dates for MODIS retrieval based on LST files.")

                # Pass the generated dates to the MODIS retrieval function
                modis_main_retrieval(
                    input_folder=lst_output_folder,
                    output_folder=modis_output_folder,
                    specific_dates=sorted(list(modis_target_dates_str))
                )

                logging.info(f"MODIS data retrieval finished for ROI '{roi_name}'.")

    except Exception as e:
        logging.critical(f"An error occurred during MODIS retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 7. Run Sentinel-2 Data Retrieval ---
    # For S2, we now process dates based on the LST files that were downloaded.
    logging.info("="*50)
    logging.info(f"Starting Sentinel-2 data retrieval for ROI '{roi_name}'...")
    try:
        lst_output_folder = os.path.join(args.output_folder, roi_name, "lst")
        
        if not os.path.exists(lst_output_folder) or not any(fname.lower().endswith('.tif') for fname in os.listdir(lst_output_folder)):
             logging.warning(f"Skipping Sentinel-2 retrieval: LST output folder is empty or does not exist at '{lst_output_folder}'.")
        else:
            # Get unique dates from the downloaded LST filenames
            lst_tif_files = [f for f in os.listdir(lst_output_folder) if f.lower().endswith('.tif')]
            s2_target_dates_str = set()
            for tif_file in lst_tif_files:
                try:
                    # Filename format is "{satellite}_lst16days_{YYYY-MM-DD}.tif"
                    date_str = tif_file.split('_')[-1].replace('.tif', '')
                    # Validate date format before adding
                    datetime.strptime(date_str, "%Y-%m-%d")
                    s2_target_dates_str.add(date_str)
                except (ValueError, IndexError):
                    logging.warning(f"Could not parse date from LST filename: {tif_file}")

            if not s2_target_dates_str:
                logging.warning("No valid dates found from LST files. Skipping Sentinel-2 retrieval.")
            else:
                target_composite_dates_s2 = [ee.Date(d) for d in sorted(list(s2_target_dates_str))]
                logging.info(f"Requesting S2 composites for {len(target_composite_dates_s2)} dates based on LST files.")
                
                # Call the S2 retrieval function
                main_s2_retrieval(target_composite_dates_s2, roi_geometry_ee, roi_name, args.output_folder)

        logging.info(f"Sentinel-2 data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during Sentinel-2 retrieval for '{roi_name}': {e}", exc_info=True)

    overall_end_time = time.time()
    logging.info("="*50)
    logging.info(f"Data retrieval orchestration finished. Total time: {timedelta(seconds=overall_end_time - overall_start_time)}")

if __name__ == '__main__':
    main() 