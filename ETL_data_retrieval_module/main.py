import argparse
import json
import logging
import os
from datetime import datetime, timedelta
import time

try:
    import shapefile
except ImportError:
    logging.critical("The 'pyshp' library is required for shapefile processing. Please install it: pip install pyshp")
    shapefile = None


# Import the offline retrieval functions from your existing modules
try:
    from .lst_retrieval import lst_retrive_offline
    from .s2_retrieval import main_s2_retrieval_offline
    from .era5_retriever import main_offline as era5_main_retrieval_offline
except ImportError:
    # Fallback for when running as script
    from lst_retrieval import lst_retrive_offline
    from s2_retrieval import main_s2_retrieval_offline
    from era5_retriever import main_offline as era5_main_retrieval_offline

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

def get_roi_from_shp_folder(folder_path: str) -> dict | None:
    """
    Reads a shapefile from a folder and returns the geometry of the first feature.
    Args:
        folder_path: Path to the folder containing shapefile components 
                     (.shp, .shx, .dbf, etc.).
    Returns:
        A GeoJSON-like geometry dictionary for Earth Engine, or None if an error occurs.
    """
    if shapefile is None:
        logging.error("'pyshp' library is not installed, cannot process shapefile ROI.")
        return None

    try:
        shp_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.shp')]
        if not shp_files:
            logging.error(f"No .shp file found in folder: {folder_path}")
            return None
        
        # Take the first .shp file found
        shp_path = os.path.join(folder_path, shp_files[0])
        
        with shapefile.Reader(shp_path) as shp:
            if not shp.shapes():
                logging.error(f"Shapefile is empty or contains no geometries: {shp_path}")
                return None
            
            # Use the geometry of the first shape as the ROI
            first_shape = shp.shapes()[0]
            
            # The __geo_interface__ provides a GeoJSON-like dictionary
            roi_geometry = first_shape.__geo_interface__
            return roi_geometry

    except FileNotFoundError:
        logging.critical(f"ROI folder not found at: {folder_path}")
        return None
    except Exception as e:
        logging.critical(f"Failed to read or process shapefile from '{folder_path}': {e}", exc_info=True)
        return None

def main_offline():
    """Main function to orchestrate offline data retrieval from processed temp data."""
    overall_start_time = time.time()
    parser = argparse.ArgumentParser(description="Orchestrate offline data retrieval from processed temp data for a specific grid and time interval.")
    parser.add_argument("--roi_name", type=str, required=True, help="The 'PhienHieu' identifier of the grid to process.")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to processed temp data directory (e.g., temp_processed_data/D-49-49-A).")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for data retrieval in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, required=True, help="End date for data retrieval in YYYY-MM-DD format.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/retrieved_data",
        help="The main folder to save the retrieved data."
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

    # --- 1. Validate processed data directory ---
    if not os.path.exists(args.processed_data_dir):
        logging.critical(f"Processed data directory not found: {args.processed_data_dir}")
        return

    # --- 2. Find the requested grid feature (for validation) ---
    feature = find_grid_feature(args.roi_name, args.grid_file)
    if not feature:
        return # Stop execution if feature is not found

    roi_name = feature['properties']['PhienHieu']
    logging.info(f"Processing ROI: {roi_name}")

    # --- 3. Generate target dates from date range ---
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Generate dates in the range (daily for now, can be adjusted)
    target_dates = []
    current_date = start_date
    while current_date <= end_date:
        target_dates.append(current_date)
        current_date += timedelta(days=1)
    
    logging.info(f"Generated {len(target_dates)} target dates from {args.start_date} to {args.end_date}")

    # --- 4. Run LST Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting LST data retrieval for ROI '{roi_name}'...")
    try:
        lst_output_folder = os.path.join(args.output_folder, roi_name, "lst")
        lst_retrive_offline(args.processed_data_dir, lst_output_folder, target_dates)
        logging.info(f"LST data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during LST retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 5. Run ERA5 Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting ERA5 data retrieval for ROI '{roi_name}'...")
    try:
        era5_output_folder = os.path.join(args.output_folder, roi_name, "era5")
        era5_main_retrieval_offline(args.processed_data_dir, era5_output_folder, target_dates)
        logging.info(f"ERA5 data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during ERA5 retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 6. Run Sentinel-2 Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting Sentinel-2 data retrieval for ROI '{roi_name}'...")
    try:
        s2_output_folder = os.path.join(args.output_folder, roi_name, "s2_images")
        main_s2_retrieval_offline(args.processed_data_dir, s2_output_folder, target_dates)
        logging.info(f"Sentinel-2 data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during Sentinel-2 retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 7. Test Mode Only Retrievals (if needed) ---
    if args.mode == 'test':
        logging.info("Test mode selected, but additional datasets (MODIS, S3 SLSTR) are not available in offline mode.")
        logging.info("These datasets would need to be processed separately if required.")

    overall_end_time = time.time()
    logging.info("="*50)
    logging.info(f"Data retrieval orchestration finished. Total time: {timedelta(seconds=overall_end_time - overall_start_time)}")

def main():
    """Legacy main function for backward compatibility."""
    logging.warning("Using legacy main function. Consider using main_offline() for processed data.")
    # This would need to be updated to work with the new offline approach
    # For now, just log a warning
    logging.info("Legacy function called - no action taken")

def main_shp_offline():
    """Main function to orchestrate offline data retrieval using a shapefile ROI."""
    overall_start_time = time.time()
    parser = argparse.ArgumentParser(description="Orchestrate offline data retrieval for a specific ROI defined by a shapefile.")
    parser.add_argument("--roi_folder", type=str, required=True, help="Folder containing the ROI shapefile components (.shp, .shx, .dbf).")
    parser.add_argument("--roi_name", type=str, required=True, help="A name for the ROI, used for output folder naming.")
    parser.add_argument("--processed_data_dir", type=str, required=True, help="Path to processed temp data directory (e.g., temp_processed_data/D-49-49-A).")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for data retrieval in YYYY-MM-DD format.")
    parser.add_argument("--end_date", type=str, required=True, help="End date for data retrieval in YYYY-MM-DD format.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/retrieved_data",
        help="The main folder to save the retrieved data."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'test'],
        default='train',
        help="Set retrieval mode. 'train' for LST, ERA5, S2. 'test' for all datasets."
    )
    args = parser.parse_args()

    # --- 1. Validate processed data directory ---
    if not os.path.exists(args.processed_data_dir):
        logging.critical(f"Processed data directory not found: {args.processed_data_dir}")
        return

    # --- 2. Get ROI from Shapefile (for validation) ---
    roi_geometry_data = get_roi_from_shp_folder(args.roi_folder)
    if not roi_geometry_data:
        return # Stop execution if feature is not found
    
    roi_name = args.roi_name
    logging.info(f"Processing ROI: {roi_name}")

    # --- 3. Generate target dates from date range ---
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Generate dates in the range (daily for now, can be adjusted)
    target_dates = []
    current_date = start_date
    while current_date <= end_date:
        target_dates.append(current_date)
        current_date += timedelta(days=1)
    
    logging.info(f"Generated {len(target_dates)} target dates from {args.start_date} to {args.end_date}")

    # --- 4. Run LST Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting LST data retrieval for ROI '{roi_name}'...")
    try:
        lst_output_folder = os.path.join(args.output_folder, roi_name, "lst")
        lst_retrive_offline(args.processed_data_dir, lst_output_folder, target_dates)
        logging.info(f"LST data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during LST retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 5. Run ERA5 Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting ERA5 data retrieval for ROI '{roi_name}'...")
    try:
        era5_output_folder = os.path.join(args.output_folder, roi_name, "era5")
        era5_main_retrieval_offline(args.processed_data_dir, era5_output_folder, target_dates)
        logging.info(f"ERA5 data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during ERA5 retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 6. Run Sentinel-2 Data Retrieval ---
    logging.info("="*50)
    logging.info(f"Starting Sentinel-2 data retrieval for ROI '{roi_name}'...")
    try:
        s2_output_folder = os.path.join(args.output_folder, roi_name, "s2_images")
        main_s2_retrieval_offline(args.processed_data_dir, s2_output_folder, target_dates)
        logging.info(f"Sentinel-2 data retrieval finished for ROI '{roi_name}'.")
    except Exception as e:
        logging.critical(f"An error occurred during Sentinel-2 retrieval for '{roi_name}': {e}", exc_info=True)

    # --- 7. Test Mode Only Retrievals (if needed) ---
    if args.mode == 'test':
        logging.info("Test mode selected, but additional datasets (MODIS, S3 SLSTR) are not available in offline mode.")
        logging.info("These datasets would need to be processed separately if required.")

    overall_end_time = time.time()
    logging.info("="*50)
    logging.info(f"Data retrieval orchestration finished. Total time: {timedelta(seconds=overall_end_time - overall_start_time)}")

def main_shp():
    """Legacy main_shp function for backward compatibility."""
    logging.warning("Using legacy main_shp function. Consider using main_shp_offline() for processed data.")
    # This would need to be updated to work with the new offline approach
    # For now, just log a warning
    logging.info("Legacy function called - no action taken")


if __name__ == '__main__':
    main_offline() 