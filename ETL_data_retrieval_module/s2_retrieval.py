import os
import shutil
import time
import json
import logging
import rasterio
import glob
import numpy as np
from datetime import datetime
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ndvi_retrieval.log')
    ]
)

def write_metadata_to_tiff(tif_path, timestamp_ms=None, acquisition_type=None):
    """Writes timestamp metadata to a GeoTIFF file."""
    if os.path.exists(tif_path):
        try:
            with rasterio.open(tif_path, 'r+') as dst:
                tags = {}
                if timestamp_ms:
                    dt_object = datetime.fromtimestamp(timestamp_ms / 1000)
                    datetime_str = dt_object.strftime('%Y:%m:%d %H:%M:%S')
                    tags['DATETIME'] = datetime_str
                    print(f"  > Wrote DATETIME: {datetime_str}")
                
                if acquisition_type:
                    tags['ACQUISITION_TYPE'] = acquisition_type
                    print(f"  > Wrote ACQUISITION_TYPE: {acquisition_type}")
                
                if tags:
                    dst.update_tags(**tags)

        except Exception as e:
            print(f"Warning: Failed to write metadata to {tif_path}: {e}")

# =============================================================================
# FUNCTIONS
# =============================================================================
def load_processed_mapping(processed_data_dir: str) -> Optional[dict]:
    """
    Load the processed_mapping.json file to get information about processed S2 files.
    
    Args:
        processed_data_dir: Path to processed data directory
    
    Returns:
        Dictionary containing processed file mapping, or None if not found
    """
    mapping_file = os.path.join(processed_data_dir, "processed_mapping.json")
    if not os.path.exists(mapping_file):
        logging.warning(f"Processed mapping file not found: {mapping_file}")
        return None
    
    try:
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        logging.info(f"Successfully loaded processed mapping from {mapping_file}")
        return mapping_data
    except Exception as e:
        logging.error(f"Error loading processed mapping: {e}")
        return None
def get_s2_files_for_date(processed_data_dir: str, target_date: datetime) -> List[str]:
    """
    Find Sentinel-2 files for a specific date in the processed data directory.
    
    Args:
        processed_data_dir: Path to processed data directory
        target_date: Target date to find files for
    
    Returns:
        List of file paths for the target date
    """
    s2_folder = os.path.join(processed_data_dir, "sentinel2")
    if not os.path.exists(s2_folder):
        logging.warning(f"Sentinel-2 folder not found: {s2_folder}")
        return []
    
    # Look for files with the target date
    date_str = target_date.strftime('%Y%m%d')
    pattern = os.path.join(s2_folder, f"*{date_str}*.tif")
    files = glob.glob(pattern)
    
    if not files:
        logging.info(f"No Sentinel-2 files found for date {target_date.strftime('%Y-%m-%d')}")
        return []
    
    logging.info(f"Found {len(files)} Sentinel-2 files for date {target_date.strftime('%Y-%m-%d')}")
    return files


def copy_s2_files_offline(processed_data_dir: str, output_folder: str, target_dates: List[datetime]):
    """
    Copies Sentinel-2 files from processed data to the output folder.
    Renames files to match the expected naming convention.
    
    Args:
        processed_data_dir: Path to processed data directory
        output_folder: Path to output folder
        target_dates: List of target dates to process
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder: {output_folder}")

    for target_date in target_dates:
        date_str = target_date.strftime('%Y-%m-%d')
        final_tif_path = os.path.join(output_folder, f"S2_8days_{date_str}.tif")

        if os.path.exists(final_tif_path):
            logging.info(f"Skipping S2 copy for {date_str}: file already exists.")
            continue

        # Find S2 files for this date
        s2_files = get_s2_files_for_date(processed_data_dir, target_date)
        
        if not s2_files:
            logging.warning(f"No S2 files found for date {date_str}")
            continue

        # Use the first available file (prefer multi-band files)
        source_file = s2_files[0]
        
        try:
            # Copy the file
            shutil.copy2(source_file, final_tif_path)
            
            # Write metadata to the copied file
            timestamp_ms = int(target_date.replace(hour=10, minute=30, second=0).timestamp() * 1000)
            write_metadata_to_tiff(final_tif_path, timestamp_ms, 'S2_8days_Processed')
            
            logging.info(f"Successfully copied S2 image for {date_str}")
            
        except Exception as e:
            logging.error(f"Failed to copy S2 file for {date_str}: {e}")


def main_s2_retrieval_offline(processed_data_dir: str, output_folder: str, target_dates: List[datetime]):
    """
    Main function to orchestrate S2 RGB-NIR retrieval from processed temp data.
    
    Args:
        processed_data_dir: Path to processed data directory
        output_folder: Path to output folder
        target_dates: List of target dates to process
    """
    logging.info(f"--- Starting S2 retrieval process for {len(target_dates)} specific dates ---")
    
    if not target_dates:
        logging.warning("No target dates provided. Skipping S2 retrieval.")
        return

    # Copy S2 files from processed data
    copy_s2_files_offline(processed_data_dir, output_folder, target_dates)
    
    logging.info(f"--- S2 retrieval process finished ---")

def main_s2_retrieval(target_composite_dates: list, roi, roi_name, big_folder):
    """
    Legacy function for backward compatibility.
    """
    logging.warning("Using legacy main_s2_retrieval function. Consider using main_s2_retrieval_offline() for processed data.")
    # This would need to be updated to work with the new offline approach
    # For now, just log a warning
    logging.info("Legacy function called - no action taken")


if __name__ == '__main__':
    # =========================================================================
    # EXAMPLE USAGE
    # =========================================================================
    # This block demonstrates how to run the S2 retrieval script with processed data.
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve Sentinel-2 data from processed temp data.")
    parser.add_argument("--processed_data_dir", required=True, help="Path to processed temp data directory (e.g., temp_processed_data/D-49-49-A).")
    parser.add_argument("--output_folder", required=True, help="Folder where the S2 data will be saved.")
    parser.add_argument("--specific_dates", nargs='*', help="Specific dates to process (format: YYYY-MM-DD). If not provided, dates will be derived from processed data.")
    
    args = parser.parse_args()
    
    try:
        # Convert date strings to datetime objects
        if args.specific_dates:
            target_dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in args.specific_dates]
        else:
            # Try to get dates from processed mapping or use default
            mapping_data = load_processed_mapping(args.processed_data_dir)
            if mapping_data and 'sentinel2' in mapping_data:
                # Extract dates from processed mapping
                s2_data = mapping_data['sentinel2']
                target_dates = []
                for file_info in s2_data:
                    try:
                        date_str = file_info.get('date', '')
                        if date_str:
                            target_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    except ValueError:
                        continue
            else:
                logging.warning("No specific dates provided and could not derive dates from processed mapping.")
                target_dates = []

        if not target_dates:
            logging.warning("No target dates available. Exiting.")
            exit(1)

        # Run the offline S2 retrieval
        main_s2_retrieval_offline(args.processed_data_dir, args.output_folder, target_dates)
        
    except Exception as e:
        logging.critical(f"An error occurred during the example run: {e}", exc_info=True)

