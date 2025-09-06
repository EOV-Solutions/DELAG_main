import os
import glob
import numpy as np
import rasterio
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import the offline modules
from . import NCEP_TPW
from . import cloudmask
from . import compute_NDVI as NDVI
from . import compute_FVC as FVC
from . import compute_emissivity as EM
from . import SMWalgorithm as LST

# Define the dictionary for Landsat band configurations
COLLECTION = {
    'L4': {
        'TIR': ['B6'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    },
    'L5': {
        'TIR': ['B6'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    },
    'L7': {
        'TIR': ['B6_VCID_1', 'B6_VCID_2'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'QA_PIXEL']
    },
    'L8': {
        'TIR': ['B10', 'B11'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    },
    'L9': {
        'TIR': ['B10', 'B11'],
        'VISW': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL']
    }
}

def load_landsat_data(processed_data_dir: str, landsat: str, target_date: datetime) -> Optional[Dict[str, Any]]:
    """
    Load Landsat data from processed directory for a specific date and satellite.
    
    Args:
      processed_data_dir (str): Path to processed data directory
      landsat (str): Landsat satellite id ('L4', 'L5', 'L7', 'L8', or 'L9')
      target_date (datetime): Target date for data loading
    
    Returns:
      dict: Dictionary containing image arrays and metadata, or None if not found
    """
    date_str = target_date.strftime('%Y%m%d')
    
    # Look for Landsat L1 and L2 files
    landsat_l1_folder = os.path.join(processed_data_dir, "landsat_l1")
    landsat_l2_folder = os.path.join(processed_data_dir, "landsat_l2")
    
    # Try L2 first, then L1
    for folder in [landsat_l2_folder, landsat_l1_folder]:
        if os.path.exists(folder):
            pattern = os.path.join(folder, f"{landsat}_L*_{date_str}_*.tif")
            files = glob.glob(pattern)
            if files:
                # Load the first matching file
                file_path = files[0]
                try:
                    with rasterio.open(file_path) as src:
                        bands = src.read()
                        profile = src.profile
                        
                        # Get band names from the collection configuration
                        collection_dict = COLLECTION.get(landsat, {})
                        visw_bands = collection_dict.get('VISW', [])
                        tir_bands = collection_dict.get('TIR', [])
                        
                        # Create image data dictionary
                        image_data = {
                            'profile': profile,
                            'transform': src.transform,
                            'crs': src.crs
                        }
                        
                        # Map bands to their names (assuming standard order)
                        band_mapping = {
                            0: 'SR_B1', 1: 'SR_B2', 2: 'SR_B3', 3: 'SR_B4', 
                            4: 'SR_B5', 5: 'SR_B6', 6: 'SR_B7', 7: 'QA_PIXEL'
                        }
                        
                        for i, band_array in enumerate(bands):
                            if i in band_mapping:
                                image_data[band_mapping[i]] = band_array
                        
                        return image_data
                        
                except Exception as e:
                    print(f"Error loading Landsat file {file_path}: {e}")
                    continue
    
    return None

def process_landsat_lst(processed_data_dir: str, landsat: str, target_date: datetime, 
                       use_ndvi: bool = True) -> Optional[Dict[str, Any]]:
    """
    Processes Landsat data for LST computation using offline modules.
    
    Args:
      processed_data_dir (str): Path to processed data directory
      landsat (str): Landsat satellite id ('L4', 'L5', 'L7', 'L8', or 'L9')
      target_date (datetime): Target date for processing
      use_ndvi (bool): If True, use NDVI-based dynamic emissivity
    
    Returns:
      dict: Processed image data with all required bands for LST computation
    """
    # Load Landsat data
    image_data = load_landsat_data(processed_data_dir, landsat, target_date)
    if image_data is None:
        print(f"No Landsat data found for {landsat} on {target_date.strftime('%Y-%m-%d')}")
        return None
    
    # Process the image through the pipeline
    try:
        # 1. Compute NDVI
        image_data = NDVI.compute_ndvi(image_data, landsat)
        
        # 2. Apply cloud mask
        image_data = cloudmask.apply_cloud_mask_sr(image_data)
        
        # 3. Compute FVC
        image_data = FVC.compute_fvc(image_data, landsat)
        
        # 4. Add TPW
        image_data = NCEP_TPW.add_tpw_offline(image_data, processed_data_dir, target_date)
        
        # 5. Compute emissivity
        image_data = EM.compute_emissivity(image_data, landsat, use_ndvi, processed_data_dir)
        
        # 6. Compute LST
        image_data = LST.compute_lst_offline(image_data, landsat)
        
        return image_data
        
    except Exception as e:
        print(f"Error processing Landsat data: {e}")
        return None

def process_landsat_collection(processed_data_dir: str, landsat: str, 
                              date_start: str, date_end: str, use_ndvi: bool = True) -> List[Dict[str, Any]]:
    """
    Processes a collection of Landsat data for the specified date range.
    
    Args:
      processed_data_dir (str): Path to processed data directory
      landsat (str): Landsat satellite id ('L4', 'L5', 'L7', 'L8', or 'L9')
      date_start (str): Start date in 'YYYY-MM-DD' format
      date_end (str): End date in 'YYYY-MM-DD' format
      use_ndvi (bool): If True, use NDVI-based dynamic emissivity
    
    Returns:
      list: List of processed image data dictionaries
    """
    start_date = datetime.strptime(date_start, '%Y-%m-%d')
    end_date = datetime.strptime(date_end, '%Y-%m-%d')
    
    processed_images = []
    current_date = start_date
    
    while current_date <= end_date:
        image_data = process_landsat_lst(processed_data_dir, landsat, current_date, use_ndvi)
        if image_data is not None:
            processed_images.append(image_data)
        
        # Move to next day (Landsat has ~16-day revisit cycle, but we check daily)
        current_date = current_date.replace(day=current_date.day + 1)
        if current_date.day > 28:  # Simple month handling
            try:
                current_date = current_date.replace(month=current_date.month + 1, day=1)
            except ValueError:
                current_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
    
    return processed_images

# Example usage:
# processed_data_dir = "temp_processed_data/D-49-49-A"
# landsat_id = 'L8'
# date_start = '2023-01-01'
# date_end = '2023-01-31'
# use_ndvi = True
#
# # Process a single date
# target_date = datetime(2023, 1, 15)
# image_data = process_landsat_lst(processed_data_dir, landsat_id, target_date, use_ndvi)
# if image_data:
#     print(f"Processed image shape: {image_data['SR_B1'].shape}")
#     print(f"Available bands: {list(image_data.keys())}")
#
# # Process a collection of dates
# processed_images = process_landsat_collection(processed_data_dir, landsat_id, date_start, date_end, use_ndvi)
# print(f"Processed {len(processed_images)} images")
