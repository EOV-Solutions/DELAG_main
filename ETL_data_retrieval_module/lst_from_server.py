import glob
import os
import shutil
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling

from .server_client import ServerClient
from .utils import extract_datetime_from_filename, group_files_by_date
from .config import config


def _ensure_dirs(base_output: str, roi_name: str) -> str:
    lst_dir = os.path.join(base_output, roi_name, config.get_output_dir("lst"))
    os.makedirs(lst_dir, exist_ok=True)
    return lst_dir


def _align_to_reference(src_path: str, ref_path: str, dst_path: str) -> None:
    """Align a raster to match reference grid exactly"""
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
    
    with rasterio.open(src_path) as src:
        # Create output array
        dst_array = np.full((src.count, ref_height, ref_width), src.nodata or np.nan, dtype=src.dtypes[0])
        
        # Reproject each band
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=dst_array[i-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
        
        # Update profile and save
        out_profile = ref_profile.copy()
        out_profile.update({
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata
        })
        
        with rasterio.open(dst_path, 'w', **out_profile) as dst:
            dst.write(dst_array)


def _compute_ndvi(red_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """Compute NDVI from red and NIR bands"""
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir_band - red_band) / (nir_band + red_band)
    return np.where(np.isfinite(ndvi), ndvi, np.nan)


def _compute_fvc(ndvi: np.ndarray) -> np.ndarray:
    """Compute Fractional Vegetation Cover from NDVI"""
    ndvi_bg = 0.2
    ndvi_vg = 0.86
    fvc = ((ndvi - ndvi_bg) / (ndvi_vg - ndvi_bg)) ** 2
    return np.clip(fvc, 0.0, 1.0)


def _compute_emissivity_from_aster(aster_em_band: np.ndarray, fvc: np.ndarray, use_ndvi: bool = True) -> np.ndarray:
    """
    Compute surface emissivity from ASTER GED and FVC.
    If use_ndvi=True, uses dynamic emissivity calculation, otherwise uses ASTER directly.
    """
    if use_ndvi:
        # Dynamic emissivity: EM = 0.99*FVC + bare_em*(1-FVC)
        # For simplicity, use ASTER as bare emissivity estimate
        bare_em = aster_em_band
        emissivity = 0.99 * fvc + bare_em * (1.0 - fvc)
    else:
        # Use ASTER emissivity directly
        emissivity = aster_em_band
    
    return np.clip(emissivity, 0.8, 1.0)


def _compute_lst_smw(tir_brightness_temp: np.ndarray, emissivity: np.ndarray, 
                     tpw: float = 20.0, landsat: str = "L8") -> np.ndarray:
    """
    Compute Land Surface Temperature using Statistical Mono-Window algorithm.
    Simplified version for server-based processing.
    """
    # Simplified SMW coefficients (would normally be lookup tables based on TPW)
    if landsat in ["L8", "L9"]:
        # Approximate coefficients for Landsat 8/9 Band 10
        A = 0.04
        B = 0.95
        C = 1.85
    else:
        # Approximate coefficients for Landsat 5/7 Band 6
        A = 0.06
        B = 0.90
        C = 2.10
    
    # SMW formula: LST = A * Tb / em + B / em + C
    lst = A * tir_brightness_temp / emissivity + B / emissivity + C
    
    return lst


def _process_landsat_scene_to_lst(
    l1_files: Dict[str, str],  # band -> file_path mapping for L1 (TOA)
    l2_files: Dict[str, str],  # band -> file_path mapping for L2 (SR)
    aster_files: Dict[str, str],  # band -> file_path for ASTER GED
    output_path: str,
    landsat: str = "L8",
    use_ndvi: bool = True
) -> bool:
    """
    Process a single Landsat scene through the LST calculation pipeline.
    Returns True if successful, False otherwise.
    """
    try:
        # Determine band mappings based on Landsat satellite
        if landsat in ["L8", "L9"]:
            red_band = "B4"
            nir_band = "B5" 
            tir_band = "B10"
            tir_band_l2 = "ST_B10"  # Surface temperature from L2
        else:  # L5, L7
            red_band = "B3"
            nir_band = "B4"
            tir_band = "B6"
            tir_band_l2 = "ST_B6"
        
        # Use L2 surface temperature if available, otherwise compute from L1 + ASTER
        if tir_band_l2 in l2_files:
            # Use processed surface temperature from L2
            with rasterio.open(l2_files[tir_band_l2]) as src:
                lst_data = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                
                # Convert from Kelvin to Kelvin (ensure proper scaling)
                lst_data = np.where(lst_data > 0, lst_data * 0.00341802 + 149.0, np.nan)
                
        else:
            # Compute LST from L1 TOA + ASTER emissivity
            # Read L1 TIR band (brightness temperature)
            if tir_band not in l1_files:
                print(f"Missing TIR band {tir_band} in L1 data")
                return False
                
            with rasterio.open(l1_files[tir_band]) as src:
                tir_data = src.read(1).astype(np.float32)
                profile = src.profile.copy()
                
                # Convert DN to brightness temperature (L1 specific scaling)
                tir_data = np.where(tir_data > 0, tir_data * 0.00341802 + 149.0, np.nan)
            
            # Compute NDVI and FVC if using dynamic emissivity
            if use_ndvi and red_band in l2_files and nir_band in l2_files:
                with rasterio.open(l2_files[red_band]) as red_src:
                    red_data = red_src.read(1).astype(np.float32) * 0.0000275 - 0.2  # SR scaling
                with rasterio.open(l2_files[nir_band]) as nir_src:
                    nir_data = nir_src.read(1).astype(np.float32) * 0.0000275 - 0.2
                
                ndvi = _compute_ndvi(red_data, nir_data)
                fvc = _compute_fvc(ndvi)
            else:
                fvc = np.ones_like(tir_data) * 0.5  # Default FVC
            
            # Get ASTER emissivity
            if 'emissivity_band10' in aster_files:
                with rasterio.open(aster_files['emissivity_band10']) as aster_src:
                    # Align ASTER to Landsat grid
                    temp_aster = tempfile.mktemp(suffix='.tif')
                    _align_to_reference(aster_files['emissivity_band10'], l1_files[tir_band], temp_aster)
                    
                    with rasterio.open(temp_aster) as aligned_src:
                        aster_em = aligned_src.read(1).astype(np.float32) * 0.001  # ASTER scaling
                    os.remove(temp_aster)
            else:
                aster_em = np.ones_like(tir_data) * 0.95  # Default emissivity
            
            # Compute emissivity
            emissivity = _compute_emissivity_from_aster(aster_em, fvc, use_ndvi)
            
            # Compute LST using SMW
            lst_data = _compute_lst_smw(tir_data, emissivity, landsat=landsat)
        
        # Save LST result
        profile.update(dtype='float32', count=1, nodata=np.nan)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(lst_data, 1)
            
            # Add metadata tags
            dst.update_tags(
                ACQUISITION_TYPE=f'{landsat}_LST',
                PROCESSING_LEVEL='LST_L3'
            )
        
        return True
        
    except Exception as e:
        print(f"Error processing LST for {output_path}: {e}")
        return False


def retrieve_lst_from_server(
    roi_name: str,
    task_ids: Dict[str, List[str]],
    output_base: str,
    api_base_url: str = "http://localhost:8000",
    use_ndvi: bool = True,
) -> None:
    """
    Download Landsat L1, L2, and ASTER GED data from server using predefined task IDs
    and compute LST following the original GEE-based LST module workflow.
    
    Args:
        roi_name: ROI identifier for output folder structure
        task_ids: Dictionary with task IDs for different data types:
                 {"L8_L1": [...], "L8_L2": [...], "L9_L1": [...], "L9_L2": [...], "aster": [...]}
        output_base: Base output directory
        api_base_url: Server API base URL
        use_ndvi: Whether to use NDVI for dynamic emissivity calculation
    
    Outputs: L8_lst16days_YYYY-MM-DD.tif and L9_lst16days_YYYY-MM-DD.tif files
    """
    lst_dir = _ensure_dirs(output_base, roi_name)
    client = ServerClient(api_base_url=api_base_url, timeout=300)
    
    # Download ASTER GED data once (it's static)
    print("Downloading ASTER GED data...")
    aster_files = {}
    if "aster" in task_ids and task_ids["aster"]:
        try:
            aster_task_id = task_ids["aster"][0]  # Use first ASTER task ID
            aster_temp_dir = client.download_and_extract("aster", aster_task_id)
            
            for f in glob.glob(os.path.join(aster_temp_dir, '**', '*.tif'), recursive=True):
                fname = os.path.basename(f)
                if 'emissivity_band10' in fname:
                    aster_files['emissivity_band10'] = f
                elif 'emissivity_band11' in fname:
                    aster_files['emissivity_band11'] = f
                elif 'ndvi' in fname:
                    aster_files['ndvi'] = f
            
            print(f"Downloaded ASTER GED: {list(aster_files.keys())}")
            
        except Exception as e:
            print(f"Warning: Failed to download ASTER GED: {e}")
    
    # Process each Landsat satellite
    for landsat_id in ["L8", "L9"]:
        l1_key = f"{landsat_id}_L1"
        l2_key = f"{landsat_id}_L2"
        
        if l1_key not in task_ids or l2_key not in task_ids:
            print(f"Missing task IDs for {landsat_id} (need {l1_key} and {l2_key})")
            continue
        
        l1_task_ids = task_ids[l1_key]
        l2_task_ids = task_ids[l2_key]
        
        # Process each L1 task (assuming 1:1 correspondence with L2 tasks)
        for i, l1_task_id in enumerate(l1_task_ids):
            print(f"Processing {landsat_id} LST from L1 task: {l1_task_id}")
            
            l1_temp_dir = None
            l2_temp_dir = None
            
            try:
                # Download L1 (TOA) data
                l1_temp_dir = client.download_and_extract(f"landsat{landsat_id[1].lower()}_l1", l1_task_id)
                
                # Download corresponding L2 (SR) data
                if i < len(l2_task_ids):
                    l2_task_id = l2_task_ids[i]
                    l2_temp_dir = client.download_and_extract(f"landsat{landsat_id[1].lower()}_l2", l2_task_id)
                else:
                    print(f"No corresponding L2 task for {landsat_id} L1 task {i}")
                    continue
                
                # Group files by date for both L1 and L2
                l1_tif_files = glob.glob(os.path.join(l1_temp_dir, '**', '*.tif'), recursive=True)
                l2_tif_files = glob.glob(os.path.join(l2_temp_dir, '**', '*.tif'), recursive=True) if l2_temp_dir else []
                
                l1_grouped = group_files_by_date(l1_tif_files)
                l2_grouped = group_files_by_date(l2_tif_files)
                
                # Process each date where we have data
                for date_str in l1_grouped.keys():
                    output_file = os.path.join(lst_dir, f"{landsat_id}_lst16days_{date_str}.tif")
                    
                    if os.path.exists(output_file):
                        print(f"{landsat_id} LST already exists: {output_file}")
                        continue
                    
                    # Organize files by band for this date
                    l1_files = {}
                    l2_files = {}
                    
                    # Parse L1 files
                    for f in l1_grouped[date_str]:
                        fname = os.path.basename(f)
                        if 'B10' in fname:
                            l1_files['B10'] = f
                        elif 'B11' in fname:
                            l1_files['B11'] = f
                        elif 'B6' in fname:
                            l1_files['B6'] = f
                    
                    # Parse L2 files for the same date
                    if date_str in l2_grouped:
                        for f in l2_grouped[date_str]:
                            fname = os.path.basename(f)
                            if 'SR_B4' in fname:
                                l2_files['B4'] = f
                            elif 'SR_B5' in fname:
                                l2_files['B5'] = f
                            elif 'SR_B3' in fname:
                                l2_files['B3'] = f
                            elif 'ST_B10' in fname:
                                l2_files['ST_B10'] = f
                            elif 'ST_B6' in fname:
                                l2_files['ST_B6'] = f
                    
                    # Process to LST
                    if l1_files or l2_files:
                        success = _process_landsat_scene_to_lst(
                            l1_files, l2_files, aster_files, output_file, landsat_id, use_ndvi
                        )
                        if success:
                            print(f"✅ Created {landsat_id} LST: {output_file}")
                        else:
                            print(f"❌ Failed to create {landsat_id} LST for {date_str}")
                    else:
                        print(f"❌ No {landsat_id} data found for {date_str}")
                        
            except Exception as e:
                print(f"Failed to process {landsat_id} LST from task {l1_task_id}: {e}")
            finally:
                # Cleanup temp directories
                for temp_dir in [l1_temp_dir, l2_temp_dir]:
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)