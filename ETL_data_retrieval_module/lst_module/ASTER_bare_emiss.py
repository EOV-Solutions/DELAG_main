import os
import glob
import numpy as np
import rasterio
from typing import Optional, Tuple

def load_aster_data(processed_data_dir: str) -> Optional[dict]:
    """
    Load ASTER emissivity and NDVI data from processed temp data directory.
    
    Args:
        processed_data_dir: Path to the processed data directory (e.g., temp_processed_data/D-49-49-A)
    
    Returns:
        Dictionary containing ASTER data arrays and metadata, or None if not found
    """
    aster_folder = os.path.join(processed_data_dir, "aster")
    if not os.path.exists(aster_folder):
        print(f"Warning: ASTER folder not found: {aster_folder}")
        return None
    
    # Look for ASTER files
    aster_files = glob.glob(os.path.join(aster_folder, "*.tif"))
    if not aster_files:
        print(f"Warning: No ASTER files found in: {aster_folder}")
        return None
    
    aster_data = {}
    
    # Load the main ASTER file (should contain all bands)
    main_aster_file = None
    for file_path in aster_files:
        if "emissivity" in os.path.basename(file_path).lower():
            main_aster_file = file_path
            break
    
    if not main_aster_file:
        print(f"Warning: No ASTER emissivity file found in: {aster_folder}")
        return None
    
    try:
        with rasterio.open(main_aster_file) as src:
            # Read all bands
            bands = src.read()
            profile = src.profile
            
            # Assuming the bands are in order: emissivity_band10, emissivity_band11, 
            # emissivity_band12, emissivity_band13, emissivity_band14, ndvi
            if bands.shape[0] >= 6:
                aster_data['emissivity_band10'] = bands[0] * 0.001  # Convert to proper scale
                aster_data['emissivity_band11'] = bands[1] * 0.001
                aster_data['emissivity_band12'] = bands[2] * 0.001
                aster_data['emissivity_band13'] = bands[3] * 0.001
                aster_data['emissivity_band14'] = bands[4] * 0.001
                aster_data['ndvi'] = bands[5] * 0.01  # Convert to proper scale
            else:
                print(f"Warning: ASTER file has {bands.shape[0]} bands, expected at least 6")
                return None
            
            aster_data['profile'] = profile
            aster_data['transform'] = src.transform
            aster_data['crs'] = src.crs
            
    except Exception as e:
        print(f"Error loading ASTER data: {e}")
        return None
    
    return aster_data

def compute_aster_fvc(aster_data: dict) -> np.ndarray:
    """
    Compute ASTER FVC from NDVI.
    
    Args:
        aster_data: Dictionary containing ASTER data arrays
    
    Returns:
        FVC array
    """
    if 'ndvi' not in aster_data:
        raise ValueError("NDVI data not found in aster_data")
    
    ndvi = aster_data['ndvi']
    ndvi_bg = 0.2
    ndvi_vg = 0.86
    
    # Compute FVC: ((ndvi - ndvi_bg)/(ndvi_vg - ndvi_bg))**2
    fvc = ((ndvi - ndvi_bg) / (ndvi_vg - ndvi_bg)) ** 2
    
    # Clamp FVC values to the [0, 1] range
    fvc = np.clip(fvc, 0.0, 1.0)
    
    return fvc

def emiss_bare_band10(aster_data: dict, fvc: np.ndarray) -> np.ndarray:
    """
    Compute bare ground emissivity for band 10.
    
    Args:
        aster_data: Dictionary containing ASTER data arrays
        fvc: Fractional vegetation cover array
    
    Returns:
        Bare ground emissivity array for band 10
    """
    if 'emissivity_band10' not in aster_data:
        raise ValueError("emissivity_band10 not found in aster_data")
    
    EM = aster_data['emissivity_band10']
    
    # Formula: (EM - 0.99*fvc)/(1.0 - fvc)
    # Handle division by zero where fvc = 1.0
    denominator = 1.0 - fvc
    denominator = np.where(denominator == 0, 1e-10, denominator)  # Avoid division by zero
    
    result = (EM - 0.99 * fvc) / denominator
    return result

def emiss_bare_band11(aster_data: dict, fvc: np.ndarray) -> np.ndarray:
    """
    Compute bare ground emissivity for band 11.
    
    Args:
        aster_data: Dictionary containing ASTER data arrays
        fvc: Fractional vegetation cover array
    
    Returns:
        Bare ground emissivity array for band 11
    """
    if 'emissivity_band11' not in aster_data:
        raise ValueError("emissivity_band11 not found in aster_data")
    
    EM = aster_data['emissivity_band11']
    
    # Formula: (EM - 0.99*fvc)/(1.0 - fvc)
    denominator = 1.0 - fvc
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    result = (EM - 0.99 * fvc) / denominator
    return result

def emiss_bare_band12(aster_data: dict, fvc: np.ndarray) -> np.ndarray:
    """
    Compute bare ground emissivity for band 12.
    
    Args:
        aster_data: Dictionary containing ASTER data arrays
        fvc: Fractional vegetation cover array
    
    Returns:
        Bare ground emissivity array for band 12
    """
    if 'emissivity_band12' not in aster_data:
        raise ValueError("emissivity_band12 not found in aster_data")
    
    EM = aster_data['emissivity_band12']
    
    # Formula: (EM - 0.99*fvc)/(1.0 - fvc)
    denominator = 1.0 - fvc
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    result = (EM - 0.99 * fvc) / denominator
    return result

def emiss_bare_band13(aster_data: dict, fvc: np.ndarray) -> np.ndarray:
    """
    Compute bare ground emissivity for band 13.
    
    Args:
        aster_data: Dictionary containing ASTER data arrays
        fvc: Fractional vegetation cover array
    
    Returns:
        Bare ground emissivity array for band 13
    """
    if 'emissivity_band13' not in aster_data:
        raise ValueError("emissivity_band13 not found in aster_data")
    
    EM = aster_data['emissivity_band13']
    
    # Formula: (EM - 0.99*fvc)/(1.0 - fvc)
    denominator = 1.0 - fvc
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    result = (EM - 0.99 * fvc) / denominator
    return result

def emiss_bare_band14(aster_data: dict, fvc: np.ndarray) -> np.ndarray:
    """
    Compute bare ground emissivity for band 14.
    
    Args:
        aster_data: Dictionary containing ASTER data arrays
        fvc: Fractional vegetation cover array
    
    Returns:
        Bare ground emissivity array for band 14
    """
    if 'emissivity_band14' not in aster_data:
        raise ValueError("emissivity_band14 not found in aster_data")
    
    EM = aster_data['emissivity_band14']
    
    # Formula: (EM - 0.99*fvc)/(1.0 - fvc)
    denominator = 1.0 - fvc
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    result = (EM - 0.99 * fvc) / denominator
    return result

def compute_all_bare_emissivity(processed_data_dir: str) -> Optional[dict]:
    """
    Compute all bare ground emissivity bands from processed ASTER data.
    
    Args:
        processed_data_dir: Path to the processed data directory
    
    Returns:
        Dictionary containing all bare ground emissivity arrays and metadata
    """
    # Load ASTER data
    aster_data = load_aster_data(processed_data_dir)
    if aster_data is None:
        return None
    
    # Compute FVC
    fvc = compute_aster_fvc(aster_data)
    
    # Compute bare ground emissivity for all bands
    result = {
        'bare_emissivity_band10': emiss_bare_band10(aster_data, fvc),
        'bare_emissivity_band11': emiss_bare_band11(aster_data, fvc),
        'bare_emissivity_band12': emiss_bare_band12(aster_data, fvc),
        'bare_emissivity_band13': emiss_bare_band13(aster_data, fvc),
        'bare_emissivity_band14': emiss_bare_band14(aster_data, fvc),
        'fvc': fvc,
        'profile': aster_data['profile'],
        'transform': aster_data['transform'],
        'crs': aster_data['crs']
    }
    
    return result

# Example usage:
# processed_data_dir = "temp_processed_data/D-49-49-A"
# result = compute_all_bare_emissivity(processed_data_dir)
# if result:
#     bare_emiss_band10 = result['bare_emissivity_band10']
#     print(f"Bare ground emissivity band 10 shape: {bare_emiss_band10.shape}")
