import numpy as np
import rasterio
from typing import Optional, Dict, Any

# Import the ASTER GED bare emissivity module.
# It should provide functions: emiss_bare_band10, emiss_bare_band11, emiss_bare_band12,
# emiss_bare_band13, and emiss_bare_band14.
from . import ASTER_bare_emiss  

def compute_broadband_emissivity(processed_data_dir: str, dynamic: bool = True, fvc_array: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """
    Computes broad-band emissivity (BBE) from ASTER GED data in processed temp directory.
    If dynamic is True, vegetation cover correction is applied; otherwise, the original 
    ASTER GED emissivity is used.
    
    Parameters:
      processed_data_dir (str): Path to the processed data directory
      dynamic (bool): True to use vegetation cover correction, False to use original emissivity
      fvc_array (np.ndarray, optional): FVC array for dynamic correction. If None, will be computed from ASTER data
    
    Returns:
      Dictionary containing BBE array and metadata, or None if computation fails
    """
    # Load ASTER data
    aster_data = ASTER_bare_emiss.load_aster_data(processed_data_dir)
    if aster_data is None:
        print("Error: Could not load ASTER data")
        return None
    
    # Get FVC array
    if fvc_array is None:
        fvc_array = ASTER_bare_emiss.compute_aster_fvc(aster_data)
    
    # Process each emissivity band
    emissivity_bands = {}
    
    for band_num in [10, 11, 12, 13, 14]:
        band_key = f'emissivity_band{band_num}'
        
        if band_key not in aster_data:
            print(f"Error: {band_key} not found in ASTER data")
            return None
        
        # Original emissivity (already scaled by 0.001 in load_aster_data)
        orig_em = aster_data[band_key]
        
        if dynamic:
            # Dynamic emissivity: fvc*0.99 + (1-fvc)*em_bare
            bare_em_func = getattr(ASTER_bare_emiss, f'emiss_bare_band{band_num}')
            em_bare = bare_em_func(aster_data, fvc_array)
            dynamic_em = fvc_array * 0.99 + (1 - fvc_array) * em_bare
            emissivity_bands[f'em{band_num}'] = dynamic_em
        else:
            # Use original emissivity
            emissivity_bands[f'em{band_num}'] = orig_em
    
    # Compute broad-band emissivity using linear combination
    # Formula: 0.128 + 0.014*em10 + 0.145*em11 + 0.241*em12 + 0.467*em13 + 0.004*em14
    bbe = (0.128 + 
           0.014 * emissivity_bands['em10'] + 
           0.145 * emissivity_bands['em11'] + 
           0.241 * emissivity_bands['em12'] + 
           0.467 * emissivity_bands['em13'] + 
           0.004 * emissivity_bands['em14'])
    
    result = {
        'BBE': bbe,
        'emissivity_bands': emissivity_bands,
        'fvc': fvc_array,
        'profile': aster_data['profile'],
        'transform': aster_data['transform'],
        'crs': aster_data['crs'],
        'dynamic': dynamic
    }
    
    return result

def save_bbe_to_geotiff(bbe_result: Dict[str, Any], output_path: str) -> bool:
    """
    Save the computed BBE array to a GeoTIFF file.
    
    Parameters:
      bbe_result (dict): Result from compute_broadband_emissivity
      output_path (str): Path where to save the GeoTIFF file
    
    Returns:
      bool: True if successful, False otherwise
    """
    try:
        # Update profile for single band output
        profile = bbe_result['profile'].copy()
        profile.update(count=1, dtype='float32')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(bbe_result['BBE'].astype(np.float32), 1)
            
        print(f"Successfully saved BBE to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error saving BBE to {output_path}: {e}")
        return False

# Example usage:
# processed_data_dir = "temp_processed_data/D-49-49-A"
# dynamic = True  # Use vegetation cover correction
# 
# # Compute broadband emissivity
# bbe_result = compute_broadband_emissivity(processed_data_dir, dynamic=dynamic)
# 
# if bbe_result:
#     # Access the BBE array
#     bbe_array = bbe_result['BBE']
#     print(f"BBE shape: {bbe_array.shape}")
#     print(f"BBE range: {bbe_array.min():.3f} - {bbe_array.max():.3f}")
#     
#     # Save to GeoTIFF
#     save_bbe_to_geotiff(bbe_result, "output_bbe.tif")
