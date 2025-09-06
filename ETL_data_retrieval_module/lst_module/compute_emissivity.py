import numpy as np
from typing import Dict, Any, Optional

# Import the module with ASTER GED bare emissivity functions.
# It should provide functions: emiss_bare_band13(aster_data, fvc) and emiss_bare_band14(aster_data, fvc)
from . import ASTER_bare_emiss as ASTERGED

def compute_emissivity(image_data: Dict[str, Any], landsat: str, use_ndvi: bool, 
                      processed_data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes the surface emissivity for a Landsat image using ASTER GED and FVC.
    
    Args:
      image_data (dict): Dictionary containing image arrays and metadata
      landsat (str): Landsat satellite id ('L4', 'L5', 'L7', 'L8', or 'L9').
      use_ndvi (bool): If True, apply dynamic emissivity (with NDVI-based vegetation correction);
                       if False, use emissivity derived directly from ASTER.
      processed_data_dir (str, optional): Path to processed data directory for ASTER data
    
    Returns:
      dict: Updated image data with EM band added
    """
    # Define coefficients based on the Landsat satellite
    if landsat == 'L4':
        c13 = 0.3222
        c14 = 0.6498
        c = 0.0272
    elif landsat == 'L5':
        c13 = -0.0723
        c14 = 1.0521
        c = 0.0195
    elif landsat == 'L7':
        c13 = 0.2147
        c14 = 0.7789
        c = 0.0059
    else:  # For L8, L9, etc.
        c13 = 0.6820
        c14 = 0.2578
        c = 0.0584

    if use_ndvi:
        # Dynamic emissivity using vegetation cover correction
        if 'FVC' not in image_data:
            print("Warning: FVC band not found, cannot compute dynamic emissivity")
            return image_data
        
        if processed_data_dir is None:
            print("Warning: processed_data_dir required for dynamic emissivity computation")
            return image_data
        
        # Load ASTER data
        aster_data = ASTERGED.load_aster_data(processed_data_dir)
        if aster_data is None:
            print("Warning: Could not load ASTER data for emissivity computation")
            return image_data
        
        # Compute FVC from ASTER data
        fvc_aster = ASTERGED.compute_aster_fvc(aster_data)
        
        # Compute ASTER-based bare emissivity
        em_bare_13 = ASTERGED.emiss_bare_band13(aster_data, fvc_aster)
        em_bare_14 = ASTERGED.emiss_bare_band14(aster_data, fvc_aster)
        emiss_bare = c13 * em_bare_13 + c14 * em_bare_14 + c
        
        # Compute dynamic emissivity: fvc * 0.99 + (1 - fvc) * em_bare
        fvc = image_data['FVC']
        EM = fvc * 0.99 + (1 - fvc) * emiss_bare
        
    else:
        # Use emissivity directly from ASTER without vegetation correction
        if processed_data_dir is None:
            print("Warning: processed_data_dir required for ASTER emissivity computation")
            return image_data
        
        # Load ASTER data
        aster_data = ASTERGED.load_aster_data(processed_data_dir)
        if aster_data is None:
            print("Warning: Could not load ASTER data for emissivity computation")
            return image_data
        
        # Use original ASTER emissivity (already scaled by 0.001 in load_aster_data)
        em13 = aster_data['emissivity_band13']
        em14 = aster_data['emissivity_band14']
        EM = c13 * em13 + c14 * em14 + c
    
    # Prescribe emissivity values for water bodies and snow/ice
    if 'QA_PIXEL' in image_data:
        qa = image_data['QA_PIXEL']
        # Bit 7: Water, Bit 5: Snow/Ice
        water_mask = (qa & (1 << 7)) != 0
        snow_mask = (qa & (1 << 5)) != 0
        
        EM[water_mask] = 0.99
        EM[snow_mask] = 0.989
    
    # Add EM to the image data
    result = image_data.copy()
    result['EM'] = EM
    
    return result

# Example usage:
# landsat_id = 'L8'
# use_ndvi = True
# em_func = addBand(landsat_id, use_ndvi)
# image_with_em = em_func(ee.Image("YOUR_IMAGE_ID_HERE"))
