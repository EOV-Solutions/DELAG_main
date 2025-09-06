import numpy as np
from typing import Dict, Any

def compute_ndvi(image_data: Dict[str, Any], landsat: str) -> Dict[str, Any]:
    """
    Computes NDVI for a given Landsat image and adds it as a new band 'NDVI'.
    
    Parameters:
      image_data (dict): Dictionary containing image arrays and metadata
      landsat (str): Landsat satellite id (e.g., 'L4', 'L5', 'L7', 'L8', or 'L9')
      
    Returns:
      dict: Updated image data with NDVI band added
    """
    # Choose bands based on the Landsat satellite
    if landsat in ['L8', 'L9']:
        nir_band = 'SR_B5'
        red_band = 'SR_B4'
    else:
        nir_band = 'SR_B4'
        red_band = 'SR_B3'
    
    # Check if required bands exist
    if nir_band not in image_data or red_band not in image_data:
        print(f"Warning: Required bands {nir_band} or {red_band} not found for NDVI computation")
        return image_data
    
    # Compute scaled reflectances for NIR and red bands
    # Formula: band * 0.0000275 - 0.2
    nir_scaled = image_data[nir_band] * 0.0000275 - 0.2
    red_scaled = image_data[red_band] * 0.0000275 - 0.2
    
    # Compute NDVI using the standard formula: (NIR - Red) / (NIR + Red)
    # Handle division by zero
    denominator = nir_scaled + red_scaled
    ndvi = np.divide(nir_scaled - red_scaled, denominator, 
                     out=np.full_like(nir_scaled, np.nan), 
                     where=denominator != 0)
    
    # Add NDVI to the image data
    result = image_data.copy()
    result['NDVI'] = ndvi
    
    return result

# Example usage:
# ndvi_func = addBand('L8')
# image_with_ndvi = ndvi_func(ee.Image("LANDSAT/LC08/C02/T1_L2/LC08_044034_20170606"))
