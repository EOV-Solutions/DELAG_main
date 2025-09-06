import numpy as np
from typing import Dict, Any

def apply_cloud_mask_toa(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies a cloud mask to TOA data using the QA_PIXEL band.
    
    Parameters:
      image_data (dict): Dictionary containing image arrays and metadata
    
    Returns:
      dict: Updated image data with cloud mask applied
    """
    if 'QA_PIXEL' not in image_data:
        print("Warning: QA_PIXEL band not found, skipping cloud masking")
        return image_data
    
    qa = image_data['QA_PIXEL']
    # Bit 3: Cloud
    cloud_mask = (qa & (1 << 3)) != 0
    
    # Apply mask to all bands (set cloudy pixels to NaN)
    result = image_data.copy()
    for band_name, band_data in image_data.items():
        if isinstance(band_data, np.ndarray) and band_name != 'QA_PIXEL':
            masked_data = band_data.copy()
            masked_data[cloud_mask] = np.nan
            result[band_name] = masked_data
    
    return result

def apply_cloud_mask_sr(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies a cloud and cloud shadow mask to Surface Reflectance (SR) data using the QA_PIXEL band.
    
    Parameters:
      image_data (dict): Dictionary containing image arrays and metadata
    
    Returns:
      dict: Updated image data with cloud and shadow mask applied
    """
    if 'QA_PIXEL' not in image_data:
        print("Warning: QA_PIXEL band not found, skipping cloud masking")
        return image_data
    
    qa = image_data['QA_PIXEL']
    # Bit 3: Cloud, Bit 4: Cloud Shadow
    cloud_mask = (qa & (1 << 3)) != 0
    shadow_mask = (qa & (1 << 4)) != 0
    combined_mask = cloud_mask | shadow_mask
    
    # Apply mask to all bands (set cloudy/shadow pixels to NaN)
    result = image_data.copy()
    for band_name, band_data in image_data.items():
        if isinstance(band_data, np.ndarray) and band_name != 'QA_PIXEL':
            masked_data = band_data.copy()
            masked_data[combined_mask] = np.nan
            result[band_name] = masked_data
    
    return result
