import numpy as np
from typing import Dict, Any

def compute_fvc(image_data: Dict[str, Any], landsat: str) -> Dict[str, Any]:
    """
    Computes FVC (fraction of vegetation cover) from the NDVI of the image.

    Parameters:
      image_data (dict): Dictionary containing image arrays and metadata
      landsat (str): Landsat satellite id (e.g., 'L4', 'L5', 'L7', 'L8').
                     (Currently not used in the calculation.)
    
    Returns:
      dict: Updated image data with FVC band added
    """
    if 'NDVI' not in image_data:
        print("Warning: NDVI band not found, cannot compute FVC")
        return image_data
    
    ndvi = image_data['NDVI']
    ndvi_bg = 0.2
    ndvi_vg = 0.86
    
    # Compute FVC using the provided expression: ((ndvi - ndvi_bg) / (ndvi_vg - ndvi_bg))**2
    fvc = ((ndvi - ndvi_bg) / (ndvi_vg - ndvi_bg)) ** 2
    
    # Clamp FVC values between 0 and 1
    fvc = np.clip(fvc, 0.0, 1.0)
    
    # Add FVC to the image data
    result = image_data.copy()
    result['FVC'] = fvc
    
    return result

# Example usage:
# fvc_func = addBand('L8')
# image_with_fvc = fvc_func(your_image)
