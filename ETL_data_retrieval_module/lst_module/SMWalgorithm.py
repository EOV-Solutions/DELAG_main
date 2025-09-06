import numpy as np
from typing import Dict, Any
from .SMW_coefficients import get_coefficients, get_coefficient_by_tpwpos

def compute_lst_offline(image_data: Dict[str, Any], landsat: str) -> Dict[str, Any]:
    """
    Computes LST (Land Surface Temperature) using the Statistical Mono-Window algorithm.
    
    Args:
      image_data (dict): Dictionary containing image arrays and metadata
      landsat (str): Landsat satellite id ('L4', 'L5', 'L7', 'L8', or 'L9')
    
    Returns:
      dict: Updated image data with LST band added
    """
    # Check if required bands exist
    required_bands = ['EM', 'TPW', 'TPWpos']
    for band in required_bands:
        if band not in image_data:
            print(f"Warning: Required band {band} not found for LST computation")
            return image_data
    
    # Select the TIR band based on Landsat satellite
    if landsat in ['L9', 'L8']:
        tir_band = 'B10'
    elif landsat == 'L7':
        tir_band = 'B6_VCID_1'
    else:
        tir_band = 'B6'
    
    if tir_band not in image_data:
        print(f"Warning: TIR band {tir_band} not found for Landsat {landsat}")
        return image_data
    
    # Get the coefficient arrays
    coeffs = get_coefficients(landsat)
    
    # Create coefficient arrays for each TPW position
    A_values = np.array([coeff['A'] for coeff in coeffs])
    B_values = np.array([coeff['B'] for coeff in coeffs])
    C_values = np.array([coeff['C'] for coeff in coeffs])
    
    # Get the input arrays
    em = image_data['EM']
    tpw = image_data['TPW']
    tpwpos = image_data['TPWpos']
    tir = image_data[tir_band]
    
    # Initialize LST array
    lst = np.zeros_like(em, dtype=np.float32)
    
    # Apply the SMW algorithm for each TPW position
    for tpw_idx in range(10):  # TPW positions 0-9
        mask = (tpwpos == tpw_idx) & (tpw >= 0)  # Valid TPW values
        
        if np.any(mask):
            A = A_values[tpw_idx]
            B = B_values[tpw_idx]
            C = C_values[tpw_idx]
            
            # SMW formula: LST = A * Tb1 / em1 + B / em1 + C
            # Handle division by zero in emissivity
            em_safe = np.where(em[mask] > 0, em[mask], 1.0)
            lst[mask] = A * tir[mask] / em_safe + B / em_safe + C
    
    # Set invalid pixels to NaN
    lst[tpw < 0] = np.nan
    lst[em <= 0] = np.nan
    
    # Add LST to the image data
    result = image_data.copy()
    result['LST'] = lst
    
    return result

def compute_lst_with_coefficients(image_data: Dict[str, Any], landsat: str, 
                                 tpwpos: int) -> Dict[str, Any]:
    """
    Computes LST using specific SMW coefficients for a given TPW position.
    
    Args:
      image_data (dict): Dictionary containing image arrays and metadata
      landsat (str): Landsat satellite id ('L4', 'L5', 'L7', 'L8', or 'L9')
      tpwpos (int): TPW position index (0-9)
    
    Returns:
      dict: Updated image data with LST band added
    """
    # Get coefficients for the specific TPW position
    try:
        coeff = get_coefficient_by_tpwpos(landsat, tpwpos)
    except ValueError as e:
        print(f"Error: {e}")
        return image_data
    
    # Select the TIR band
    if landsat in ['L9', 'L8']:
        tir_band = 'B10'
    elif landsat == 'L7':
        tir_band = 'B6_VCID_1'
    else:
        tir_band = 'B6'
    
    if tir_band not in image_data or 'EM' not in image_data:
        print(f"Warning: Required bands not found for LST computation")
        return image_data
    
    # Get the input arrays
    em = image_data['EM']
    tir = image_data[tir_band]
    
    # Apply the SMW formula: LST = A * Tb1 / em1 + B / em1 + C
    A, B, C = coeff['A'], coeff['B'], coeff['C']
    
    # Handle division by zero in emissivity
    em_safe = np.where(em > 0, em, 1.0)
    lst = A * tir / em_safe + B / em_safe + C
    
    # Set invalid pixels to NaN
    lst[em <= 0] = np.nan
    
    # Add LST to the image data
    result = image_data.copy()
    result['LST'] = lst
    
    return result

# Example usage:
# from ETL_data_retrieval_module.lst_module.SMWalgorithm import compute_lst_offline
# 
# # Assuming you have image_data with required bands (EM, TPW, TPWpos, B10)
# landsat_id = 'L8'
# image_data_with_lst = compute_lst_offline(image_data, landsat_id)
# 
# if 'LST' in image_data_with_lst:
#     print(f"LST computed successfully")
#     print(f"LST range: {image_data_with_lst['LST'].min():.2f} - {image_data_with_lst['LST'].max():.2f} K")
