"""
Data preprocessing for the DELAG project.
"""
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling as RasterioResampling
from rasterio.windows import Window
from sklearn.impute import KNNImputer
from tqdm import tqdm
import os
import glob
import warnings

import config # Updated config with ROI structure
import utils

def load_landsat_lst(
    landsat_lst_dir: str,
    lst_nodata_val: float,
    app_config: 'config', # For OUTPUT_DIR
    reference_grid_path: str = None
) -> tuple[np.ndarray, list, dict, str]:
    """
    Loads, preprocesses, and aligns Landsat LST data for a given ROI
    ONLY for dates where LST files are found.
    Cloud mask is derived from LST_NODATA_VALUE.

    Args:
        landsat_lst_dir (str): Directory containing Landsat LST files for the ROI.
        lst_nodata_val (float): NoData value in LST files.
        app_config: Configuration object.
        reference_grid_path (str, optional): Path to a GeoTIFF defining the reference grid.
                                          If None, the first valid LST image is used as reference.

    Returns:
        tuple[np.ndarray, list, dict, str]:
            - lst_stack_nan (np.ndarray): Time-series stack of LST data (time, height, width)
                                         for dates with actual files, with np.nan for nodata pixels.
            - loaded_dates (list): List of datetime objects for which LST files were found and loaded.
            - geo_profile (dict): Georeferencing profile from the reference raster.
            - actual_reference_grid_path (str): Path of the raster used as the reference grid.
    """
    print(f"Loading and preprocessing Landsat LST data from: {landsat_lst_dir} (sparse approach)")
    lst_data_list = []
    loaded_dates = []
    actual_reference_grid_path = reference_grid_path

    all_lst_files_in_dir = sorted(glob.glob(os.path.join(landsat_lst_dir, "*.tif")) + glob.glob(os.path.join(landsat_lst_dir, "*.img")))

    if not all_lst_files_in_dir:
        # If no files at all, return empty structures that subsequent functions can handle
        print(f"No LST .tif or .img files found in {landsat_lst_dir}. Returning empty data.")
        # It's crucial that the return types match for downstream consistency, even if empty.
        # Determine a default geo_profile structure or handle this upstream if truly no reference can be established.
        # For now, this scenario will likely cause issues if no reference grid can be determined.
        # A better approach might be to raise FileNotFoundError here if no LST files are found,
        # as a reference grid is essential.
        raise FileNotFoundError(f"No LST .tif or .img files found in {landsat_lst_dir}. Cannot establish reference grid.")

    file_date_mapping = {} # Maps datetime object to list of file paths
    for f_path in all_lst_files_in_dir:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-', '').replace('_', '')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping:
                    file_date_mapping[file_dt] = []
                file_date_mapping[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from LST filename: {fname}, skipping file.")
        else:
            print(f"Could not find date pattern in LST filename: {fname}, skipping file.")

    if not file_date_mapping:
        raise FileNotFoundError(f"No LST files with parseable dates found in {landsat_lst_dir}.")

    # Determine reference grid: use provided one or the first chronological LST file
    if not actual_reference_grid_path:
        first_available_date = sorted(file_date_mapping.keys())[0]
        actual_reference_grid_path = file_date_mapping[first_available_date][0]
        print(f"Using {actual_reference_grid_path} as reference grid for LST processing.")

    with rasterio.open(actual_reference_grid_path) as ref_src:
        geo_profile = ref_src.profile.copy()
        target_height, target_width = ref_src.height, ref_src.width
        geo_profile.update({
            'width': target_width,
            'height': target_height,
            'dtype': 'float32',
            'nodata': lst_nodata_val # LST files' nodata, will be converted to np.nan in stack
        })

    # Iterate through the sorted dates for which LST files were actually found and parsed
    sorted_file_dates = sorted(file_date_mapping.keys())
    
    # Filter dates based on config.START_DATE and config.END_DATE if they are set
    # This allows user to still constrain the overall period even with sparse loading
    if hasattr(app_config, 'START_DATE') and app_config.START_DATE and \
       hasattr(app_config, 'END_DATE') and app_config.END_DATE:
        start_dt_config = pd.to_datetime(app_config.START_DATE)
        end_dt_config = pd.to_datetime(app_config.END_DATE)
        dates_to_process = [
            dt for dt in sorted_file_dates if start_dt_config <= dt <= end_dt_config
        ]
        print(f"Filtered LST dates based on config START/END_DATE. Processing {len(dates_to_process)} dates.")
    else:
        dates_to_process = sorted_file_dates
        print(f"Processing all {len(dates_to_process)} found LST dates (no START/END_DATE filter from config).")


    for current_day_dt in tqdm(dates_to_process, desc="Processing Found Landsat LST Files"):
        lst_files_for_day = file_date_mapping.get(current_day_dt, []) # Should always find files here

        # Averaging logic for multiple files on the same day remains
        daily_lst_sum = np.zeros((target_height, target_width), dtype=np.float64)
        daily_valid_pixel_count = np.zeros((target_height, target_width), dtype=np.int16)

        processed_at_least_one_file_for_day = False
        for i, lst_fpath in enumerate(lst_files_for_day):
            temp_aligned_lst_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_lst_{current_day_dt.strftime('%Y%m%d')}_{i}.tif")
            try:
                utils.align_rasters(actual_reference_grid_path, lst_fpath, temp_aligned_lst_path,
                                    resampling_method=RasterioResampling.bilinear)
                with rasterio.open(temp_aligned_lst_path) as lsrc:
                    current_lst_data = lsrc.read(1, out_dtype=np.float32)
                    # Identify nodata pixels from the source (which match lst_nodata_val or are NaN from resampling)
                    nodata_mask_this_obs = ((current_lst_data == lst_nodata_val) | np.isnan(current_lst_data))
                    
                    # For averaging, use only valid data
                    valid_pixels_this_obs = ~nodata_mask_this_obs
                    
                    daily_lst_sum[valid_pixels_this_obs] += current_lst_data[valid_pixels_this_obs]
                    daily_valid_pixel_count[valid_pixels_this_obs] += 1
                os.remove(temp_aligned_lst_path)
                processed_at_least_one_file_for_day = True
            except Exception as e:
                print(f"Error processing or aligning LST file {lst_fpath} for date {current_day_dt}: {e}. Skipping this file.")
                continue
        
        if not processed_at_least_one_file_for_day and not lst_files_for_day:
            # This should not be reached if dates_to_process is derived from file_date_mapping keys
            # and file_date_mapping is not empty. But as a safeguard:
            print(f"Warning: No LST files were successfully processed for date {current_day_dt}, though it was in the list. Skipping.")
            continue

        avg_daily_lst = np.full((target_height, target_width), np.nan, dtype=np.float32) # Default to NaN
        valid_for_avg = daily_valid_pixel_count > 0
        avg_daily_lst[valid_for_avg] = (daily_lst_sum[valid_for_avg] / daily_valid_pixel_count[valid_for_avg]).astype(np.float32)
        
        # Where original LST file might have had lst_nodata_val, and averaging didn't occur, it remains NaN.
        # If all observations for a pixel were nodata, it also remains NaN.
        
        lst_data_list.append(avg_daily_lst)
        loaded_dates.append(current_day_dt)

    if not lst_data_list:
        # This implies dates_to_process was empty or all processing failed.
        raise ValueError("No Landsat LST data could be loaded based on available files and date filters.")

    lst_stack_nan = np.stack(lst_data_list, axis=0)

    # --- Outlier removal based on selected method from config ---
    outlier_method = getattr(app_config, 'LST_OUTLIER_METHOD', 'none').lower()
    print(f"Applying '{outlier_method}' LST outlier detection method.")

    if outlier_method == 'percentile':
        lower_percentile = getattr(app_config, 'LST_PERCENTILE_LOWER', 10)
        upper_percentile = getattr(app_config, 'LST_PERCENTILE_UPPER', 90)
        valid_lst_values = lst_stack_nan[~np.isnan(lst_stack_nan)]
        
        if valid_lst_values.size > 0:
            lower_bound = np.percentile(valid_lst_values, lower_percentile)
            upper_bound = np.percentile(valid_lst_values, upper_percentile)
            
            print(f"Using percentile method: Lower bound ({lower_percentile}%)={lower_bound:.2f}, Upper bound ({upper_percentile}%)={upper_bound:.2f}")
            
            outlier_mask = (lst_stack_nan < lower_bound) | (lst_stack_nan > upper_bound)
            num_outliers = np.sum(outlier_mask)
            
            if num_outliers > 0:
                print(f"Removing {num_outliers} outlier pixels ({num_outliers / valid_lst_values.size * 100:.2f}% of valid data).")
                lst_stack_nan[outlier_mask] = np.nan
        else:
            print("No valid LST data to perform outlier removal on.")


    elif outlier_method != 'none':
        print(f"Warning: Unknown LST_OUTLIER_METHOD '{outlier_method}'. Skipping outlier removal.")
    
    # --- End of outlier removal ---
    
    print(f"Loaded {len(loaded_dates)} LST scenes. Stack shape: {lst_stack_nan.shape}")
    # Return LST stack (with NaNs for nodata pixels) and the list of dates for which data was loaded.
    # Cloud mask stack is implicitly handled by NaNs in lst_stack_nan.
    return lst_stack_nan, loaded_dates, geo_profile, actual_reference_grid_path

def load_era5_skin_temp(
    era5_skin_temp_dir: str, 
    target_dates: list, # list of datetime objects from LST loading (actual LST dates)
    reference_grid_path: str, 
    app_config: 'config' # For OUTPUT_DIR
) -> tuple[np.ndarray, list]:
    """
    Loads, preprocesses, and aligns ERA5 skin temperature data for the ROI.
    Ensures data for all dates in `target_dates` (which should become `final_common_dates` upstream).
    Performs temporal linear interpolation then spatial linear interpolation to fill NaNs.

    Args:
        era5_skin_temp_dir (str): Directory containing ERA5 files.
        target_dates (list): List of datetime objects for which ERA5 data is required (e.g., final_common_dates).
        reference_grid_path (str): Path to the reference raster for alignment.
        app_config: Configuration object for accessing OUTPUT_DIR.

    Returns:
        tuple[np.ndarray, list]:
            - era5_stack_interpolated (np.ndarray): Time-series stack of ERA5 data (time, height, width),
                                                 aligned and fully interpolated.
            - target_dates (list): The input list of target_dates, returned for consistency.
    """
    print(f"Loading, aligning, and interpolating ERA5 skin temperature from: {era5_skin_temp_dir}")
    
    if not target_dates:
        print("Warning: load_era5_skin_temp received an empty list of target_dates. Returning empty ERA5 data.")
        # Determine target_height/width for empty array shape, or handle this error more strictly upstream.
        try:
            with rasterio.open(reference_grid_path) as ref_src_for_shape:
                h_empty, w_empty = ref_src_for_shape.height, ref_src_for_shape.width
            return np.empty((0, h_empty, w_empty), dtype=np.float32), []
        except Exception:
            return np.empty((0, 0, 0), dtype=np.float32), []

    # Get target H, W from ref grid
    with rasterio.open(reference_grid_path) as ref_src:
        target_height, target_width = ref_src.height, ref_src.width
        # Number of ERA5 bands to read (only skin temperature)
        n_bands = 1

    # Initialize the full stack with NaNs for one band
    era5_stack_full = np.full((len(target_dates), target_height, target_width), np.nan, dtype=np.float32)
    
    all_era5_files_in_dir = sorted(glob.glob(os.path.join(era5_skin_temp_dir, "*.tif")) + glob.glob(os.path.join(era5_skin_temp_dir, "*.img")))
    file_date_mapping_era5 = {}
    for f_path in all_era5_files_in_dir:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-','').replace('_','')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping_era5:
                    file_date_mapping_era5[file_dt] = []
                file_date_mapping_era5[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from ERA5 filename: {fname}")
        else:
            print(f"Could not find date pattern in ERA5 filename: {fname}")

    # Populate the stack with available data
    for i, date_dt in enumerate(tqdm(target_dates, desc="Processing ERA5 Data (Initial Load)")):
        era5_files_for_day = file_date_mapping_era5.get(date_dt, [])
        if not era5_files_for_day:
            continue # Leaves NaNs for this date in era5_stack_full
        
        era5_fpath = era5_files_for_day[0] # Take the first file if multiple
        temp_aligned_era5_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_era5_{date_dt.strftime('%Y%m%d')}.tif")
        try:
            utils.align_rasters(reference_grid_path, era5_fpath, temp_aligned_era5_path, 
                                resampling_method=RasterioResampling.nearest) # Nearest for ERA5 is often preferred
            with rasterio.open(temp_aligned_era5_path) as src:
                # Read only the first band (skin temperature)
                era5_data_single_band = src.read(1).astype(np.float32)  # shape (H, W)
                # Assign to stack: (time, row, col)
                era5_stack_full[i, :, :] = era5_data_single_band
            os.remove(temp_aligned_era5_path)
        except Exception as e:
            print(f"Error aligning ERA5 file {era5_fpath} for date {date_dt.strftime('%Y-%m-%d')}: {e}. Leaving NaNs for this date.")
            continue

    if hasattr(app_config, 'INTERPOLATE_ERA5') and not app_config.INTERPOLATE_ERA5:
        print("ERA5 interpolation skipped due to app_config.INTERPOLATE_ERA5 = False.")
        if np.isnan(era5_stack_full).any():
             print(f"Warning: Non-interpolated ERA5 stack contains {np.isnan(era5_stack_full).sum()} NaNs.")
        return era5_stack_full, target_dates

    # Temporal Interpolation (Linear)
    print("Performing temporal linear interpolation on ERA5 stack...")
    for r in tqdm(range(target_height), desc="ERA5 Temporal Interpolation", leave=False):
        for c in range(target_width):
            series = pd.Series(era5_stack_full[:, r, c])
            era5_stack_full[:, r, c] = series.interpolate(method='linear', limit_direction='both').to_numpy()

    # Spatial Interpolation (Linear using griddata) for remaining NaNs
    print("Performing spatial linear interpolation on ERA5 stack for remaining NaNs...")
    from scipy.interpolate import griddata
    points = None # To be initialized once
    for t in tqdm(range(len(target_dates)), desc="ERA5 Spatial Interpolation (Time Slices)", leave=False):
        slice_data = era5_stack_full[t, :, :]
        if np.isnan(slice_data).any():
            if points is None:
                x_coords, y_coords = np.meshgrid(np.arange(target_width), np.arange(target_height))
                points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
            valid_mask = ~np.isnan(slice_data)
            values = slice_data[valid_mask]
            points_with_values = points[valid_mask.ravel()]
            if points_with_values.shape[0] < 3:
                if points_with_values.shape[0] > 0:
                    nan_locs = np.where(np.isnan(slice_data))
                    nearest = griddata(points_with_values, values, (nan_locs[1], nan_locs[0]), method='nearest')
                    slice_data[nan_locs] = nearest
                era5_stack_full[t, :, :] = slice_data
                continue
            nan_locs = np.where(np.isnan(slice_data))
            grid_x, grid_y = nan_locs[1], nan_locs[0]
            try:
                interp_vals = griddata(points_with_values, values, (grid_x, grid_y), method='linear')
                if not np.all(np.isnan(interp_vals)):
                    slice_data[np.isnan(slice_data)] = interp_vals
            except Exception:
                interp_vals_nearest = griddata(points_with_values, values, (grid_x, grid_y), method='nearest')
                if not np.all(np.isnan(interp_vals_nearest)):
                    slice_data[np.isnan(slice_data)] = interp_vals_nearest
            era5_stack_full[t, :, :] = slice_data

    if np.isnan(era5_stack_full).any():
        print(f"Warning: {np.isnan(era5_stack_full).sum()} NaNs still present in ERA5 stack after all interpolation attempts.")
    else:
        print("ERA5 stack fully interpolated.")

    return era5_stack_full, target_dates

def load_era5_as_primary_timeline(
    era5_skin_temp_dir: str,
    app_config: 'config',
    reference_grid_path: str = None
) -> tuple[np.ndarray, list, str]:
    """
    Load ERA5 data to establish the primary timeline for the entire pipeline.
    This function determines what dates will be included in the final reconstruction.
    
    Args:
        era5_skin_temp_dir (str): Directory containing ERA5 files
        app_config: Configuration object
        reference_grid_path (str, optional): Reference grid path (can be None initially)
        
    Returns:
        tuple[np.ndarray, list, str]:
            - era5_stack (np.ndarray): ERA5 data stack (time, height, width)
            - era5_dates (list): List of datetime objects for ERA5 timeline
            - actual_reference_grid_path (str): Reference grid path used
    """
    print(f"Loading ERA5 data to establish primary timeline from: {era5_skin_temp_dir}")
    
    # Discover all ERA5 files
    all_era5_files = sorted(glob.glob(os.path.join(era5_skin_temp_dir, "*.tif")) + 
                           glob.glob(os.path.join(era5_skin_temp_dir, "*.img")))
    
    if not all_era5_files:
        raise FileNotFoundError(f"No ERA5 .tif or .img files found in {era5_skin_temp_dir}")
    
    # Parse dates from ERA5 filenames
    file_date_mapping = {}
    for f_path in all_era5_files:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-', '').replace('_', '')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping:
                    file_date_mapping[file_dt] = []
                file_date_mapping[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from ERA5 filename: {fname}, skipping file.")
        else:
            print(f"Could not find date pattern in ERA5 filename: {fname}, skipping file.")
    
    if not file_date_mapping:
        raise FileNotFoundError(f"No ERA5 files with parseable dates found in {era5_skin_temp_dir}")
    
    # Apply date filtering if configured
    sorted_era5_dates = sorted(file_date_mapping.keys())
    if hasattr(app_config, 'START_DATE') and app_config.START_DATE and \
       hasattr(app_config, 'END_DATE') and app_config.END_DATE:
        start_dt_config = pd.to_datetime(app_config.START_DATE)
        end_dt_config = pd.to_datetime(app_config.END_DATE)
        era5_dates_to_process = [
            dt for dt in sorted_era5_dates if start_dt_config <= dt <= end_dt_config
        ]
        print(f"Filtered ERA5 dates based on config START/END_DATE. Processing {len(era5_dates_to_process)} dates.")
    else:
        era5_dates_to_process = sorted_era5_dates
        print(f"Processing all {len(era5_dates_to_process)} found ERA5 dates (no START/END_DATE filter).")
    
    # Use the first ERA5 file as reference if none provided
    if not reference_grid_path:
        first_era5_date = era5_dates_to_process[0]
        reference_grid_path = file_date_mapping[first_era5_date][0]
        print(f"Using {reference_grid_path} as reference grid for ERA5 primary timeline.")
    
    # Get target dimensions
    with rasterio.open(reference_grid_path) as ref_src:
        target_height, target_width = ref_src.height, ref_src.width
    
    # Load ERA5 data for all dates
    era5_stack = np.full((len(era5_dates_to_process), target_height, target_width), np.nan, dtype=np.float32)
    
    for i, date_dt in enumerate(tqdm(era5_dates_to_process, desc="Loading ERA5 Primary Timeline")):
        era5_files_for_day = file_date_mapping.get(date_dt, [])
        if not era5_files_for_day:
            continue  # Leave NaNs for this date
        
        era5_fpath = era5_files_for_day[0]  # Take first file if multiple
        temp_aligned_era5_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_era5_primary_{date_dt.strftime('%Y%m%d')}.tif")
        
        try:
            utils.align_rasters(reference_grid_path, era5_fpath, temp_aligned_era5_path,
                              resampling_method=RasterioResampling.nearest)
            with rasterio.open(temp_aligned_era5_path) as src:
                era5_data = src.read(1).astype(np.float32)
                era5_stack[i, :, :] = era5_data
            os.remove(temp_aligned_era5_path)
        except Exception as e:
            print(f"Error loading ERA5 file {era5_fpath} for date {date_dt}: {e}")
            continue
    
    # Apply ERA5 interpolation if enabled
    if hasattr(app_config, 'INTERPOLATE_ERA5') and app_config.INTERPOLATE_ERA5:
        print("Performing temporal-spatial interpolation on ERA5 primary timeline...")
        era5_stack = interpolate_era5_stack(era5_stack, target_height, target_width, era5_dates_to_process)
    
    print(f"ERA5 primary timeline established with {len(era5_dates_to_process)} dates. Shape: {era5_stack.shape}")
    return era5_stack, era5_dates_to_process, reference_grid_path


def create_synthetic_lst_for_era5_timeline(
    era5_dates: list,
    landsat_lst_dir: str,
    reference_grid_path: str,
    lst_nodata_val: float,
    app_config: 'config'
) -> tuple[np.ndarray, list]:
    """
    Create LST stack aligned to ERA5 timeline by:
    1. Loading available LST data
    2. Creating synthetic LST for missing dates via temporal interpolation
    
    Args:
        era5_dates (list): Primary timeline dates from ERA5
        landsat_lst_dir (str): Directory containing LST files
        reference_grid_path (str): Reference grid for alignment
        lst_nodata_val (float): NoData value for LST
        app_config: Configuration object
        
    Returns:
        tuple[np.ndarray, list]:
            - lst_stack_aligned (np.ndarray): LST stack aligned to ERA5 timeline
            - era5_dates (list): Input ERA5 dates (returned for consistency)
    """
    print(f"Creating LST stack aligned to ERA5 timeline with synthetic gap-filling...")
    
    # Get target dimensions
    with rasterio.open(reference_grid_path) as ref_src:
        target_height, target_width = ref_src.height, ref_src.width
    
    # Initialize LST stack with NaNs
    lst_stack_aligned = np.full((len(era5_dates), target_height, target_width), np.nan, dtype=np.float32)
    
    # Load available LST data
    all_lst_files = sorted(glob.glob(os.path.join(landsat_lst_dir, "*.tif")) + 
                          glob.glob(os.path.join(landsat_lst_dir, "*.img")))
    
    lst_file_date_mapping = {}
    for f_path in all_lst_files:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-', '').replace('_', '')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in lst_file_date_mapping:
                    lst_file_date_mapping[file_dt] = []
                lst_file_date_mapping[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from LST filename: {fname}")
        else:
            print(f"Could not find date pattern in LST filename: {fname}")
    
    # Load LST data for available dates
    lst_dates_loaded = []
    for i, era5_date in enumerate(tqdm(era5_dates, desc="Loading Available LST Data")):
        lst_files_for_day = lst_file_date_mapping.get(era5_date, [])
        
        if lst_files_for_day:
            # Process LST data (same logic as original load_landsat_lst)
            daily_lst_sum = np.zeros((target_height, target_width), dtype=np.float64)
            daily_valid_count = np.zeros((target_height, target_width), dtype=np.int16)
            
            for j, lst_fpath in enumerate(lst_files_for_day):
                temp_aligned_lst_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_lst_era5_{era5_date.strftime('%Y%m%d')}_{j}.tif")
                try:
                    utils.align_rasters(reference_grid_path, lst_fpath, temp_aligned_lst_path,
                                      resampling_method=RasterioResampling.bilinear)
                    with rasterio.open(temp_aligned_lst_path) as lsrc:
                        current_lst_data = lsrc.read(1, out_dtype=np.float32)
                        nodata_mask = ((current_lst_data == lst_nodata_val) | np.isnan(current_lst_data))
                        valid_pixels = ~nodata_mask
                        
                        daily_lst_sum[valid_pixels] += current_lst_data[valid_pixels]
                        daily_valid_count[valid_pixels] += 1
                    os.remove(temp_aligned_lst_path)
                except Exception as e:
                    print(f"Error processing LST file {lst_fpath}: {e}")
                    continue
            
            # Calculate average LST for the day
            avg_daily_lst = np.full((target_height, target_width), np.nan, dtype=np.float32)
            valid_for_avg = daily_valid_count > 0
            avg_daily_lst[valid_for_avg] = (daily_lst_sum[valid_for_avg] / daily_valid_count[valid_for_avg]).astype(np.float32)
            
            lst_stack_aligned[i, :, :] = avg_daily_lst
            lst_dates_loaded.append(era5_date)
    
    print(f"Loaded LST data for {len(lst_dates_loaded)} out of {len(era5_dates)} ERA5 dates")
    
    # Apply outlier removal to loaded LST data
    outlier_method = getattr(app_config, 'LST_OUTLIER_METHOD', 'none').lower()
    print(f"Applying '{outlier_method}' LST outlier detection method.")
    
    if outlier_method == 'percentile':
        lower_percentile = getattr(app_config, 'LST_PERCENTILE_LOWER', 10)
        upper_percentile = getattr(app_config, 'LST_PERCENTILE_UPPER', 90)
        valid_lst_values = lst_stack_aligned[~np.isnan(lst_stack_aligned)]
        
        if valid_lst_values.size > 0:
            lower_bound = np.percentile(valid_lst_values, lower_percentile)
            upper_bound = np.percentile(valid_lst_values, upper_percentile)
            
            print(f"Using percentile method: Lower bound ({lower_percentile}%)={lower_bound:.2f}, Upper bound ({upper_percentile}%)={upper_bound:.2f}")
            
            outlier_mask = (lst_stack_aligned < lower_bound) | (lst_stack_aligned > upper_bound)
            num_outliers = np.sum(outlier_mask)
            
            if num_outliers > 0:
                print(f"Removing {num_outliers} outlier pixels ({num_outliers / valid_lst_values.size * 100:.2f}% of valid data).")
                lst_stack_aligned[outlier_mask] = np.nan
    
    # Perform temporal interpolation to fill missing LST dates
    print("Performing temporal interpolation to create synthetic LST for missing ERA5 dates...")
    for r in tqdm(range(target_height), desc="LST Temporal Interpolation", leave=False):
        for c in range(target_width):
            pixel_series = pd.Series(lst_stack_aligned[:, r, c])
            lst_stack_aligned[:, r, c] = pixel_series.interpolate(method='linear', limit_direction='both').to_numpy()
    
    # Optional: Apply spatial interpolation for remaining NaNs
    synthetic_dates_created = len(era5_dates) - len(lst_dates_loaded)
    print(f"Created synthetic LST for {synthetic_dates_created} dates via temporal interpolation")
    
    return lst_stack_aligned, era5_dates


def interpolate_era5_stack(era5_stack: np.ndarray, target_height: int, target_width: int, era5_dates: list) -> np.ndarray:
    """Helper function to interpolate ERA5 stack temporally and spatially."""
    # Temporal interpolation
    for r in tqdm(range(target_height), desc="ERA5 Temporal Interpolation", leave=False):
        for c in range(target_width):
            series = pd.Series(era5_stack[:, r, c])
            era5_stack[:, r, c] = series.interpolate(method='linear', limit_direction='both').to_numpy()
    
    # Spatial interpolation for remaining NaNs
    from scipy.interpolate import griddata
    points = None
    for t in tqdm(range(len(era5_dates)), desc="ERA5 Spatial Interpolation", leave=False):
        slice_data = era5_stack[t, :, :]
        if np.isnan(slice_data).any():
            if points is None:
                x_coords, y_coords = np.meshgrid(np.arange(target_width), np.arange(target_height))
                points = np.vstack((x_coords.ravel(), y_coords.ravel())).T
            
            valid_mask = ~np.isnan(slice_data)
            values = slice_data[valid_mask]
            points_with_values = points[valid_mask.ravel()]
            
            if points_with_values.shape[0] >= 3:
                nan_locs = np.where(np.isnan(slice_data))
                try:
                    interp_vals = griddata(points_with_values, values, 
                                         (nan_locs[1], nan_locs[0]), method='linear')
                    if not np.all(np.isnan(interp_vals)):
                        slice_data[nan_locs] = interp_vals
                except Exception:
                    interp_vals_nearest = griddata(points_with_values, values,
                                                 (nan_locs[1], nan_locs[0]), method='nearest')
                    if not np.all(np.isnan(interp_vals_nearest)):
                        slice_data[nan_locs] = interp_vals_nearest
                era5_stack[t, :, :] = slice_data
    
    return era5_stack


def load_sentinel2_reflectance(
    s2_dir: str, 
    # unique_dates_str: list[str], # No longer takes unique_dates_str directly
    target_dates: list, # List of datetime objects for which S2 data is required (e.g., primary_common_dates)
    actual_ref_grid_path: str, 
    target_height: int, 
    target_width: int,
    s2_nodata_value: float,
    app_config: 'config',
    num_s2_bands: int = 4 # Default S2 bands, can be passed from config or inferred if needed
) -> tuple[np.ndarray, list]: # Returns stack and the original target_dates
    """
    Loads, preprocesses, and temporally aligns Sentinel-2 reflectance data.
    Ensures data for all dates in `target_dates`. If S2 data is missing for a specific
    date, the corresponding slice in the output stack will be np.nan.
    Selects the least cloudy S2 image if multiple exist for a date.

    Args:
        s2_dir (str): Directory containing Sentinel-2 TIFF files.
        target_dates (list): list of datetime objects for which S2 data is required.
        actual_ref_grid_path (str): Path to the reference raster file used for alignment.
        target_height (int): Target height for aligned rasters.
        target_width (int): Target width for aligned rasters.
        s2_nodata_value (float): NoData value in S2 files.
        app_config: Configuration object.
        num_s2_bands (int): Number of expected S2 bands.

    Returns:
        tuple[np.ndarray, list]:
            - s2_stack_final (np.ndarray): Stack of S2 reflectance data (time, bands, height, width)
                                     aligned to target_dates, with np.nan for missing data.
            - target_dates (list): The input list of target_dates, returned for consistency.
    """
    print(f"Loading, merging, and aligning Sentinel-2 reflectance from: {s2_dir} for {len(target_dates)} target dates.")
    
    if not target_dates:
        print("Warning: load_sentinel2_reflectance received empty target_dates. Returning empty S2 data.")
        return np.empty((0, num_s2_bands, target_height, target_width), dtype=np.float32), []

    # Initialize the full stack with NaNs
    s2_stack_full = np.full((len(target_dates), num_s2_bands, target_height, target_width), np.nan, dtype=np.float32)

    # Discover S2 files and map them to dates
    all_s2_files_in_dir = sorted(glob.glob(os.path.join(s2_dir, "*.tif")) + glob.glob(os.path.join(s2_dir, "*.img")))
    file_date_mapping_s2 = {}
    for f_path in all_s2_files_in_dir:
        fname = os.path.basename(f_path)
        import re
        match = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', fname)
        if match:
            date_str_from_fname = match.group(1).replace('-','').replace('_','')
            try:
                file_dt = pd.to_datetime(date_str_from_fname, format='%Y%m%d')
                if file_dt not in file_date_mapping_s2:
                    file_date_mapping_s2[file_dt] = []
                file_date_mapping_s2[file_dt].append(f_path)
            except ValueError:
                print(f"Could not parse date from S2 filename: {fname}")
        else:
            print(f"Could not find date pattern in S2 filename: {fname}")

    for idx, date_dt in enumerate(tqdm(target_dates, desc="Processing S2 Data against Target Dates")):
        s2_files_for_this_date = file_date_mapping_s2.get(date_dt, [])

        if not s2_files_for_this_date:
            # print(f"No S2 file found for target date {date_dt.strftime('%Y-%m-%d')}. Leaving NaNs.")
            continue # Leaves NaNs for this date in s2_stack_full

        aligned_bands_for_day = []
        for s2_fpath in s2_files_for_this_date:
            temp_aligned_s2_path = os.path.join(app_config.OUTPUT_DIR, f"temp_aligned_s2_{os.path.basename(s2_fpath)}")
            try:
                utils.align_rasters(actual_ref_grid_path, s2_fpath, temp_aligned_s2_path)
                with rasterio.open(temp_aligned_s2_path) as src:
                    s2_data_all_bands_raw = src.read().astype(np.float32) # Reads all bands
                    
                    if s2_data_all_bands_raw.shape[0] != num_s2_bands:
                        print(f"Warning: S2 file {s2_fpath} has {s2_data_all_bands_raw.shape[0]} bands, expected {num_s2_bands}. Skipping this file for this date.")
                        continue # Skip this particular file if band count mismatch

                    # Handle known NoData values for S2.
                    current_s2_nodata_val = app_config.S2_NODATA_VALUE if hasattr(app_config, 'S2_NODATA_VALUE') else s2_nodata_value
                    s2_data_all_bands_raw[s2_data_all_bands_raw == current_s2_nodata_val] = np.nan
                    s2_data_all_bands_raw[s2_data_all_bands_raw < -99] = np.nan  # Handle -100 values
                    s2_data_all_bands_raw[np.isinf(s2_data_all_bands_raw)] = np.nan # Handle -inf too
                    
                    aligned_bands_for_day.append(s2_data_all_bands_raw) # (bands, height, width)
                os.remove(temp_aligned_s2_path)
            except Exception as e:
                print(f"Error aligning S2 file {s2_fpath} for date {date_dt.strftime('%Y-%m-%d')}: {e}")
                # If one file fails, we might still be able to use others for the same day.

        if not aligned_bands_for_day:
            # print(f"No S2 images successfully aligned for {date_dt.strftime('%Y-%m-%d')}. Leaving NaNs.")
            continue
        
        # Merge aligned images for the day using np.nanmean
        if len(aligned_bands_for_day) > 1:
            # print(f"Merging {len(aligned_bands_for_day)} S2 images for date {date_dt.strftime('%Y-%m-%d')} using nanmean.")
            stacked_for_mean = np.stack(aligned_bands_for_day, axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Mean of empty slice')
                warnings.filterwarnings('ignore', r'invalid value encountered in scalar divide')
                merged_s2_data = np.nanmean(stacked_for_mean, axis=0) # (bands, height, width)
        else:
            merged_s2_data = aligned_bands_for_day[0]
        
        s2_stack_full[idx, :, :, :] = merged_s2_data

    nan_pixels_in_s2_stack = np.isnan(s2_stack_full).sum()
    if nan_pixels_in_s2_stack > 0:
        print(f"S2 stack created with {nan_pixels_in_s2_stack} NaN pixel values (across all bands/times). These might be from missing files or original nodata.")
    print(f"Loaded and merged S2 data for {len(target_dates)} target dates. Stack shape: {s2_stack_full.shape}")
    return s2_stack_full, target_dates

def load_coordinates(reference_grid_path: str, normalize: bool = True) -> tuple[np.ndarray, np.ndarray, object, object]: # Updated Scaler type
    """
    Extracts x and y coordinates for each pixel from a reference grid.
    Optionally normalizes coordinates.
    """
    print("Loading and normalizing coordinates...")
    with rasterio.open(reference_grid_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        x_coords, y_coords = rasterio.transform.xy(transform, rows, cols, offset='center')
        x_coords = np.array(x_coords, dtype=np.float32)
        y_coords = np.array(y_coords, dtype=np.float32)

    x_scaler, y_scaler = None, None
    if normalize:
        # Ensure no NaNs/Infs if coords came from somewhere weird, though unlikely from rasterio.transform.xy
        x_coords_flat = x_coords.flatten()
        y_coords_flat = y_coords.flatten()
        x_coords_norm, y_coords_norm, x_scaler, y_scaler = utils.normalize_coordinates(x_coords_flat, y_coords_flat)
        x_coords = x_coords_norm.reshape(height, width)
        y_coords = y_coords_norm.reshape(height, width)
        print("Coordinates normalized.")
    else:
        print("Coordinates loaded without normalization.")
        
    return x_coords, y_coords, x_scaler, y_scaler

def preprocess_all_data_era5_primary(app_config) -> dict:
    """
    Alternative preprocessing function that uses ERA5 as the primary timeline.
    This ensures that reconstruction images will cover all ERA5 dates.
    
    Timeline is defined by ERA5 dates. LST data is loaded for available dates,
    and synthetic LST is created for missing dates via temporal interpolation.
    S2 and other data are aligned to this ERA5-based timeline.
    """
    print(f"Starting ERA5-primary preprocessing for ROI: {app_config.ROI_NAME}")
    os.makedirs(app_config.OUTPUT_DIR, exist_ok=True)

    # --- Dynamic Path Construction ---
    roi_base_path = os.path.join(app_config.BASE_DATA_DIR, app_config.ROI_NAME)
    landsat_lst_path = os.path.join(roi_base_path, app_config.LANDSAT_LST_SUBDIR)
    era5_skin_temp_path = os.path.join(roi_base_path, app_config.ERA5_SKIN_TEMP_SUBDIR)
    s2_reflectance_path = os.path.join(roi_base_path, app_config.SENTINEL2_REFLECTANCE_SUBDIR)
    
    # 1. Load ERA5 data first to establish the primary timeline
    try:
        era5_stack_primary, era5_primary_dates, actual_reference_grid_path = load_era5_as_primary_timeline(
            era5_skin_temp_dir=era5_skin_temp_path,
            app_config=app_config,
            reference_grid_path=None
        )
    except FileNotFoundError as e:
        raise ValueError(f"Critical error during ERA5 loading for ROI {app_config.ROI_NAME}: {e}") from e
    
    if not era5_primary_dates:
        raise ValueError(f"No ERA5 data found for ROI {app_config.ROI_NAME} after initial load. Cannot proceed.")

    target_height = era5_stack_primary.shape[1]
    target_width = era5_stack_primary.shape[2]

    # 2. Create LST stack aligned to ERA5 timeline with synthetic gap-filling
    try:
        lst_stack_aligned, _ = create_synthetic_lst_for_era5_timeline(
            era5_dates=era5_primary_dates,
            landsat_lst_dir=landsat_lst_path,
            reference_grid_path=actual_reference_grid_path,
            lst_nodata_val=app_config.LST_NODATA_VALUE,
            app_config=app_config
        )
    except Exception as e:
        print(f"Warning: Could not create synthetic LST stack: {e}")
        print("Creating empty LST stack for ERA5 timeline...")
        lst_stack_aligned = np.full((len(era5_primary_dates), target_height, target_width), np.nan, dtype=np.float32)
    
    print(f"Primary timeline established with {len(era5_primary_dates)} ERA5 dates.")
    print(f"  ERA5 stack shape: {era5_stack_primary.shape}")
    print(f"  LST stack shape (with synthetic): {lst_stack_aligned.shape}")

    # 3. Load Sentinel-2 reflectance data, aligned to ERA5 primary timeline
    num_s2_bands_expected = getattr(app_config, 'NUM_S2_BANDS', 4)
    s2_reflectance_stack_primary, _ = load_sentinel2_reflectance(
        s2_dir=s2_reflectance_path,
        target_dates=era5_primary_dates,  # Use ERA5 dates as target
        actual_ref_grid_path=actual_reference_grid_path,
        target_height=target_height,
        target_width=target_width,
        s2_nodata_value=app_config.S2_NODATA_VALUE,
        app_config=app_config,
        num_s2_bands=num_s2_bands_expected
    )
    print(f"S2 stack loaded for ERA5 timeline. Shape: {s2_reflectance_stack_primary.shape}")

    # 4. Create Day of Year (DOY) stack for the ERA5 timeline
    doy_stack_1d = np.array([date.timetuple().tm_yday for date in era5_primary_dates], dtype=np.int16)
    print(f"DOY stack (1D) created for ERA5 timeline. Shape: {doy_stack_1d.shape}")

    # 5. Load Coordinate data
    lon_coords, lat_coords, lon_scaler, lat_scaler = load_coordinates(
        reference_grid_path=actual_reference_grid_path, 
        normalize=True
    )
    print(f"Coordinates loaded. Lon shape: {lon_coords.shape}, Lat shape: {lat_coords.shape}")
    
    # 6. S2 Interpolation (applied to the S2 stack that's aligned with ERA5 dates)
    s2_final_stack = s2_reflectance_stack_primary  # Initialize
    if hasattr(app_config, 'INTERPOLATE_S2') and not app_config.INTERPOLATE_S2:
        print("S2 interpolation skipped due to app_config.INTERPOLATE_S2 = False.")
        if np.isnan(s2_final_stack).any():
            print(f"Warning: Non-interpolated S2 stack contains {np.isnan(s2_final_stack).sum()} NaNs.")
    elif s2_reflectance_stack_primary.size > 0 and era5_primary_dates:
        print(f"S2 stack for interpolation (aligned to ERA5 dates). Shape: {s2_reflectance_stack_primary.shape}")
        s2_interpolated_stack = s2_reflectance_stack_primary.copy()
        num_bands_s2 = s2_interpolated_stack.shape[1]
        num_iterations = getattr(app_config, 'S2_INTERPOLATION_ITERATIONS', 2)

        print(f"Starting iterative temporal-spatial interpolation for S2 stack ({num_iterations} iterations)...")
        from scipy.interpolate import griddata

        for iteration in range(num_iterations):
            print(f"  S2 Interpolation Iteration {iteration + 1}/{num_iterations}")
            nan_count_before_iter = np.isnan(s2_interpolated_stack).sum()
            if nan_count_before_iter == 0: 
                print("    No NaNs remaining in S2 stack. Stopping S2 interpolation early.")
                break

            # Temporal Linear Interpolation
            for b in tqdm(range(num_bands_s2), desc=f"  Iter {iteration+1} S2 Temporal Interp (Bands)", leave=False):
                for r in range(target_height):
                    for c in range(target_width):
                        pixel_series = pd.Series(s2_interpolated_stack[:, b, r, c])
                        s2_interpolated_stack[:, b, r, c] = pixel_series.interpolate(method='linear', limit_direction='both', limit_area=None).to_numpy()
            nan_count_after_temporal = np.isnan(s2_interpolated_stack).sum()
            print(f"    NaNs after temporal: {nan_count_after_temporal} (was {nan_count_before_iter})")
            if nan_count_after_temporal == 0: 
                print("    No NaNs remaining in S2 stack after temporal. Stopping S2 interpolation early this iteration.")
                break 

            # Spatial griddata (linear) - same logic as original
            points_spatial_s2 = None 
            for t in tqdm(range(s2_interpolated_stack.shape[0]), desc=f"  Iter {iteration+1} S2 Spatial Interp (Time)", leave=False):
                for b in range(num_bands_s2):
                    slice_data = s2_interpolated_stack[t, b, :, :]
                    if np.isnan(slice_data).any():
                        if points_spatial_s2 is None or points_spatial_s2.shape[0] != target_height*target_width :
                            x_coords_s2, y_coords_s2 = np.meshgrid(np.arange(target_width), np.arange(target_height))
                            points_spatial_s2 = np.vstack((x_coords_s2.ravel(), y_coords_s2.ravel())).T
                        
                        valid_mask_s2 = ~np.isnan(slice_data)
                        values_s2 = slice_data[valid_mask_s2]
                        points_with_values_s2 = points_spatial_s2[valid_mask_s2.ravel()]

                        if points_with_values_s2.shape[0] >= 3:
                            nan_locations_s2 = np.where(np.isnan(slice_data))
                            grid_x_nan_s2, grid_y_nan_s2 = nan_locations_s2[1], nan_locations_s2[0]
                            try:
                                interpolated_values_s2 = griddata(points_with_values_s2, values_s2, (grid_x_nan_s2, grid_y_nan_s2), method='linear')
                                if not np.all(np.isnan(interpolated_values_s2)):
                                    slice_data[np.isnan(slice_data)] = interpolated_values_s2
                            except Exception:
                                try:
                                    interpolated_values_s2_nearest = griddata(points_with_values_s2, values_s2, (grid_x_nan_s2, grid_y_nan_s2), method='nearest')
                                    if not np.all(np.isnan(interpolated_values_s2_nearest)):
                                        slice_data[np.isnan(slice_data)] = interpolated_values_s2_nearest
                                except Exception:
                                    pass
                        elif points_with_values_s2.shape[0] > 0 : 
                            nan_locations_s2 = np.where(np.isnan(slice_data))
                            try:
                                interpolated_values_s2_nearest = griddata(points_with_values_s2, values_s2, (nan_locations_s2[1], nan_locations_s2[0]), method='nearest')
                                if not np.all(np.isnan(interpolated_values_s2_nearest)):
                                    slice_data[nan_locations_s2] = interpolated_values_s2_nearest
                            except Exception:
                                pass
                        s2_interpolated_stack[t, b, :, :] = slice_data
            nan_count_after_spatial = np.isnan(s2_interpolated_stack).sum()
            print(f"    NaNs after spatial: {nan_count_after_spatial} (was {nan_count_after_temporal})")
        
        s2_final_stack = s2_interpolated_stack
        if np.isnan(s2_final_stack).any():
            print(f"Warning: S2 stack still contains {np.isnan(s2_final_stack).sum()} NaNs after {num_iterations} iter(s) of interpolation.")
        else:
            print("S2 stack successfully interpolated.")
    else: 
        print("S2 stack is empty or no ERA5 dates, or interpolation disabled. Skipping S2 interpolation.")
        s2_final_stack = np.full((len(era5_primary_dates), num_s2_bands_expected, target_height, target_width), np.nan, dtype=np.float32)

    # 7. Spatial Sampling for Training (same as original)
    training_pixel_mask = np.ones((target_height, target_width), dtype=bool)
    if hasattr(app_config, 'SPATIAL_TRAINING_SAMPLE_PERCENTAGE') and app_config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE < 1.0:
        sample_percentage = app_config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE
        min_pixels = getattr(app_config, 'MIN_PIXELS_FOR_SPATIAL_SAMPLING', 100)
        
        total_pixels_in_grid = target_height * target_width
        num_pixels_to_sample_float = total_pixels_in_grid * sample_percentage
        num_pixels_to_sample = max(min_pixels, int(num_pixels_to_sample_float))
        num_pixels_to_sample = min(num_pixels_to_sample, total_pixels_in_grid)

        print(f"Spatially sampling {num_pixels_to_sample} pixels ({sample_percentage*100}%, min set to {min_pixels}) for training.")
        
        all_pixel_indices = np.array([(r, c) for r in range(target_height) for c in range(target_width)])
        
        np.random.seed(getattr(app_config, 'RANDOM_SEED', 42))
        selected_indices_flat = np.random.choice(len(all_pixel_indices), size=num_pixels_to_sample, replace=False)
        selected_pixel_coords = all_pixel_indices[selected_indices_flat]
        
        training_pixel_mask = np.zeros((target_height, target_width), dtype=bool)
        for r_idx, c_idx in selected_pixel_coords:
            training_pixel_mask[r_idx, c_idx] = True
        print(f"Actual number of pixels selected for training mask: {np.sum(training_pixel_mask)}")
    else:
        print("Using all spatial pixels for training (SPATIAL_TRAINING_SAMPLE_PERCENTAGE is 1.0 or not defined).")

    print(f"ERA5-primary preprocessing complete for ROI: {app_config.ROI_NAME}. Final number of dates in ERA5 timeline: {len(era5_primary_dates)}")
    
    # Create geo_profile from the reference grid
    with rasterio.open(actual_reference_grid_path) as ref_src:
        geo_profile = ref_src.profile.copy()
        geo_profile.update({
            'width': target_width,
            'height': target_height,
            'dtype': 'float32',
            'nodata': app_config.LST_NODATA_VALUE
        })
    
    output_data = {
        "lst_stack": np.array(lst_stack_aligned, dtype=np.float32),
        "era5_stack": np.array(era5_stack_primary, dtype=np.float32),
        "s2_reflectance_stack": np.array(s2_final_stack, dtype=np.float32),
        "doy_stack": np.array(doy_stack_1d, dtype=np.int16),
        "lon_coords": np.array(lon_coords, dtype=np.float32),
        "lat_coords": np.array(lat_coords, dtype=np.float32),
        "geo_profile": geo_profile,
        "common_dates": era5_primary_dates,  # This is now the ERA5 timeline
        "lon_scaler": lon_scaler,
        "lat_scaler": lat_scaler,
        "reference_grid_path": actual_reference_grid_path,
        "roi_name": app_config.ROI_NAME,
        "training_pixel_mask": training_pixel_mask,
        "timeline_source": "era5_primary"  # Add metadata about which timeline approach was used
    }
        
    return output_data

    print(f"ERA5-primary preprocessing complete for ROI: {app_config.ROI_NAME}. Final number of dates in ERA5 timeline: {len(era5_primary_dates)}")
    
    # Create geo_profile from the reference grid
    with rasterio.open(actual_reference_grid_path) as ref_src:
        geo_profile = ref_src.profile.copy()
        geo_profile.update({
            'width': target_width,
            'height': target_height,
            'dtype': 'float32',
            'nodata': app_config.LST_NODATA_VALUE
        })
    
    output_data = {
        "lst_stack": np.array(lst_stack_aligned, dtype=np.float32),
        "era5_stack": np.array(era5_stack_primary, dtype=np.float32),
        "s2_reflectance_stack": np.array(s2_final_stack, dtype=np.float32),
        "doy_stack": np.array(doy_stack_1d, dtype=np.int16),
        "lon_coords": np.array(lon_coords, dtype=np.float32),
        "lat_coords": np.array(lat_coords, dtype=np.float32),
        "geo_profile": geo_profile,
        "common_dates": era5_primary_dates,  # This is now the ERA5 timeline
        "lon_scaler": lon_scaler,
        "lat_scaler": lat_scaler,
        "reference_grid_path": actual_reference_grid_path,
        "roi_name": app_config.ROI_NAME,
        "training_pixel_mask": training_pixel_mask,
        "timeline_source": "era5_primary"  # Add metadata about which timeline approach was used
    }
        
    return output_data