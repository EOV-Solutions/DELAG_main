"""
Utility functions for the DELAG project.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling as RasterioResampling # Alias to avoid conflict

def normalize_coordinates(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Normalize x and y coordinates to the range [0, 1].

    Args:
        x_coords (np.ndarray): Array of x coordinates.
        y_coords (np.ndarray): Array of y coordinates.

    Returns:
        tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
            Normalized x coordinates, normalized y coordinates,
            fitted x scaler, and fitted y scaler.
    """
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    # Reshape if 1D array for scaler
    x_coords_reshaped = x_coords.reshape(-1, 1) if x_coords.ndim == 1 else x_coords
    y_coords_reshaped = y_coords.reshape(-1, 1) if y_coords.ndim == 1 else y_coords

    norm_x = x_scaler.fit_transform(x_coords_reshaped).reshape(x_coords.shape)
    norm_y = y_scaler.fit_transform(y_coords_reshaped).reshape(y_coords.shape)

    return norm_x, norm_y, x_scaler, y_scaler

def create_output_directories(config):
    """
    Creates necessary output directories defined in the config.
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.RECONSTRUCTED_LST_PATH, exist_ok=True)
    os.makedirs(config.UNCERTAINTY_MAPS_PATH, exist_ok=True)
    os.makedirs(config.MODEL_WEIGHTS_PATH, exist_ok=True)
    print(f"Output directories created/ensured at {config.OUTPUT_DIR} and model weights at {config.MODEL_WEIGHTS_PATH}")

def resample_raster(input_path: str, output_path: str, target_resolution: tuple[float, float], target_crs: str = None, resampling_method: RasterioResampling = RasterioResampling.nearest):
    """
    Resamples a raster to a target resolution and optionally a target CRS.

    Args:
        input_path (str): Path to the input raster file.
        output_path (str): Path to save the resampled raster.
        target_resolution (tuple[float, float]): Target resolution (x_res, y_res).
        target_crs (str, optional): Target CRS (e.g., 'EPSG:32632'). If None, uses source CRS.
        resampling_method (RasterioResampling, optional): Resampling method. Defaults to nearest.
    """
    with rasterio.open(input_path) as src:
        dst_crs = rasterio.CRS.from_string(target_crs) if target_crs else src.crs
        
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=target_resolution
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )
    print(f"Resampled {input_path} to {output_path} with resolution {target_resolution}")

def align_rasters(reference_raster_path: str, raster_to_align_path: str, output_aligned_path: str, resampling_method: RasterioResampling = RasterioResampling.nearest):
    """
    Aligns a raster to a reference raster's grid (extent, resolution, CRS).

    Args:
        reference_raster_path (str): Path to the reference raster.
        raster_to_align_path (str): Path to the raster to be aligned.
        output_aligned_path (str): Path to save the aligned raster.
        resampling_method (RasterioResampling, optional): Resampling method. Defaults to nearest.
    """
    with rasterio.open(reference_raster_path) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_bounds = ref.bounds
        ref_meta = ref.meta.copy()

    with rasterio.open(raster_to_align_path) as src:
        # Update meta for the output file based on the reference
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": ref_height,
            "width": ref_width,
            "transform": ref_transform,
            "crs": ref_crs
        })
        
        # Create an empty array for the output aligned raster
        # Using src.count to handle multi-band rasters correctly
        dst_array = np.empty((src.count, ref_height, ref_width), dtype=src.dtypes[0])


        with rasterio.open(output_aligned_path, 'w', **out_meta) as dst:
             for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=dst_array[i-1], # reproject into the pre-allocated array band
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling_method
                )
             dst.write(dst_array) # Write all bands at once

    # print(f"Aligned {raster_to_align_path} to {reference_raster_path} and saved to {output_aligned_path}")

def get_day_of_year(dates: np.ndarray) -> np.ndarray:
    """
    Convert an array of datetime64 objects or YYYY-MM-DD strings to day of year (1-366).
    """
    if not isinstance(dates, np.ndarray):
        dates = np.array(dates)
    if dates.dtype != '<M8[D]': # Check if not already datetime64[D]
        dates = dates.astype('datetime64[D]')
    
    # pandas Series can simplify datetime operations
    try:
        import pandas as pd
        return pd.Series(dates).dt.dayofyear.values
    except ImportError:
        # Manual calculation if pandas is not available or preferred
        # This is a simplified version, for more robust handling, pandas is recommended
        year_starts = dates.astype('datetime64[Y]')
        day_of_year = (dates - year_starts).astype('timedelta64[D]').astype(int) + 1
        return day_of_year

def save_array_as_geotiff(data_array: np.ndarray, reference_geotiff_path: str, output_path: str, nodata_value=None):
    """
    Saves a NumPy array as a GeoTIFF file, using metadata from a reference GeoTIFF.
    Assumes data_array is 2D (single band) or 3D (bands, height, width).
    """
    with rasterio.open(reference_geotiff_path) as ref:
        profile = ref.profile
        profile['driver'] = 'GTiff' # Ensure GeoTIFF format
        
        if data_array.ndim == 2:
            profile['count'] = 1
            data_array_to_write = data_array[np.newaxis, :, :] # Add band dimension
        elif data_array.ndim == 3:
            profile['count'] = data_array.shape[0]
            data_array_to_write = data_array
        else:
            raise ValueError(f"Input data_array must be 2D or 3D, got {data_array.ndim}D")

        profile['dtype'] = data_array.dtype
        if nodata_value is not None:
            profile['nodata'] = nodata_value

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data_array_to_write)
    print(f"Saved array to {output_path}")

def save_processed_data(data_dict: dict, output_dir: str):
    """
    Saves a preprocessed data dictionary to a specified output directory.
    Numpy arrays are saved as .npy files, and other metadata is saved as a JSON file.

    Args:
        data_dict (dict): The dictionary containing data to save.
        output_dir (str): The full path to the directory to save the data into.
    """
    import json
    import pandas as pd

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {}
    
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            # Define path for .npy file and store it in metadata
            filename = f"{key}.npy"
            array_path = os.path.join(output_dir, filename)
            np.save(array_path, value)
            metadata[key] = filename # Store filename in metadata
        else:
            # Collect other metadata
            metadata[key] = value
            
    # Custom JSON serializer for non-serializable types
    def json_serializer(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(obj, 'strftime'): # Handles datetime.datetime, etc.
            return obj.strftime('%Y-%m-%dT%H:%M:%S')
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Fallback for other types, e.g. Path objects
        return str(obj)

    # Save metadata as json, including the list of saved arrays
    metadata_filepath = os.path.join(output_dir, "metadata.json")
    try:
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=4, default=json_serializer)
        print(f"Saved processed data to {output_dir}")
    except Exception as e:
        print(f"Could not save metadata to {metadata_filepath}: {e}")

def load_processed_data(data_dir: str) -> dict:
    """
    Loads preprocessed data that was saved by save_processed_data.
    This function reconstructs the data dictionary by loading the metadata.json file
    and then scanning the directory for all .npy files.

    Args:
        data_dir (str): The full path to the directory containing the data files.

    Returns:
        dict: The loaded data dictionary.
    
    Raises:
        FileNotFoundError: If the data directory or metadata.json file does not exist.
    """
    import json
    import pandas as pd

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    metadata_path = os.path.join(data_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found in directory: {metadata_path}")

    # 1. Load the base metadata from the JSON file
    with open(metadata_path, 'r') as f:
        data_dict = json.load(f)

    # 2. Scan the directory for all .npy files and load them
    print(f"Scanning {data_dir} for .npy files...")
    for filename in os.listdir(data_dir):
        if filename.endswith(".npy"):
            key = os.path.splitext(filename)[0]
            array_path = os.path.join(data_dir, filename)
            try:
                data_dict[key] = np.load(array_path, allow_pickle=True)
                print(f"  Loaded numpy array: {filename}")
            except Exception as e:
                print(f"Warning: Could not load numpy array {filename}: {e}")

    # 3. Convert date strings back to datetime objects if they exist
    if 'common_dates' in data_dict and data_dict['common_dates'] and isinstance(data_dict['common_dates'][0], str):
        try:
            # Use pandas for robust date parsing
            data_dict['common_dates'] = pd.to_datetime(data_dict['common_dates']).tolist()
        except ImportError:
            print("Warning: pandas is not installed. Dates will be left as strings.")
        except Exception as e:
            print(f"Warning: Could not parse dates in metadata: {e}")

    return data_dict

def save_atc_predictions(mean_predictions: np.ndarray, variance_predictions: np.ndarray, output_dir: str):
    """
    Saves ATC model predictions (mean and variance) to a specified directory.
    
    Args:
        mean_predictions (np.ndarray): The mean predictions from the ATC model.
        variance_predictions (np.ndarray): The variance predictions from the ATC model.
        output_dir (str): The full path to the directory to save the predictions into.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'mean_predictions.npy'), mean_predictions)
    np.save(os.path.join(output_dir, 'variance_predictions.npy'), variance_predictions)
    print(f"ATC predictions saved to {output_dir}")

def load_atc_predictions(predictions_dir: str) -> dict:
    """
    Loads ATC predictions from a specified directory.

    Args:
        predictions_dir (str): The full path to the directory containing prediction files.

    Returns:
        A dictionary with 'mean_predictions' and 'variance_predictions'.
        
    Raises:
        FileNotFoundError: If the predictions directory or .npy files do not exist.
    """
    if not os.path.exists(predictions_dir):
        raise FileNotFoundError(f"ATC predictions directory not found: {predictions_dir}")
        
    mean_path = os.path.join(predictions_dir, "mean_predictions.npy")
    variance_path = os.path.join(predictions_dir, "variance_predictions.npy")
    
    if not os.path.exists(mean_path) or not os.path.exists(variance_path):
        raise FileNotFoundError(f"ATC prediction files not found in {predictions_dir}. Run predict_atc.py first.")

    mean_predictions = np.load(mean_path)
    variance_predictions = np.load(variance_path)
    
    return {
        'mean_predictions': mean_predictions,
        'variance_predictions': variance_predictions
    }

def visualize_reconstruction_results(
    original_lst_day: np.ndarray, 
    reconstructed_lst_day: np.ndarray, 
    total_variance_day: np.ndarray, 
    reference_grid_path: str, # For extent/aspect if needed, or just for consistency
    output_dir: str,
    date_str: str,
    roi_name: str,
    save_plot: bool = True
):
    """
    Visualizes the original LST, reconstructed LST, and total variance for a specific day.

    Args:
        original_lst_day (np.ndarray): 2D array of original LST data (with NaNs).
        reconstructed_lst_day (np.ndarray): 2D array of reconstructed LST data.
        total_variance_day (np.ndarray): 2D array of total variance data.
        reference_grid_path (str): Path to a reference GeoTIFF (can be used for extent/aspect or context).
        output_dir (str): Directory to save the visualization.
        date_str (str): Date string for titles and filename (e.g., 'YYYYMMDD').
        roi_name (str): Name of the ROI for filename uniqueness.
        save_plot (bool): Whether to save the plot to a file.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import contextily as cx # For adding basemaps, optional enhancement
    except ImportError:
        print("Matplotlib or contextily is not installed. Skipping visualization.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"LST Reconstruction Results for {roi_name} - {date_str}", fontsize=16)

    # Determine common min/max for LST plots for consistent color scaling
    valid_lst_values = np.concatenate([
        original_lst_day[~np.isnan(original_lst_day)].flatten(),
        reconstructed_lst_day[~np.isnan(reconstructed_lst_day)].flatten()
    ])
    lst_min, lst_max = (np.min(valid_lst_values), np.max(valid_lst_values)) if valid_lst_values.size > 0 else (270, 320)

    # 1. Original LST
    im1 = axes[0].imshow(original_lst_day, cmap='plasma', vmin=lst_min, vmax=lst_max)
    axes[0].set_title("Original LST (with clouds as NaN)")
    axes[0].set_xlabel("X-coordinate")
    axes[0].set_ylabel("Y-coordinate")
    fig.colorbar(im1, ax=axes[0], orientation='horizontal', label='Temperature (K)')
    axes[0].grid(True, alpha=0.3)

    # 2. Reconstructed LST
    im2 = axes[1].imshow(reconstructed_lst_day, cmap='plasma', vmin=lst_min, vmax=lst_max)
    axes[1].set_title("Reconstructed LST")
    axes[1].set_xlabel("X-coordinate")
    axes[1].set_ylabel("Y-coordinate")
    fig.colorbar(im2, ax=axes[1], orientation='horizontal', label='Temperature (K)')
    axes[1].grid(True, alpha=0.3)

    # 3. Total Variance
    # Determine min/max for variance plot, ensuring it's not too skewed by outliers
    valid_var_values = total_variance_day[~np.isnan(total_variance_day)].flatten()
    var_min, var_max = (np.min(valid_var_values), np.percentile(valid_var_values, 98)) if valid_var_values.size > 0 else (0, 1)
    if var_max <= var_min: # Handle case where variance is constant or all NaN
        var_max = var_min + 1e-6 if valid_var_values.size > 0 else 1.0
        var_min = var_min -1e-6 if valid_var_values.size > 0 and var_min >0 else 0.0
        if var_max <= var_min: var_max = var_min + 0.1 # Final fallback for constant zero variance
            
    im3 = axes[2].imshow(total_variance_day, cmap='magma', vmin=var_min, vmax=var_max)
    axes[2].set_title("Total Variance")
    axes[2].set_xlabel("X-coordinate")
    axes[2].set_ylabel("Y-coordinate")
    fig.colorbar(im3, ax=axes[2], orientation='horizontal', label='Variance (K^2)')
    axes[2].grid(True, alpha=0.3)
    
    # Attempt to add basemap if contextily is available and reference grid provides CRS/extent
    # This is an enhancement and might require further adjustments based on CRS.
    try:
        with rasterio.open(reference_grid_path) as ref_src:
            extent = [ref_src.bounds.left, ref_src.bounds.right, ref_src.bounds.bottom, ref_src.bounds.top]
            for ax in axes:
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
                # cx.add_basemap(ax, crs=ref_src.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik, alpha=0.5)
                # Note: Contextily might require reprojection of data if not in web mercator.
                # For simplicity, this example just sets extent. True geographic overlay is more complex.
                pass # Placeholder for more advanced basemap integration if desired
    except Exception as e:
        print(f"Could not use reference grid for plot extent/basemap: {e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

    if save_plot:
        plot_subdir = os.path.join(output_dir, "plots")
        os.makedirs(plot_subdir, exist_ok=True)
        filename = os.path.join(plot_subdir, f"reconstruction_vis_{roi_name}_{date_str}.png")
        plt.savefig(filename, dpi=150)
        print(f"Saved visualization to {filename}")
        plt.close(fig) # Close the figure to free memory
    else:
        plt.show()

def plot_s2_rgb(s2_slice_bands_hw, ax, title="S2 RGB", band_indices_rgb=(2,1,0)):
    """Helper to plot an S2 RGB image from a (bands, height, width) slice."""
    if s2_slice_bands_hw is None or s2_slice_bands_hw.shape[0] < max(band_indices_rgb) + 1:
        ax.text(0.5, 0.5, 'S2 Data Not Available', horizontalalignment='center', verticalalignment='center')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    # Extract R, G, B bands
    # Assuming band_indices_rgb provides (Red_idx, Green_idx, Blue_idx)
    red = s2_slice_bands_hw[band_indices_rgb[0], :, :]
    green = s2_slice_bands_hw[band_indices_rgb[1], :, :]
    blue = s2_slice_bands_hw[band_indices_rgb[2], :, :]

    # Stack to (height, width, channels)
    rgb_image = np.stack([red, green, blue], axis=-1)

    # Normalize/clip for display (simple percentile clip)
    # Replace NaNs with a value (e.g., 0) before percentile calculation to avoid issues, or use np.nanpercentile
    rgb_image_no_nan = np.nan_to_num(rgb_image, nan=0.0)
    
    # Calculate percentiles for each channel independently on non-NaN data if possible
    # For simplicity here, applying a general clip or normalization
    # A more robust approach would handle NaNs carefully in percentile calculation
    # and potentially apply per-channel contrast stretching.
    
    # Simple clip based on overall min/max of each channel after NaN handling
    # or use fixed percentile like 2% and 98% to enhance contrast if data range is large.
    # For reflectance data (0-1 theoretically, but can be higher/lower due to processing)
    # We can clip to a reasonable range e.g. 0 to 0.3-0.4 for visualization
    
    # Apply a robust scaling: map 2nd to 98th percentile to 0-1 for each band
    # This helps with atmospheric effects or outliers making images too dark/light.
    scaled_rgb_image = np.zeros_like(rgb_image_no_nan, dtype=float)
    for i in range(3):
        band_data = rgb_image_no_nan[..., i]
        if np.any(band_data): # Check if band has non-zero data
            min_val = np.percentile(band_data[band_data > -np.inf], 2) # use > -np.inf to handle potential -inf from nan_to_num if input was weird
            max_val = np.percentile(band_data[band_data < np.inf], 98)
            if max_val <= min_val: # Prevent division by zero or negative range
                scaled_rgb_image[..., i] = np.clip((band_data - min_val) / (1e-6), 0, 1)
            else:
                scaled_rgb_image[..., i] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
        else:
            scaled_rgb_image[..., i] = 0 # Black if band is all zero
            
    # Handle any remaining NaNs that might have slipped through (e.g. if original was all NaN)
    scaled_rgb_image = np.nan_to_num(scaled_rgb_image, nan=0.0)
    
    ax.imshow(scaled_rgb_image)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def visualize_daily_stacks_comparison(
    lst_observed_stack: np.ndarray, 
    model_predicted_lst_stack: np.ndarray,
    reconstructed_lst_stack: np.ndarray, 
    era5_stack: np.ndarray, # ERA5 stack (time, H, W) - single band skin temperature
    s2_reflectance_stack: np.ndarray, # (time, bands, H, W)
    ndvi_stack: np.ndarray, # (time, H, W), can be None
    common_dates: list, 
    output_base_dir: str, # e.g., config.OUTPUT_DIR
    roi_name: str,
    app_config: 'config', # Added to access GP_USE_NDVI_FEATURE
    s2_rgb_indices: tuple = (2, 1, 0), # (R, G, B) assuming B2,B3,B4,B8 -> B4=idx 2, B3=idx 1, B2=idx 0
    max_days_to_plot: int = 10, # Limit the number of columns for readability
    lst_contrast_percentiles: tuple = (2, 98) # Percentiles for LST contrast stretching (e.g., 2nd and 98th)
):
    """
    Visualizes a comparison of observed LST, reconstructed LST, and either S2 RGB or NDVI images 
    across multiple days in a grid plot.

    Args:
        lst_observed_stack (np.ndarray): (time, H, W) stack of observed LST.
        model_predicted_lst_stack (np.ndarray): (time, H, W) stack of model predicted LST.
        reconstructed_lst_stack (np.ndarray): (time, H, W) stack of reconstructed LST.
        era5_stack (np.ndarray): (time, H, W) stack of ERA5 data.
        s2_reflectance_stack (np.ndarray): (time, bands, H, W) stack of S2 reflectance.
        ndvi_stack (np.ndarray): (time, H, W) stack of NDVI data, can be None.
        common_dates (list): List of datetime objects corresponding to the time dimension.
        output_base_dir (str): Base output directory (e.g., config.OUTPUT_DIR).
        roi_name (str): Name of the ROI for filenames and titles.
        app_config (config): The application configuration object.
        s2_rgb_indices (tuple): Indices for R, G, B bands in the s2_reflectance_stack's band dimension.
        max_days_to_plot (int): Maximum number of days (columns) to plot.
        lst_contrast_percentiles (tuple): Lower and upper percentiles to clip LST data for contrast enhancement.
                                         Set to (0, 100) or None to use absolute min/max.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping daily stacks comparison visualization.")
        return

    # Determine the number of days to plot
    num_days_to_plot = min(lst_observed_stack.shape[0], max_days_to_plot)

    # Global LST range for consistent color scaling (now removed)
    # all_lst_values = np.concatenate([
    #     lst_observed_stack.flatten(),
    #     model_predicted_lst_stack.flatten(),
    #     reconstructed_lst_stack.flatten()
    # ])
    # lst_min_val = np.nanmin(all_lst_values) if not np.all(np.isnan(all_lst_values)) else 273
    # lst_max_val = np.nanmax(all_lst_values) if not np.all(np.isnan(all_lst_values)) else 313
    
    # Create subplots
    # Number of rows is now 5: Obs LST, Pred LST, Recon LST, ERA5 Skin Temp, S2/NDVI
    fig, axes = plt.subplots(5, num_days_to_plot, figsize=(num_days_to_plot * 3.5, 5 * 3), squeeze=False)
    # squeeze=False ensures axes is always 2D, even if num_days=1
    
    plot_title_suffix = "S2 RGB"
    if getattr(app_config, 'GP_USE_NDVI_FEATURE', False) and ndvi_stack is not None:
        plot_title_suffix = "NDVI"

    fig.suptitle(f"Daily Comparison for {roi_name} (Obs, Pred, Recon LST, ERA5 Skin Temp, {plot_title_suffix})", fontsize=14, y=0.99)

    for i in range(num_days_to_plot):
        date_str = common_dates[i].strftime('%Y-%m-%d')
        
        # --- Calculate vmin and vmax for this iteration only ---
        iter_obs_lst = lst_observed_stack[i]
        iter_pred_lst = model_predicted_lst_stack[i]
        iter_recon_lst = reconstructed_lst_stack[i]
        
        iter_all_values = np.concatenate([
            iter_obs_lst.flatten(),
            iter_pred_lst.flatten(),
            iter_recon_lst.flatten()
        ])
        
        iter_min_val = np.nanmin(iter_all_values)
        iter_max_val = np.nanmax(iter_all_values)

        if not np.isfinite(iter_min_val) or not np.isfinite(iter_max_val):
            iter_min_val, iter_max_val = 273, 313 # Default fallback
        # --- End of dynamic vmin/vmax calculation ---

        # Row 0: Observed LST
        ax_obs = axes[0, i]
        im_obs = ax_obs.imshow(iter_obs_lst, cmap='coolwarm', vmin=iter_min_val, vmax=iter_max_val)
        ax_obs.set_title(f"{date_str}\\nObserved LST (K)")
        ax_obs.axis('off')
        if i == 0: ax_obs.set_ylabel("Observed LST", fontsize=12)

        # Row 1: Model Predicted LST
        ax_pred = axes[1, i]
        im_pred = ax_pred.imshow(iter_pred_lst, cmap='coolwarm', vmin=iter_min_val, vmax=iter_max_val)
        ax_pred.set_title(f"Model Predicted LST (K)")
        ax_pred.axis('off')
        if i == 0: ax_pred.set_ylabel("Predicted LST", fontsize=12)

        # Row 2: Reconstructed LST
        ax_recon = axes[2, i]
        im_recon = ax_recon.imshow(iter_recon_lst, cmap='coolwarm', vmin=iter_min_val, vmax=iter_max_val)
        ax_recon.set_title(f"Reconstructed LST (K)\\n(Range: {iter_min_val:.1f} - {iter_max_val:.1f})") 
        ax_recon.axis('off')
        if i == 0: ax_recon.set_ylabel("Reconstructed LST", fontsize=12)

        # Row 3: ERA5 Skin Temperature
        ax_era1 = axes[3, i]
        if era5_stack is not None and era5_stack.shape[0] > i:
            im_era1 = ax_era1.imshow(era5_stack[i, :, :], cmap='coolwarm', vmin=iter_min_val, vmax=iter_max_val)
            ax_era1.set_title(f"ERA5 Skin Temp (K)")
        else:
            ax_era1.text(0.5, 0.5, 'ERA5 N/A', ha='center', va='center')
            ax_era1.set_title(f"ERA5 Skin Temp (K)")
        ax_era1.axis('off')
        if i == 0: ax_era1.set_ylabel("ERA5 Skin Temperature", fontsize=12)

        # Remove Row 4: ERA5 Band 2 (no longer exists)
        # We need to shift the remaining rows up

        # Row 4: S2 RGB or NDVI (previously Row 5)
        ax_s2_or_ndvi = axes[4, i]
        use_ndvi_plot = getattr(app_config, 'GP_USE_NDVI_FEATURE', False)

        if use_ndvi_plot and ndvi_stack is not None and ndvi_stack.shape[0] > i:
            im_ndvi = ax_s2_or_ndvi.imshow(ndvi_stack[i], cmap='RdYlGn', vmin=-1, vmax=1)
            ax_s2_or_ndvi.set_title(f"NDVI")
            ax_s2_or_ndvi.axis('off')
            # Optional: Add a colorbar for NDVI if desired, though often the range is standard
            # if i == num_days - 1: # Add colorbar to the last plot
            #     fig.colorbar(im_ndvi, ax=ax_s2_or_ndvi, orientation='vertical', label='NDVI', fraction=0.046, pad=0.04)
        elif s2_reflectance_stack is not None and s2_reflectance_stack.shape[0] > i:
            plot_s2_rgb(s2_reflectance_stack[i], ax_s2_or_ndvi, title=f"S2 RGB", band_indices_rgb=s2_rgb_indices)
            # plot_s2_rgb already calls axis('off')
        else:
            ax_s2_or_ndvi.text(0.5, 0.5, 'Image Data Not Available', horizontalalignment='center', verticalalignment='center')
            ax_s2_or_ndvi.set_title(f"{plot_title_suffix}")
            ax_s2_or_ndvi.axis('off')

        if i == 0:
            y_label_row5 = "NDVI" if use_ndvi_plot and ndvi_stack is not None else "S2 RGB"
            ax_s2_or_ndvi.set_ylabel(y_label_row5, fontsize=12)

    # Final layout adjustment
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect to prevent title overlap
    
    # Save the figure
    output_filename = os.path.join(output_base_dir, f"daily_comparison_stacks_{roi_name}.png")
    try:
        plt.savefig(output_filename, dpi=150, format='png')
        print(f"Saved daily comparison visualization to {output_filename}")
    except Exception as e:
        print(f"Error saving daily comparison plot: {e}")
            
    plt.close(fig) # Close the figure to free memory 

def plot_mean_atc_loss_over_intervals(
    mean_train_losses: list[float], 
    mean_val_losses: list[float],
    epoch_intervals_x_axis: list[int], 
    output_dir: str, 
    roi_name: str,
    loss_logging_interval: int
):
    """
    Plots the mean training and validation loss for the ATC model over training intervals.

    Args:
        mean_train_losses (list[float]): List of mean training losses per interval.
        mean_val_losses (list[float] or np.ndarray): List or array of mean validation losses per interval. Can be None.
        epoch_intervals_x_axis (list[int]): X-axis ticks representing the end epoch of each interval.
        output_dir (str): Directory to save the plot.
        roi_name (str): Name of the ROI for the plot title and filename.
        loss_logging_interval (int): The interval used for logging, for the plot label.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping ATC loss plot.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot training loss
    # Filter out NaNs for plotting
    valid_train_indices = [i for i, loss in enumerate(mean_train_losses) if np.isfinite(loss)]
    if valid_train_indices:
        valid_train_epochs = [epoch_intervals_x_axis[i] for i in valid_train_indices]
        valid_train_losses = [mean_train_losses[i] for i in valid_train_indices]
        ax.plot(valid_train_epochs, valid_train_losses, 'o-', color='dodgerblue', label='Mean Training Loss', markersize=4)

    # Plot validation loss if available
    if mean_val_losses is not None:
        # Check if it's a list or numpy array and if it has finite values
        valid_val_indices = [i for i, loss in enumerate(mean_val_losses) if np.isfinite(loss)]
        if valid_val_indices:
            valid_val_epochs = [epoch_intervals_x_axis[i] for i in valid_val_indices]
            valid_val_losses = [mean_val_losses[i] for i in valid_val_indices]
            if len(valid_val_losses) == len(valid_val_epochs):
                ax.plot(valid_val_epochs, valid_val_losses, 's--', color='orangered', label='Mean Validation Loss', markersize=4)

    ax.set_xlabel(f"Epoch (Logged Every {loss_logging_interval} Epochs)")
    ax.set_ylabel("Mean Squared Error (MSE) Loss")
    ax.set_title(f"ATC Model - Mean Training & Validation Loss for {roi_name}")
    ax.legend()
    ax.set_yscale('log') # Log scale is often better for viewing loss curves
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Improve x-axis ticks if there are many intervals
    if len(epoch_intervals_x_axis) > 20:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    
    # Save the figure
    plot_filename = os.path.join(output_dir, f"atc_mean_loss_curve_{roi_name}.png")
    try:
        fig.savefig(plot_filename, dpi=300)
        print(f"ATC mean loss curve plot saved to {plot_filename}")
    except Exception as e:
        print(f"Error saving ATC mean loss curve plot: {e}")
    
    plt.close(fig)

def plot_mean_gp_loss_over_intervals(
    mean_interval_losses: list[float], 
    epoch_intervals_x_axis: list[int], 
    output_dir: str, 
    roi_name: str,
    loss_logging_interval: int
):
    """
    Plots the mean GP training loss over specified epoch intervals and saves the plot.

    Args:
        mean_interval_losses (list[float]): List of mean loss values for each interval.
        epoch_intervals_x_axis (list[int]): List of epoch numbers marking the end of each interval (for x-axis).
        output_dir (str): The base output directory (e.g., config.OUTPUT_DIR).
        roi_name (str): Name of the ROI for the plot filename.
        loss_logging_interval (int): The interval at which losses were logged (e.g., 10 epochs).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping mean GP loss plot.")
        return

    if not mean_interval_losses or not epoch_intervals_x_axis:
        print("Mean GP interval losses or epoch intervals are empty. Skipping plot.")
        return
    
    if len(mean_interval_losses) != len(epoch_intervals_x_axis):
        print(f"Warning: Mismatch in length of mean_gp_interval_losses ({len(mean_interval_losses)}) and epoch_intervals_x_axis ({len(epoch_intervals_x_axis)}). Skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_indices = [i for i, loss in enumerate(mean_interval_losses) if not np.isnan(loss)]
    plottable_losses = [mean_interval_losses[i] for i in valid_indices]
    plottable_epochs = [epoch_intervals_x_axis[i] for i in valid_indices]

    if not plottable_losses:
        print("No valid (non-NaN) mean GP interval losses to plot. Skipping plot.")
        plt.close(fig)
        return

    ax.plot(plottable_epochs, plottable_losses, marker='o', linestyle='-', color='green')
    
    ax.set_xlabel(f"Epoch (Loss averaged over previous {loss_logging_interval} epochs)")
    ax.set_ylabel("Mean GP Training Loss (Negative ELBO)")
    ax.set_title(f"Mean GP Training Loss for {roi_name}")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if len(plottable_epochs) > 10:
        ax.set_xticks(plottable_epochs[::len(plottable_epochs)//10]) 
    else:
        ax.set_xticks(plottable_epochs)
    
    plt.tight_layout()

    plot_viz_dir = os.path.join(output_dir, "viz")
    os.makedirs(plot_viz_dir, exist_ok=True)
    
    filename = os.path.join(plot_viz_dir, f"mean_gp_training_loss_{roi_name}.png")
    try:
        plt.savefig(filename, dpi=150)
        print(f"Saved mean GP training loss plot to {filename}")
    except Exception as e:
        print(f"Error saving mean GP loss plot: {e}")
    
    plt.close(fig)

def plot_input_data_timeseries_overview(
    doy_stack_numpy: np.ndarray,
    lst_stack: np.ndarray, # (time, height, width)
    era5_stack: np.ndarray, # (time, height, width) - single band skin temperature
    training_pixel_mask: np.ndarray, # (height, width)
    output_dir: str,
    roi_name: str
):
    """
    Plots an overview of the input data timeseries (LST, ERA5 bands)
    averaged over the spatial domain defined by the training_pixel_mask.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping input data timeseries overview plot.")
        return

    if training_pixel_mask is None:
        print("Warning: training_pixel_mask is None in plot_input_data_timeseries_overview. Plotting average over all pixels.")
        training_pixel_mask = np.ones((lst_stack.shape[1], lst_stack.shape[2]), dtype=bool)

    # Apply training_pixel_mask to LST and ERA5 stacks
    # We want to calculate mean only over the training pixels.
    # Create masked versions of the stacks
    lst_masked = np.where(training_pixel_mask[np.newaxis, :, :], lst_stack, np.nan)
    
    # era5_stack has shape (time, height, width) - single band now
    # training_pixel_mask needs to be broadcast correctly: (1, height, width)
    era5_masked = np.where(training_pixel_mask[np.newaxis, :, :], era5_stack, np.nan)

    # Calculate spatial means, ignoring NaNs
    mean_lst_timeseries = np.nanmean(lst_masked, axis=(1, 2))
    mean_era5_timeseries = np.nanmean(era5_masked, axis=(1, 2))  # Single band ERA5
    
    # Ensure the "visualizations" subdirectory exists
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    output_filename = os.path.join(viz_dir, f"input_timeseries_overview_{roi_name}.png")

    plt.figure(figsize=(12, 7))
    
    plt.scatter(doy_stack_numpy, mean_lst_timeseries, marker='o', label='Mean LST (Training Pixels)', s=25)
    plt.scatter(doy_stack_numpy, mean_era5_timeseries, marker='x', label='Mean ERA5 Skin Temperature (Training Pixels)', s=25)
    # The original code had mean_era5_band2_timeseries, but era5_stack is now single band.
    # Assuming the intent was to plot the single band mean if ndvi_stack is None or not used.
    # For now, removing the line as it's no longer applicable.
    # plt.scatter(doy_stack_numpy, mean_era5_band2_timeseries, marker='s', label='Mean ERA5 Band 2 (Training Pixels)', s=25)

    plt.title(f"Input Data Timeseries Overview for {roi_name} (Spatially Averaged over Training Pixels)")
    plt.xlabel("Day of Year (DOY)")
    plt.ylabel("Temperature (K) / Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(output_filename)
        print(f"Input data timeseries overview plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving input data timeseries overview plot: {e}")
    plt.close()