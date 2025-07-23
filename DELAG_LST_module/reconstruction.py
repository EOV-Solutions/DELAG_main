"""
Combines predictions from ATC and GP models and quantifies uncertainty.
This script can be run as a standalone module in the pipeline.
"""
import numpy as np
import torch # For any torch specific operations if needed, though mostly numpy here
from scipy.stats import norm # For confidence intervals
import argparse
import os

# Import project modules
import config
import utils

def combine_predictions(
    atc_predictions: np.ndarray, 
    atc_variance: np.ndarray, 
    gp_mean_residuals_map: np.ndarray, 
    gp_variance_residuals_map: np.ndarray,
    preprocessed_data: dict,
    app_config: 'config'
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combines ATC and GP predictions to get reconstructed LST and total uncertainty.
    Uses np.nan in lst_observed (from preprocessed_data['lst_stack']) to identify clear/cloudy pixels.

    Args:
        atc_predictions (np.ndarray): Mean ATC predictions (time, height, width).
        atc_variance (np.ndarray): Variance from ATC ensemble (time, height, width).
        gp_mean_residuals_map (np.ndarray): Predicted mean of GP residuals (time, height, width).
        gp_variance_residuals_map (np.ndarray): Predicted variance of GP residuals (time, height, width).
        preprocessed_data (dict): Dictionary containing preprocessed data, expects 'lst_stack'.
        app_config: Configuration object.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - reconstructed_lst (np.ndarray): Final reconstructed LST (time, height, width).
            - total_variance (np.ndarray): Total variance (time, height, width).
            - ci_lower (np.ndarray): Lower 95% confidence interval (time, height, width).
            - ci_upper (np.ndarray): Upper 95% confidence interval (time, height, width).
    """
    print("Combining ATC and GP predictions...")
    lst_observed = preprocessed_data['lst_stack'] # (time, height, width)
    num_times, height, width = atc_predictions.shape

    # The core idea is to start with the full model prediction and then merge
    # in the observed data where it is available. This prevents sharp edges
    # between observed and predicted regions.
    
    # 1. Calculate the full model prediction (ATC + GP) for all pixels
    # Where GP might be NaN (e.g., outside the training mask), it will result in NaN.
    # We will handle these cases.
    gp_mean_residuals_expanded = gp_mean_residuals_map
    gp_variance_residuals_expanded = gp_variance_residuals_map

    # Initial full reconstruction is ATC + GP
    reconstructed_lst = atc_predictions + gp_mean_residuals_expanded
    
    # Initial total variance is ATC + GP
    total_variance = atc_variance + gp_variance_residuals_expanded

    # 2. Identify locations of valid observations
    clear_pixels_mask = ~np.isnan(lst_observed)

    # 3. Replace the reconstructed values with the observed values at clear-sky locations
    reconstructed_lst[clear_pixels_mask] = lst_observed[clear_pixels_mask]
    
    # Optional: For clear pixels, you might want to adjust the variance.
    # The current logic (ATC_var + GP_var) is retained, reflecting model uncertainty
    # even at observed locations. This can be debated, but we'll keep it for now.
    # If we wanted to imply zero uncertainty for observations, we would do:
    # total_variance[clear_pixels_mask] = 0 
    
    # 4. Handle any remaining NaNs
    # This can happen if ATC or GP predictions were NaN for some pixels.
    # The `reconstructed_lst` will have NaNs in those locations. The code below
    # which calculates confidence intervals will correctly propagate these NaNs.
    
    print("Vectorized combination of ATC and GP predictions completed.")

    # Ensure variances are non-negative (e.g. due to numerical precision)
    total_variance[total_variance < 0] = 0 
    total_std_dev = np.sqrt(total_variance)

    # Calculate 95% confidence intervals
    # Z-score for 95% CI is approx 1.96
    z_score = norm.ppf(0.975) # More precise: 1.95996...
    ci_lower = reconstructed_lst - z_score * total_std_dev
    ci_upper = reconstructed_lst + z_score * total_std_dev

    print("Finished combining predictions and quantifying uncertainty.")
    return reconstructed_lst, total_variance, ci_lower, ci_upper


def main(args):
    """Main reconstruction pipeline function."""
    print("Starting DELAG Final Reconstruction Pipeline...")

    # --- Setup ---
    roi_name = args.roi_name
    data_split = args.data_split
    config.ROI_NAME = roi_name # Set for other modules if they use it
    
    # Suffix for directory names based on data split
    dir_suffix = '_train' if data_split == 'train' else '_test'

    # Define input directories
    data_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f'data_{data_split}')
    atc_pred_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f'atc_predicted_data{dir_suffix}')
    gp_pred_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f'gp_predicted_data{dir_suffix}')
    
    # Define output directories
    reconstructed_lst_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f'reconstructed_lst{dir_suffix}')
    uncertainty_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f'uncertainty_maps{dir_suffix}')
    os.makedirs(reconstructed_lst_dir, exist_ok=True)
    os.makedirs(uncertainty_dir, exist_ok=True)

    # --- 1. Load Data and Predictions ---
    print(f"Step 1: Loading data for ROI: '{roi_name}', Split: '{data_split}'")
    try:
        preprocessed_data = utils.load_processed_data(data_dir)
        atc_preds_dict = utils.load_atc_predictions(atc_pred_dir)
        atc_mean_predictions = atc_preds_dict['mean_predictions']
        atc_variance = atc_preds_dict['variance_predictions']
        print(f"Loaded data from {data_dir} and ATC predictions from {atc_pred_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hint: Make sure preprocess_data.py, train_atc.py, and predict_atc.py have been run for this ROI and split.")
        return

    # Initialize GP residuals to zero (ATC-only baseline)
    gp_mean_residuals_map = np.zeros_like(atc_mean_predictions)
    gp_variance_residuals_map = np.zeros_like(atc_mean_predictions)
    gp_model_applied = False # Flag to track if GP model was successfully loaded and applied

    if config.USE_GP_MODEL:
        print(f"Step 2: Attempting to load GP predictions from {gp_pred_dir}")
        try:
            gp_mean_residuals_file = os.path.join(gp_pred_dir, 'gp_mean_residuals.npy')
            gp_variance_residuals_file = os.path.join(gp_pred_dir, 'gp_variance_residuals.npy')
            
            gp_mean_residuals_map = np.load(gp_mean_residuals_file)
            gp_variance_residuals_map = np.load(gp_variance_residuals_file)
            gp_model_applied = True
            print("Successfully loaded GP predictions.")
        except FileNotFoundError:
            print(f"Warning: GP prediction files not found in {gp_pred_dir}. Proceeding with ATC-only reconstruction.")
            print("Hint: Run train_gp.py and predict_gp.py if GP model is intended to be used.")
        except Exception as e:
            print(f"An unexpected error occurred while loading GP predictions: {e}. Proceeding with ATC-only reconstruction.")
    else:
        print("Step 2: Skipping GP model (USE_GP_MODEL is False). Proceeding with ATC-only reconstruction.")

    if not gp_model_applied:
        print("Note: GP model was not applied. Final LST reconstruction is based on ATC predictions only.")

    # --- 2. Combine Predictions ---
    print("\nStep 3: Combining predictions and quantifying uncertainty...")
    reconstructed_lst, total_variance, _, _ = combine_predictions(
        atc_predictions=atc_mean_predictions,
        atc_variance=atc_variance,
        gp_mean_residuals_map=gp_mean_residuals_map,
        gp_variance_residuals_map=gp_variance_residuals_map,
        preprocessed_data=preprocessed_data,
        app_config=config
    )
    print("Prediction combination complete.")

    # --- 3. Save Outputs ---
    print(f"\nStep 4: Saving outputs to {reconstructed_lst_dir} and {uncertainty_dir}")
    try:
        num_times = reconstructed_lst.shape[0]
        dates = preprocessed_data['common_dates']
        ref_grid_path = preprocessed_data['reference_grid_path']

        for t in range(num_times):
            date_str = dates[t].strftime('%Y%m%d')
            
            # Reconstructed LST
            recon_filename = os.path.join(reconstructed_lst_dir, f"LST_RECON_{data_split.upper()}_{date_str}.tif")
            utils.save_array_as_geotiff(
                data_array=reconstructed_lst[t, :, :],
                reference_geotiff_path=ref_grid_path,
                output_path=recon_filename,
                nodata_value=np.nan
            )
            
            # Total Variance
            var_filename = os.path.join(uncertainty_dir, f"LST_TOTAL_VARIANCE_{data_split.upper()}_{date_str}.tif")
            utils.save_array_as_geotiff(
                data_array=total_variance[t, :, :],
                reference_geotiff_path=ref_grid_path,
                output_path=var_filename,
                nodata_value=np.nan
            )
        print(f"Successfully saved {num_times} reconstructed LST and variance maps.")
    except Exception as e:
        print(f"Error during saving outputs: {e}")
        import traceback
        traceback.print_exc()

    print("\nDELAG Final Reconstruction Pipeline Completed Successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DELAG Final LST Reconstruction from Predictions")
    parser.add_argument('--roi_name', type=str, required=True, help="Name of the Region of Interest (ROI) to process.")
    parser.add_argument(
        '--data_split', 
        type=str, 
        default='test', 
        choices=['train', 'test'],
        help="Data split to reconstruct: 'train' or 'test'. Defaults to 'test'."
    )
    args = parser.parse_args()
    main(args) 