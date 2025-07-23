"""
Main script to run the DELAG LST reconstruction pipeline.
"""
import numpy as np
import torch
import os
import json
import pandas as pd
from typing import Dict, Tuple
import glob
import rasterio

# Import project modules
import config
import utils
import reconstruction
import evaluation

def load_geotiff_stack(
    directory: str, 
    dates: list, 
    reference_geotiff_path: str, 
    file_pattern: str = "LST_RECON_TEST_*.tif"
) -> np.ndarray:
    """Loads a stack of GeoTIFFs from a directory based on a list of dates."""
    print(f"Loading GeoTIFF stack from: {directory} with pattern {file_pattern}")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    with rasterio.open(reference_geotiff_path) as ref_src:
        height, width = ref_src.height, ref_src.width

    stack = np.full((len(dates), height, width), np.nan, dtype=np.float32)
    
    all_files = glob.glob(os.path.join(directory, file_pattern))
    file_map = {os.path.basename(f).split('_')[-1].replace('.tif', ''): f for f in all_files}
    
    loaded_count = 0
    for i, date_obj in enumerate(dates):
        date_str = date_obj.strftime('%Y%m%d')
        if date_str in file_map:
            try:
                with rasterio.open(file_map[date_str]) as src:
                    stack[i, :, :] = src.read(1)
                loaded_count += 1
            except Exception as e:
                print(f"Warning: Could not read file {file_map[date_str]}. Error: {e}")
        else:
            print(f"Warning: No reconstructed file found for date {date_str} in {directory}")
            
    if loaded_count < len(dates):
        print(f"Warning: Only loaded {loaded_count}/{len(dates)} files from the reconstructed directory.")
        
    return stack


def main():
    """
    Main pipeline execution function for DELAG LST Reconstruction.
    
    This script is the final step for EVALUATION and VISUALIZATION.
    It assumes the following modular pipeline has already been executed:
    1. preprocess_data.py
    2. train_atc.py
    3. predict_atc.py
    4. train_gp.py (if enabled)
    5. predict_gp.py (if enabled)
    6. reconstruction.py - To generate the final reconstructed images
    
    This script then:
    - Loads test data and model predictions
    - Prepares raw model predictions for evaluation
    - Evaluates model performance against test data
    - Loads final reconstructed images for visualization
    - Saves final outputs and visualizations
    """
    print("Starting DELAG LST Evaluation & Visualization Pipeline...")
    
    if not hasattr(config, 'ROI_NAME'):
        print("Error: config.ROI_NAME must be set before running.")
        return
        
    roi_name = config.ROI_NAME
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR_BASE, roi_name)
    
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 1. Load Pre-processed Test Data and Train Data (for split info)
    print("\nStep 1: Loading Pre-processed Test Data")
    try:
        train_data_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, 'data_train')
        test_data_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, 'data_test')

        train_data = utils.load_processed_data(train_data_dir)
        test_data = utils.load_processed_data(test_data_dir)

        print(f"Test data loaded from: {test_data_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hint: Run preprocess_data.py first to generate data.")
        return
    except Exception as e:  
        print(f"An unexpected error occurred while loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load ATC & GP Predictions for the Test set
    print("\nStep 2: Loading Model Predictions (Test Set)")
    try:
        atc_predictions_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, "atc_predicted_data_train")
        atc_predictions = utils.load_atc_predictions(atc_predictions_dir)
        atc_mean_predictions_test = atc_predictions['mean_predictions']
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hint: Run predict_atc.py --data_split test first.")
        return

    # -------------------------------------------------------------
    # Attempt to load GP residual predictions – optional component
    # -------------------------------------------------------------
    gp_mean_residuals_map_test = np.zeros_like(atc_mean_predictions_test)
    if config.USE_GP_MODEL:
        gp_loaded = False
        gp_pred_dir_candidates = [os.path.join(config.OUTPUT_DIR_BASE, roi_name, "gp_predicted_data_train")]

        for gp_predictions_dir in gp_pred_dir_candidates:
            gp_mean_residuals_file = os.path.join(gp_predictions_dir, "gp_mean_residuals.npy")
            if os.path.exists(gp_mean_residuals_file):
                try:
                    gp_mean_residuals_map_test = np.load(gp_mean_residuals_file)
                    print(f"Loaded GP residuals from {gp_mean_residuals_file}")
                    gp_loaded = True
                    break
                except Exception as e:
                    print(f"Warning: Failed to load GP residuals from {gp_mean_residuals_file}: {e}")

        if not gp_loaded:
            print("Warning: GP predictions not available. Proceeding with ATC-only evaluation (zero residuals).")

    # --- Add a check for the content of GP residuals ---
    if np.isnan(gp_mean_residuals_map_test).all():
        print("\nFATAL WARNING: The loaded GP residual predictions are all NaN.")
        print("This indicates a critical failure in the GP prediction step.")
        print("Reverting to ATC-only data for evaluation and visualization to prevent errors.")
        gp_mean_residuals_map_test = np.zeros_like(atc_mean_predictions_test) # Reset
    elif np.isnan(gp_mean_residuals_map_test).any():
        nan_count = np.isnan(gp_mean_residuals_map_test).sum()
        total_count = gp_mean_residuals_map_test.size
        nan_percentage = (nan_count / total_count) * 100
        print(f"\nWARNING: GP residuals contain {nan_count} / {total_count} ({nan_percentage:.2f}%) NaN values.")
        print("Replacing NaN values with 0 to allow for visualization.")
        np.nan_to_num(gp_mean_residuals_map_test, nan=0.0, copy=False)
    
    print("Model predictions loading completed.")
    
    # 3. Prepare model's raw predictions for evaluation purposes
    print("\nStep 3: Preparing raw model predictions for evaluation...")
    model_predictions_for_eval = atc_mean_predictions_test + gp_mean_residuals_map_test

    # 4. Evaluate Model Performance on Test Data
    print("\nStep 4: Model Evaluation on Test Data")
    try:
        all_eval_metrics = evaluation.run_all_evaluations(
            model_predicted_lst=model_predictions_for_eval,
            observed_lst_clear=train_data['lst_stack'],
            app_config=config
        )
        print("\nFinal Evaluation Metrics (Test Data):")
        for k, v in all_eval_metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    # 5. Load Final Reconstructed Images for Visualization
    print("\nStep 5: Loading final reconstructed LST for visualization")
    try:
        reconstructed_lst_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, 'reconstructed_lst_train')
        reconstructed_lst_test = load_geotiff_stack(
            directory=reconstructed_lst_dir,
            dates=train_data['common_dates'],
            reference_geotiff_path=train_data['reference_grid_path'],
            file_pattern="LST_RECON_TRAIN_*.tif"
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Hint: Run reconstruction.py --data_split train first.")
        # As a fallback for visualization, use the raw predictions
        reconstructed_lst_test = model_predictions_for_eval
        print("Warning: Using raw model predictions for visualization as reconstructed files were not found.")
    
    # 6. Visualize daily comparison stacks
    print("\nStep 6: Visualizing daily comparison stacks for test data...")
    try:
        utils.visualize_daily_stacks_comparison(
            lst_observed_stack=train_data['lst_stack'],
            model_predicted_lst_stack=model_predictions_for_eval,
            reconstructed_lst_stack=reconstructed_lst_test,
            era5_stack=train_data['era5_stack'],
            s2_reflectance_stack=train_data['s2_reflectance_stack'],
            ndvi_stack=train_data.get('ndvi_stack'),
            common_dates=train_data['common_dates'],
            output_base_dir=config.OUTPUT_DIR, 
            roi_name=roi_name,
            app_config=config,
            s2_rgb_indices=getattr(config, 'S2_RGB_INDICES', (2, 1, 0)),
            max_days_to_plot=getattr(config, 'MAX_DAYS_FOR_DAILY_VISUALIZATION_PLOT', 10)
        )
    except Exception as e:
        print(f"Error during daily comparison visualization: {e}")
        import traceback
        traceback.print_exc()

    # 7. Save Run Configuration & Data Split Info
    print("\nStep 7: Saving Run Configuration and Data Split Info")
    try:
        # Save config
        config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__') and not callable(getattr(config, key))}
        config_filename = os.path.join(config.OUTPUT_DIR, 'run_config.json')
        with open(config_filename, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str)
        print(f"Run configuration saved to {config_filename}")
        
        # Save split info
        split_info = {
            'train_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in train_data['common_dates']],
            'test_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in test_data['common_dates']],
        }
        split_info_filename = os.path.join(config.OUTPUT_DIR, 'data_split_info.json')
        with open(split_info_filename, 'w') as f:
            json.dump(split_info, f, indent=4)
        print(f"Data split information saved to {split_info_filename}")
    except Exception as e:
        print(f"Error saving metadata: {e}")

    print("\nDELAG LST Evaluation & Visualization Pipeline Completed.")


if __name__ == '__main__':
    # Before running this final script, ensure that config.ROI_NAME is set correctly
    # and that all previous pipeline steps (preprocessing, training, prediction, reconstruction)
    # have been completed for that ROI.
    main() 