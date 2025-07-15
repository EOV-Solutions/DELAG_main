"""
Main script to run the DELAG LST reconstruction pipeline.
"""
import numpy as np
import torch
import os
import json
import pandas as pd
from typing import Dict, Tuple

# Import project modules
import config
import utils
import reconstruction
import evaluation





def main():
    """
    Main pipeline execution function for DELAG LST Reconstruction.
    
    This script assumes the following modular pipeline has been executed:
    1. preprocess_data.py - Preprocesses and splits data into train/test sets
    2. train_atc.py - Trains ATC models using training data
    3. predict_atc.py - Generates ATC predictions on test data
    4. train_gp.py - Trains GP model for residuals (if enabled)
    5. predict_gp.py - Generates GP residual predictions on test data (if enabled)
    
    This script then:
    - Loads all pre-generated predictions
    - Combines ATC and GP predictions
    - Performs final reconstruction and uncertainty quantification
    - Evaluates model performance
    - Saves final outputs and visualizations
    """
    print("Starting DELAG LST Reconstruction Pipeline...")
    print("Note: This script loads pre-generated predictions from modular pipeline components.")
    
    # To keep output consistent, we define the output directory based on the ROI name from config.
    # It is assumed that config.ROI_NAME is set appropriately before this script is run,
    # for example, by an external script or a known convention.
    if hasattr(config, 'ROI_NAME'):
        # The main output directory for this ROI run
        config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR_BASE, config.ROI_NAME)
        # The specific directory for model weights
        config.MODEL_WEIGHTS_PATH = os.path.join(config.OUTPUT_DIR_BASE, 'output_models', config.ROI_NAME)
    else:
        # Fallback for when ROI_NAME is not in config, though it should be.
        print("Warning: config.ROI_NAME is not set. Using default output directory.")

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)
        # Potentially set deterministic algorithms if desired, though might impact performance
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    # 0. Create output directories (idempotent)
    utils.create_output_directories(config)

    # 1. Load Pre-processed Data
    print("\nStep 1: Loading Pre-processed Train and Test Data")
    try:
        # The ROI name must be consistent. We get it from the config, assuming it's set there.
        roi_name = getattr(config, 'ROI_NAME', 'UnknownROI')
        if roi_name == 'UnknownROI':
            print("Warning: ROI_NAME not found in config. Defaulting to 'UnknownROI'.")
            print("         Ensure the data exists in 'output/data_train/UnknownROI/'")

        # Define base directories for train and test data using the modular structure
        train_data_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, 'data_train', roi_name)
        test_data_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, 'data_test', roi_name)

        train_data = utils.load_processed_data(train_data_dir)
        test_data = utils.load_processed_data(test_data_dir)

        print("Pre-processed data loading completed.")
        print(f"  Train data loaded from: {train_data_dir}")
        print(f"  Test data loaded from: {test_data_dir}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("  Hint: Run preprocess_data.py first to generate preprocessed data.")
        return # Stop pipeline if data is not found
    except Exception as e:  
        print(f"An unexpected error occurred while loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load ATC Predictions
    print("\nStep 2: Loading ATC Predictions")
    try:
        # Load ATC predictions from the correct directory structure
        atc_predictions_dir = os.path.join(config.OUTPUT_DIR_BASE,roi_name, "atc_predicted_data")
        # print(os.listdir(atc_predictions_dir))
        atc_predictions = utils.load_atc_predictions(atc_predictions_dir)
        atc_mean_predictions_test = atc_predictions['mean_predictions']
        atc_variance_test = atc_predictions['variance_predictions']
        
        print(f"ATC predictions loaded from: {atc_predictions_dir}")
        print(f"  Mean predictions shape: {atc_mean_predictions_test.shape}")
        print(f"  Variance predictions shape: {atc_variance_test.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("  Hint: Run train_atc.py and predict_atc.py first to generate ATC predictions.")
        return
    except Exception as e:
        print(f"Error loading ATC predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("ATC predictions loading completed.")
    
    # --- START DIAGNOSTIC BLOCK FOR ATC ---
    if 'lst_stack' in test_data and atc_mean_predictions_test is not None:
        nan_in_lst_stack = np.isnan(test_data['lst_stack'])
        nan_in_atc_pred = np.isnan(atc_mean_predictions_test)
        total_nan_in_lst = np.sum(nan_in_lst_stack)
        total_nan_in_atc = np.sum(nan_in_atc_pred)
        print(f"  DIAGNOSTIC: Total NaNs in test LST stack: {total_nan_in_lst}")
        print(f"  DIAGNOSTIC: Total NaNs in ATC mean predictions on test: {total_nan_in_atc}")
        if total_nan_in_lst > 0 and total_nan_in_atc > 0:
            matching_nans_atc = np.sum(nan_in_lst_stack & nan_in_atc_pred)
            print(f"  DIAGNOSTIC: NaN locations in test LST stack that are also NaN in ATC predictions: {matching_nans_atc}")
            if matching_nans_atc > 0 and matching_nans_atc == total_nan_in_atc and matching_nans_atc >= np.sum(nan_in_lst_stack): # Check if all NaNs in ATC are from LST
                 print("  WARNING: ATC predictions appear to carry over NaNs from the input LST stack. Gap-filling by ATC may not be effective.")
        elif total_nan_in_atc == 0 and total_nan_in_lst > 0:
            print("  INFO: ATC predictions do not contain NaNs, suggesting it might be performing gap-filling.")
    # --- END DIAGNOSTIC BLOCK FOR ATC ---

    # 3. Load GP Residual Predictions (if enabled)
    if config.USE_GP_MODEL:
        print("\nStep 3: Loading GP Residual Predictions (USE_GP_MODEL is True)")
        try:
            # Load pre-generated GP residual predictions
            gp_predictions_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, "gp_predicted_data")
            
            if not os.path.exists(gp_predictions_dir):
                raise FileNotFoundError(f"GP predictions directory not found: {gp_predictions_dir}. Run train_gp.py and predict_gp.py first.")
            
            # Load GP residual predictions
            gp_mean_residuals_file = os.path.join(gp_predictions_dir, 'gp_mean_residuals.npy')
            gp_variance_residuals_file = os.path.join(gp_predictions_dir, 'gp_variance_residuals.npy')
            
            if not os.path.exists(gp_mean_residuals_file) or not os.path.exists(gp_variance_residuals_file):
                raise FileNotFoundError(f"GP residual prediction files not found. Run predict_gp.py first.")
            
            gp_mean_residuals_map_test = np.load(gp_mean_residuals_file)
            gp_variance_residuals_map_test = np.load(gp_variance_residuals_file)
            
            print(f"GP residual predictions loaded from: {gp_predictions_dir}")
            print(f"  Mean residuals shape: {gp_mean_residuals_map_test.shape}")
            print(f"  Variance residuals shape: {gp_variance_residuals_map_test.shape}")

        except Exception as e:
            print(f"Error loading GP residual predictions: {e}")
            import traceback
            traceback.print_exc()
            print("  WARNING: Falling back to ATC-only prediction due to an error loading GP predictions.")
            gp_mean_residuals_map_test = np.zeros_like(atc_mean_predictions_test)
            gp_variance_residuals_map_test = np.zeros_like(atc_mean_predictions_test)
    else:
        print("\nStep 3: Skipping GP Model processing (USE_GP_MODEL is False)")
        # If GP is disabled, residuals are zero, and their variance is zero.
        gp_mean_residuals_map_test = np.zeros_like(atc_mean_predictions_test)
        gp_variance_residuals_map_test = np.zeros_like(atc_mean_predictions_test)

    # 4. Combine Predictions and Quantify Uncertainty for Test Data
    print("\nStep 4: Combining Predictions and Quantifying Uncertainty on Test Data")
    # This step produces the 'reconstructed_lst' for test data that is used for evaluation,
    # which correctly incorporates observed data for clear pixels.
    try:
        reconstructed_lst_test, total_variance_test, ci_lower_test, ci_upper_test = reconstruction.combine_predictions(
            atc_predictions=atc_mean_predictions_test,
            atc_variance=atc_variance_test,
            gp_mean_residuals_map=gp_mean_residuals_map_test,
            gp_variance_residuals_map=gp_variance_residuals_map_test,
            preprocessed_data=test_data,
            app_config=config
        )
    except Exception as e:
        print(f"Error during final reconstruction and uncertainty quantification: {e}")
        import traceback
        traceback.print_exc()
        return
    print("Final reconstruction and uncertainty quantification completed.")
    
    # --- START DIAGNOSTIC BLOCK FOR RECONSTRUCTION ---
    if 'lst_stack' in test_data and reconstructed_lst_test is not None:
        nan_in_lst_stack = np.isnan(test_data['lst_stack'])
        nan_in_reconstructed = np.isnan(reconstructed_lst_test)
        total_nan_in_lst = np.sum(nan_in_lst_stack)
        total_nan_in_reconstructed = np.sum(nan_in_reconstructed)
        print(f"  DIAGNOSTIC: Total NaNs in test LST stack: {total_nan_in_lst}")
        print(f"  DIAGNOSTIC: Total NaNs in final reconstructed LST: {total_nan_in_reconstructed}")
        if total_nan_in_lst > 0 and total_nan_in_reconstructed > 0:
            matching_nans_reconstructed = np.sum(nan_in_lst_stack & nan_in_reconstructed)
            print(f"  DIAGNOSTIC: NaN locations in test LST stack that are also NaN in reconstructed LST: {matching_nans_reconstructed}")
            if matching_nans_reconstructed > 0 and matching_nans_reconstructed == total_nan_in_reconstructed and matching_nans_reconstructed >= np.sum(nan_in_lst_stack):
                print("  WARNING: Final reconstructed LST appears to carry over NaNs from the input LST stack. Overall gap-filling may not be effective.")
        elif total_nan_in_reconstructed == 0 and total_nan_in_lst > 0:
             print("  INFO: Final reconstructed LST does not contain NaNs, suggesting pipeline might be performing gap-filling.")
    # --- END DIAGNOSTIC BLOCK FOR RECONSTRUCTION ---

    # 5. Save Reconstructed LST and Uncertainty Products for Test Data
    print("\nStep 5: Saving Outputs for Test Data")
    try:
        # Save reconstructed LST for each time step
        num_times = reconstructed_lst_test.shape[0]
        dates = test_data['common_dates']
        for t in range(num_times):
            date_str = dates[t].strftime('%Y%m%d')
            
            # Reconstructed LST
            recon_filename = os.path.join(config.RECONSTRUCTED_LST_PATH, f"LST_RECON_TEST_{date_str}.tif")
            utils.save_array_as_geotiff(
                data_array=reconstructed_lst_test[t, :, :],
                reference_geotiff_path=test_data['reference_grid_path'],
                output_path=recon_filename,
                nodata_value=np.nan # Or a specific nodata value if preferred
            )
            
            # Total Variance
            var_filename = os.path.join(config.UNCERTAINTY_MAPS_PATH, f"LST_TOTAL_VARIANCE_TEST_{date_str}.tif")
            utils.save_array_as_geotiff(
                data_array=total_variance_test[t, :, :],
                reference_geotiff_path=test_data['reference_grid_path'],
                output_path=var_filename,
                nodata_value=np.nan
            )
            
            # Confidence Intervals (Optional - can be large)
            # ci_low_filename = os.path.join(config.UNCERTAINTY_MAPS_PATH, f"LST_CI_LOWER_TEST_{date_str}.tif")
            # utils.save_array_as_geotiff(ci_lower_test[t,:,:], test_data['reference_grid_path'], ci_low_filename, nodata_value=np.nan)
            # ci_up_filename = os.path.join(config.UNCERTAINTY_MAPS_PATH, f"LST_CI_UPPER_TEST_{date_str}.tif")
            # utils.save_array_as_geotiff(ci_upper_test[t,:,:], test_data['reference_grid_path'], ci_up_filename, nodata_value=np.nan)
        print(f"Saved reconstructed LST and variance maps for test data to {config.RECONSTRUCTED_LST_PATH} and {config.UNCERTAINTY_MAPS_PATH}")

    except Exception as e:
        print(f"Error during saving outputs: {e}")
        import traceback
        traceback.print_exc()
        # Continue to evaluation even if saving fails for some reason

    # Create the 'model_predictions_for_eval' for fair evaluation
    # This represents the model's raw output before merging with observed data.
    print("\nPreparing model's raw predictions for evaluation purposes...")
    # Start with ATC predictions on test data
    model_predictions_for_eval = np.copy(atc_mean_predictions_test)
    
    # Add GP residuals where they are valid
    # If gp_mean_residuals_map_test is None or all NaN (e.g. GP failed/skipped), this won't add anything or add NaNs
    if gp_mean_residuals_map_test is not None:
        # Ensure gp_mean_residuals_map_test is not all NaNs before attempting to add
        if not np.all(np.isnan(gp_mean_residuals_map_test)):
            # Where gp_mean_residuals_map_test is NaN, adding it will result in NaN, which is fine.
            # Where atc_mean_predictions_test is NaN, result will be NaN.
            model_predictions_for_eval = atc_mean_predictions_test + gp_mean_residuals_map_test
        else:
            print("  GP mean residuals map is all NaN; using only ATC predictions for model evaluation output.")
            # model_predictions_for_eval already holds atc_mean_predictions_test
    else:
        print("  GP mean residuals map is None; using only ATC predictions for model evaluation output.")
        # model_predictions_for_eval already holds atc_mean_predictions_test

    # 6. Evaluate Model Performance on Test Data
    print("\nStep 6: Model Evaluation on Test Data")
    try:
        all_eval_metrics = evaluation.run_all_evaluations(
            model_predicted_lst=model_predictions_for_eval, # Use the model's raw predictions on test data
            observed_lst_clear=test_data['lst_stack'], # Original test LST with NaNs for clouds
            app_config=config
        )
        print("\nFinal Evaluation Metrics (Test Data):")
        for k, v in all_eval_metrics.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()

    # Visualize daily comparison stacks (Observed LST, Reconstructed LST, S2 RGB) for test data
    if reconstructed_lst_test.shape[0] > 0 and \
       test_data.get('lst_stack') is not None and \
       test_data.get('s2_reflectance_stack') is not None and \
       test_data.get('common_dates'):
        print("\nVisualizing daily comparison stacks for test data (Observed LST vs Predicted LST vs Reconstructed LST)...")
        try:
            # Assuming S2 bands are [B2, B3, B4, B8], so RGB indices are (B4=2, B3=1, B2=0)
            s2_rgb_indices_param = getattr(config, 'S2_RGB_INDICES', (2, 1, 0)) 
            max_days_plot_param = getattr(config, 'MAX_DAYS_FOR_DAILY_VISUALIZATION_PLOT', 10)

            utils.visualize_daily_stacks_comparison(
                lst_observed_stack=test_data['lst_stack'],
                model_predicted_lst_stack=model_predictions_for_eval, # ADDED: Pass model's direct predictions
                reconstructed_lst_stack=reconstructed_lst_test,
                era5_stack=test_data['era5_stack'], # ADDED: Pass ERA5 stack
                s2_reflectance_stack=test_data['s2_reflectance_stack'],
                ndvi_stack=test_data.get('ndvi_stack'), # Pass NDVI stack, could be None
                common_dates=test_data['common_dates'],
                output_base_dir=config.OUTPUT_DIR, 
                roi_name=roi_name,
                app_config=config, # Pass the config object
                s2_rgb_indices=s2_rgb_indices_param,
                max_days_to_plot=max_days_plot_param
            )
        except Exception as e:
            print(f"Error during daily comparison visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping daily comparison visualization as not all required data stacks are available.")
    
    print("\nDELAG LST Reconstruction Pipeline Completed.")

    # Save the configuration used for this run
    print("\nStep 7: Saving Run Configuration")
    try:
        config_dict = {key: getattr(config, key) for key in dir(config) if not key.startswith('__') and not callable(getattr(config, key))}
        config_filename = os.path.join(config.OUTPUT_DIR, 'run_config.json')
        with open(config_filename, 'w') as f:
            json.dump(config_dict, f, indent=4, default=str) # Use default=str to handle non-serializable types like Path objects if any
        print(f"Run configuration saved to {config_filename}")
    except Exception as e:
        print(f"Error saving run configuration: {e}")
        import traceback
        traceback.print_exc()

    # Save data split information
    print("\nStep 8: Saving Data Split Information")
    try:
        split_info = {
            'train_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in train_data['common_dates']],
            'test_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in test_data['common_dates']],
            'train_count': len(train_data['common_dates']),
            'test_count': len(test_data['common_dates']),
            'train_date_range': f"2015-01-01 to 2024-01-01",
            'test_date_range': f"2024-01-01 to 2025-01-01"
        }
        split_info_filename = os.path.join(config.OUTPUT_DIR, 'data_split_info.json')
        with open(split_info_filename, 'w') as f:
            json.dump(split_info, f, indent=4)
        print(f"Data split information saved to {split_info_filename}")
    except Exception as e:
        print(f"Error saving data split information: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # Before running this final reconstruction script, ensure that:
    # 1. All necessary libraries (numpy, pandas, torch, gpytorch, rasterio, scikit-learn, tqdm) are installed.
    # 2. The following modular pipeline scripts have been executed in order:
    #    a. preprocess_data.py --roi-name <ROI_NAME>
    #    b. train_atc.py --roi-name <ROI_NAME>
    #    c. predict_atc.py --roi-name <ROI_NAME>
    #    d. train_gp.py --roi-name <ROI_NAME> (if GP is enabled)
    #    e. predict_gp.py --roi-name <ROI_NAME> (if GP is enabled)
    # 3. config.ROI_NAME is set correctly in config.py
    # 4. All intermediate outputs exist in the expected directory structure:
    #    - output/data_train/<ROI_NAME>/
    #    - output/data_test/<ROI_NAME>/
    #    - output/output_models/<ROI_NAME>/
    #    - output/atc_predicted_data/<ROI_NAME>/
    #    - output/gp_predicted_data/<ROI_NAME>/ (if GP enabled)

    # This main script loads all pre-generated predictions and performs final reconstruction.
    main() 