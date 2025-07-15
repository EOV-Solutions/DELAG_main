"""
Script to generate ATC predictions using trained models.
This script will:
1. Load preprocessed test data from output/ROI/data_test/
2. Load trained ATC models from output_models/ROI/
3. Generate ATC predictions on test data
4. Save predictions to output/ROI/atc_predicted_data/
"""
import numpy as np
import torch
import os
import argparse
from typing import Dict

# Import project modules
import config
import utils
import atc_model

def main(args):
    """Main ATC prediction function."""
    print("Starting DELAG ATC Prediction Pipeline...")
    
    # Set config attributes from args
    config.ROI_NAME = args.roi_name
    
    # Set up directories based on data split
    input_data_dir = os.path.join(config.OUTPUT_DIR_BASE, config.ROI_NAME, f'data_{args.data_split}')
    models_input_dir = os.path.join(config.OUTPUT_DIR_BASE, config.ROI_NAME, "output_models")
    
    # Naming convention for prediction output directory
    # 'atc_predicted_data' for test, 'atc_predicted_data_train' for train
    prediction_dir_suffix = '' if args.data_split == 'test' else '_train'
    atc_predictions_dir = os.path.join(config.OUTPUT_DIR_BASE, config.ROI_NAME, f'atc_predicted_data{prediction_dir_suffix}')
    
    # Create ATC predictions output directory
    os.makedirs(atc_predictions_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 1. Load Pre-processed Test Data
    print(f"\nStep 1: Loading Pre-processed '{args.data_split}' Data")
    try:
        data_to_predict = utils.load_processed_data(input_data_dir)
        print(f"'{args.data_split}' data loading completed.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading test data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Load ATC Model Snapshots and Spatial Mask
    print("\nStep 2: Loading ATC Model Snapshots and Spatial Mask")
    try:
        snapshots_filename = f"atc_snapshots_{data_to_predict.get('roi_name', 'all')}.npz"
        snapshots_filepath = os.path.join(models_input_dir, snapshots_filename)
        
        if not os.path.exists(snapshots_filepath):
            print(f"Error: ATC model snapshots not found at {snapshots_filepath}")
            print("Please run train_atc.py first to train the models.")
            return
            
        loaded_snapshots_data = atc_model.load_atc_snapshots(snapshots_filepath)
        print("ATC snapshots loaded successfully.")

        # Load the spatial mask
        mask_filepath = os.path.join(models_input_dir, 'spatial_training_mask.npy')
        if os.path.exists(mask_filepath):
            spatial_mask = np.load(mask_filepath)
            print(f"Loaded spatial training mask from {mask_filepath}")
        else:
            print("Warning: Spatial training mask not found. Predicting for all pixels.")
            spatial_mask = None

    except Exception as e:
        print(f"Error loading ATC snapshots or mask: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Generate ATC Predictions on Test Data
    print(f"\nStep 3: Generating ATC Predictions on '{args.data_split}' Data")
    try:
        # For prediction, we use the TEST set DOY and ERA5 data
        doy_for_prediction = data_to_predict["doy_stack"]
        era5_for_prediction = data_to_predict["era5_stack"]

        atc_mean_predictions_test, atc_variance_test = atc_model.predict_atc_from_loaded_snapshots(
            loaded_snapshots_data,
            doy_for_prediction_numpy=doy_for_prediction,
            era5_for_prediction_numpy=era5_for_prediction,
            app_config=config,
            prediction_mask=spatial_mask
        )
        
        print("ATC predictions generated successfully.")
        
    except Exception as e:
        print(f"Error during ATC prediction: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Save ATC Predictions
    print("\nStep 4: Saving ATC Predictions")
    try:
        utils.save_atc_predictions(atc_mean_predictions_test, atc_variance_test, atc_predictions_dir)
        
        # Save metadata about the predictions
        prediction_metadata = {
            'roi_name': config.ROI_NAME,
            'test_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) 
                          for date in data_to_predict['common_dates']],
            'prediction_shape': list(atc_mean_predictions_test.shape),
            'variance_shape': list(atc_variance_test.shape),
            'model_snapshots_used': snapshots_filename
        }
        
        import json
        metadata_path = os.path.join(atc_predictions_dir, 'prediction_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(prediction_metadata, f, indent=4)
        print(f"Prediction metadata saved to {metadata_path}")
        
    except Exception as e:
        print(f"Error saving ATC predictions: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("ATC prediction pipeline completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DELAG ATC Prediction Pipeline")
    parser.add_argument('--roi_name', type=str, required=True, dest='roi_name', help="Name of the Region of Interest (ROI).")
    parser.add_argument(
        '--data_split', 
        type=str, 
        default='train', 
        choices=['train', 'test'],
        help="Data split to generate predictions for: 'train' or 'test'. Defaults to 'test'."
    )
    
    args = parser.parse_args()
    main(args) 