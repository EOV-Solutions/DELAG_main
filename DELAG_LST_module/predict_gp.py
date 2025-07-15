#!/usr/bin/env python3
"""
Predict GP Residuals Script for DELAG LST Reconstruction Pipeline

This script:
1. Loads preprocessed test data
2. Loads pre-generated ATC predictions on test data
3. Loads trained GP model
4. Predicts residuals on test data using the GP model
5. Saves GP residual predictions

Usage:
    python predict_gp.py --roi-name <ROI_NAME>
"""

import os
import sys
import argparse
import numpy as np
import time
from datetime import timedelta

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
import utils
import gp_model


def setup_output_directories(roi_name, data_split):
    """
    Set up output directories for GP predictions.
    
    Args:
        roi_name (str): Name of the ROI being processed
        data_split (str): Data split ('train' or 'test')
        
    Returns:
        tuple: (model_weights_dir, predictions_output_dir)
    """
    # Model weights directory (where trained GP model is stored)
    model_weights_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, "output_models")
    
    # GP predictions output directory
    prediction_dir_suffix = '' if data_split == 'test' else '_train'
    predictions_output_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f"gp_predicted_data{prediction_dir_suffix}")
    os.makedirs(predictions_output_dir, exist_ok=True)
    
    # Update config paths
    config.MODEL_WEIGHTS_PATH = model_weights_dir
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR_BASE, roi_name)
    
    return model_weights_dir, predictions_output_dir


def load_data(roi_name, data_split):
    """
    Load preprocessed data for a given split.
    
    Args:
        roi_name (str): Name of the ROI
        data_split (str): Data split ('train' or 'test')
        
    Returns:
        dict: Preprocessed data
    """
    print(f"Loading preprocessed {data_split} data...")
    
    # Load data for the specified split
    data_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f"data_{data_split}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_split.capitalize()} data directory not found: {data_dir}")
    
    data = utils.load_processed_data(data_dir)
    print(f"{data_split.capitalize()} data loaded from: {data_dir}")
    print(f"  LST stack shape: {data['lst_stack'].shape}")
    print(f"  S2 reflectance stack shape: {data['s2_reflectance_stack'].shape}")
    
    return data


def load_atc_predictions(roi_name, data_split):
    """
    Load pre-generated ATC predictions on a given data split.
    
    Args:
        roi_name (str): Name of the ROI
        data_split (str): Data split ('train' or 'test')
        
    Returns:
        tuple: (atc_mean_predictions, atc_variance_predictions)
    """
    print(f"Loading ATC predictions on {data_split} data...")
    
    # Load ATC predictions from the correct directory structure
    prediction_dir_suffix = '' if data_split == 'test' else '_train'
    atc_predictions_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, f"atc_predicted_data{prediction_dir_suffix}")
    
    if not os.path.exists(atc_predictions_dir):
        raise FileNotFoundError(f"ATC predictions directory not found: {atc_predictions_dir}. Run predict_atc.py --data_split {data_split} first.")
    
    atc_predictions = utils.load_atc_predictions(atc_predictions_dir)
    atc_mean_predictions = atc_predictions['mean_predictions']
    atc_variance_predictions = atc_predictions['variance_predictions']
    
    print(f"ATC predictions loaded from: {atc_predictions_dir}")
    print(f"  Mean predictions shape: {atc_mean_predictions.shape}")
    print(f"  Variance predictions shape: {atc_variance_predictions.shape}")
    
    return atc_mean_predictions, atc_variance_predictions


def predict_gp_residuals_pipeline(data_to_predict, atc_mean_predictions, roi_name):
    """
    Load trained GP model and predict residuals on the given data.
    
    Args:
        data_to_predict (dict): Preprocessed data for prediction
        atc_mean_predictions (np.ndarray): ATC predictions on the data
        roi_name (str): Name of the ROI
        
    Returns:
        tuple: (gp_mean_residuals, gp_variance_residuals)
    """
    print(f"\nPredicting GP residuals on {data_to_predict.get('split_name', 'data')}...")
    start_time = time.time()
    
    # Check if trained GP model exists
    gp_model_filepath = os.path.join(config.MODEL_WEIGHTS_PATH, config.GP_MODEL_WEIGHT_FILENAME)
    
    # Load the spatial mask used during training
    mask_filepath = os.path.join(config.MODEL_WEIGHTS_PATH, 'spatial_training_mask.npy')
    if os.path.exists(mask_filepath):
        spatial_mask = np.load(mask_filepath)
        print(f"Loaded spatial training mask for GP prediction: {np.sum(spatial_mask)} pixels will be predicted")
    else:
        print("Warning: Spatial training mask not found. Predicting for all pixels.")
        spatial_mask = None

    if not os.path.exists(gp_model_filepath):
        print(f"Trained GP model not found at: {gp_model_filepath}")
        print("This might happen if GP training was skipped or failed.")
        print("Returning zero residuals (fallback to ATC-only predictions).")
        
        # Return zero residuals as fallback
        gp_mean_residuals = np.zeros_like(atc_mean_predictions)
        gp_variance_residuals = np.zeros_like(atc_mean_predictions)
        return gp_mean_residuals, gp_variance_residuals
    
    # Load GP model and predict residuals, passing the mask
    gp_mean_residuals, gp_variance_residuals = gp_model.load_and_predict_gp_residuals(
        data_to_predict, atc_mean_predictions, config, prediction_mask=spatial_mask
    )
    
    end_time = time.time()
    prediction_duration = timedelta(seconds=end_time - start_time)
    print(f"GP residual prediction completed in: {prediction_duration}")
    print(f"  Mean residuals shape: {gp_mean_residuals.shape}")
    print(f"  Variance residuals shape: {gp_variance_residuals.shape}")
    
    return gp_mean_residuals, gp_variance_residuals


def save_gp_predictions(gp_mean_residuals, gp_variance_residuals, data, predictions_output_dir, roi_name):
    """
    Save GP residual predictions and metadata.
    
    Args:
        gp_mean_residuals (np.ndarray): GP mean residual predictions
        gp_variance_residuals (np.ndarray): GP variance residual predictions
        data (dict): Data containing metadata
        predictions_output_dir (str): Directory to save predictions
        roi_name (str): Name of the ROI
    """
    print(f"\nSaving GP predictions to: {predictions_output_dir}")
    
    # Save GP residual predictions
    np.save(os.path.join(predictions_output_dir, 'gp_mean_residuals.npy'), gp_mean_residuals)
    np.save(os.path.join(predictions_output_dir, 'gp_variance_residuals.npy'), gp_variance_residuals)
    
    # Save metadata
    metadata = {
        'roi_name': roi_name,
        'prediction_shape': gp_mean_residuals.shape,
        'data_type': 'gp_residuals',
        'mean_residuals_file': 'gp_mean_residuals.npy',
        'variance_residuals_file': 'gp_variance_residuals.npy',
        'coordinate_system': data.get('coordinate_system', 'Unknown'),
        'spatial_resolution': data.get('spatial_resolution', 'Unknown'),
        'temporal_resolution': data.get('temporal_resolution', 'Unknown'),
        'creation_timestamp': utils.get_current_timestamp()
    }
    
    utils.save_metadata(metadata, predictions_output_dir)
    
    print("GP predictions saved successfully:")
    print(f"  Mean residuals: gp_mean_residuals.npy")
    print(f"  Variance residuals: gp_variance_residuals.npy") 
    print(f"  Metadata: metadata.json")


def generate_summary_statistics(gp_mean_residuals, gp_variance_residuals, roi_name):
    """
    Generate and print summary statistics for GP predictions.
    
    Args:
        gp_mean_residuals (np.ndarray): GP mean residual predictions
        gp_variance_residuals (np.ndarray): GP variance residual predictions
        roi_name (str): Name of the ROI
    """
    print(f"\n=== GP Residual Prediction Summary for {roi_name} ===")
    
    # Calculate statistics (ignoring NaN values)
    mean_residual_stats = {
        'mean': np.nanmean(gp_mean_residuals),
        'std': np.nanstd(gp_mean_residuals),
        'min': np.nanmin(gp_mean_residuals),
        'max': np.nanmax(gp_mean_residuals),
        'valid_pixels': np.sum(~np.isnan(gp_mean_residuals))
    }
    
    variance_residual_stats = {
        'mean': np.nanmean(gp_variance_residuals),
        'std': np.nanstd(gp_variance_residuals),
        'min': np.nanmin(gp_variance_residuals),
        'max': np.nanmax(gp_variance_residuals),
        'valid_pixels': np.sum(~np.isnan(gp_variance_residuals))
    }
    
    print("GP Mean Residuals Statistics:")
    print(f"  Mean: {mean_residual_stats['mean']:.4f} K")
    print(f"  Std:  {mean_residual_stats['std']:.4f} K")
    print(f"  Min:  {mean_residual_stats['min']:.4f} K")
    print(f"  Max:  {mean_residual_stats['max']:.4f} K")
    print(f"  Valid pixels: {mean_residual_stats['valid_pixels']:,}")
    
    print("\nGP Variance Residuals Statistics:")
    print(f"  Mean: {variance_residual_stats['mean']:.4f} K²")
    print(f"  Std:  {variance_residual_stats['std']:.4f} K²")
    print(f"  Min:  {variance_residual_stats['min']:.4f} K²")
    print(f"  Max:  {variance_residual_stats['max']:.4f} K²")
    print(f"  Valid pixels: {variance_residual_stats['valid_pixels']:,}")


def main():
    """Main function for GP residual prediction."""
    parser = argparse.ArgumentParser(description="Predict GP Residuals for DELAG LST Reconstruction")
    parser.add_argument(
        "--roi_name", 
        required=True, 
        dest='roi_name',
        help="Name of the ROI to process"
    )
    parser.add_argument(
        '--data_split',
        type=str,
        default='train',
        choices=['train', 'test'],
        help="Data split to generate predictions for: 'train' or 'test'. Defaults to 'test'."
    )
    
    args = parser.parse_args()
    roi_name = args.roi_name
    data_split = args.data_split
    
    print("="*60)
    print(f"DELAG GP Residual Prediction Pipeline")
    print(f"ROI: {roi_name} ({data_split.upper()})")
    print("="*60)
    
    try:
        # Setup directories
        model_weights_dir, predictions_output_dir = setup_output_directories(roi_name, data_split)
        print(f"Model weights directory: {model_weights_dir}")
        print(f"Predictions output directory: {predictions_output_dir}")
        
        # Check if GP model is enabled
        if not config.USE_GP_MODEL:
            print("GP model is disabled in config (USE_GP_MODEL=False). Exiting.")
            return
        
        # Load data for the specified split
        data_to_predict = load_data(roi_name, data_split)
        data_to_predict['split_name'] = data_split # For logging inside pipeline
        
        # Load ATC predictions on the corresponding data split
        atc_mean_predictions, atc_variance_predictions = load_atc_predictions(roi_name, data_split)
        
        # Predict GP residuals
        gp_mean_residuals, gp_variance_residuals = predict_gp_residuals_pipeline(
            data_to_predict, atc_mean_predictions, roi_name
        )
        
        # Save GP predictions
        save_gp_predictions(
            gp_mean_residuals, gp_variance_residuals, 
            data_to_predict, predictions_output_dir, roi_name
        )
        
        # Generate summary statistics
        generate_summary_statistics(gp_mean_residuals, gp_variance_residuals, roi_name)
        
        print("\n" + "="*60)
        print("GP Residual Prediction Pipeline Completed Successfully!")
        print(f"Predictions saved to: {predictions_output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during GP residual prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 