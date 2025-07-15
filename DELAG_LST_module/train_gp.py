#!/usr/bin/env python3
"""
Train GP Model Script for DELAG LST Reconstruction Pipeline

This script:
1. Loads preprocessed training data
2. Loads ATC predictions on training data  
3. Trains the GP model for residuals
4. Saves the trained GP model and training metrics

Usage:
    python train_gp.py --roi-name <ROI_NAME> [--spatial-sample-percentage <percentage>]
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


def setup_output_directories(roi_name):
    """
    Set up output directories for GP model training.
    
    Args:
        roi_name (str): Name of the ROI being processed
        
    Returns:
        str: Path to model weights directory
    """
    # Create model weights directory
    model_weights_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, "output_models")
    os.makedirs(model_weights_dir, exist_ok=True)
    
    # Update config paths
    config.MODEL_WEIGHTS_PATH = model_weights_dir
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR_BASE, roi_name)
    
    return model_weights_dir


def load_training_data(roi_name):
    """
    Load preprocessed training data.
    
    Args:
        roi_name (str): Name of the ROI
        
    Returns:
        dict: Preprocessed training data
    """
    print("Loading preprocessed training data...")
    
    # Load training data  
    train_data_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, "data_train")
    if not os.path.exists(train_data_dir):
        raise FileNotFoundError(f"Training data directory not found: {train_data_dir}")
    
    train_data = utils.load_processed_data(train_data_dir)
    print(f"Training data loaded from: {train_data_dir}")
    print(f"  Available keys in loaded data: {list(train_data.keys())}")  # Diagnostic print
    print(f"  LST stack shape: {train_data['lst_stack'].shape}")
    print(f"  S2 reflectance stack shape: {train_data['s2_reflectance_stack'].shape}")
    
    return train_data


def load_atc_predictions_on_training_data(roi_name):
    """
    Load ATC predictions on training data from saved files.
    
    Args:
        roi_name (str): Name of the ROI
        
    Returns:
        np.ndarray: ATC predictions on training data (time, height, width)
    """
    print("Loading ATC predictions from saved files...")
    
    # Load ATC predictions from the saved directory for the training split
    atc_predictions_dir = os.path.join(config.OUTPUT_DIR_BASE, roi_name, "atc_predicted_data_train")
    
    if not os.path.exists(atc_predictions_dir):
        raise FileNotFoundError(
            f"ATC predictions directory for training data not found: {atc_predictions_dir}. "
            f"Run 'python predict_atc.py --roi_name {roi_name} --data_split train' first."
        )
    
    # Load predictions using the utils function
    try:
        # Assuming mean_predictions.npy and variance_predictions.npy exist in the directory
        mean_path = os.path.join(atc_predictions_dir, "mean_predictions.npy")
        if not os.path.exists(mean_path):
            raise FileNotFoundError(f"mean_predictions.npy not found in {atc_predictions_dir}")
            
        atc_mean_predictions_train = np.load(mean_path)
        print(f"ATC predictions loaded from: {mean_path}")
        print(f"ATC predictions shape: {atc_mean_predictions_train.shape}")
        return atc_mean_predictions_train
        
    except Exception as e:
        print(f"Error loading ATC predictions: {e}")
        raise


def apply_spatial_masking_for_gp(train_data, model_weights_dir):
    """
    Apply spatial masking for GP training, using existing mask or creating a new one.
    
    Args:
        train_data (dict): Training data dictionary
        model_weights_dir (str): Directory where models are saved
    """
    print("\nApplying spatial masking for GP training...")
    
    # First, try to load existing spatial mask from ATC training
    mask_filepath = os.path.join(model_weights_dir, 'spatial_training_mask.npy')
    
    if os.path.exists(mask_filepath):
        training_pixel_mask = np.load(mask_filepath)
        print(f"Loaded existing spatial training mask: {np.sum(training_pixel_mask)} pixels selected")
    else:
        # Create spatial sampling mask for training if it doesn't exist
        print("No existing spatial mask found. Creating new spatial mask for GP training...")
        try:
            _, height, width = train_data['lst_stack'].shape
            
            if config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE < 1.0:
                num_pixels_total = height * width
                num_pixels_to_sample = int(num_pixels_total * config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE)
                
                min_pixels = getattr(config, 'MIN_PIXELS_FOR_SPATIAL_SAMPLING', 100)
                num_pixels_to_sample = max(num_pixels_to_sample, min_pixels)
                num_pixels_to_sample = min(num_pixels_to_sample, num_pixels_total)
                
                all_indices = np.arange(num_pixels_total)
                np.random.shuffle(all_indices)
                
                sampled_indices = all_indices[:num_pixels_to_sample]
                
                training_pixel_mask = np.zeros(num_pixels_total, dtype=bool)
                training_pixel_mask[sampled_indices] = True
                training_pixel_mask = training_pixel_mask.reshape((height, width))
                
                print(f"Spatially sampling {np.sum(training_pixel_mask)} pixels ({config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE*100:.2f}%) for GP training.")
            else:
                training_pixel_mask = np.ones((height, width), dtype=bool)
                print("Using all pixels for GP training (SPATIAL_TRAINING_SAMPLE_PERCENTAGE is 1.0).")

            # Save the mask to the models directory
            np.save(mask_filepath, training_pixel_mask)
            print(f"Saved spatial training mask to {mask_filepath}")
            
        except Exception as e:
            print(f"Error during spatial sampling: {e}")
            # Fallback to all pixels
            _, height, width = train_data['lst_stack'].shape
            training_pixel_mask = np.ones((height, width), dtype=bool)
            print("Falling back to using all pixels for GP training.")
    
    # Add the mask to training data
    train_data['training_pixel_mask'] = training_pixel_mask
    return train_data


def train_gp_model_pipeline(train_data, atc_predictions_train, roi_name):
    """
    Train the GP model and save results.
    
    Args:
        train_data (dict): Preprocessed training data
        atc_predictions_train (np.ndarray): ATC predictions on training data
        roi_name (str): Name of the ROI
    """
    print("\nTraining GP model...")
    start_time = time.time()
    
    # Train and save GP model
    gp_model.train_and_save_gp_model(
        train_data, atc_predictions_train, config
    )
    
    end_time = time.time()
    training_duration = timedelta(seconds=end_time - start_time)
    print(f"GP model training completed in: {training_duration}")
    
    # Plot training loss if available
    plot_gp_training_loss(roi_name)


def plot_gp_training_loss(roi_name):
    """
    Plot GP training loss curves.
    
    Args:
        roi_name (str): Name of the ROI
    """
    print("Plotting GP training loss...")
    
    try:
        gp_model_filepath = os.path.join(config.MODEL_WEIGHTS_PATH, config.GP_MODEL_WEIGHT_FILENAME)
        gp_interval_losses = gp_model.load_gp_interval_losses(gp_model_filepath)
        
        if gp_interval_losses:
            # Calculate epoch intervals for plotting
            num_gp_intervals = len(gp_interval_losses)
            gp_loss_logging_interval = getattr(config, 'GP_LOSS_LOGGING_INTERVAL', 10)
            total_gp_epochs = config.GP_EPOCHS_INITIAL + config.GP_EPOCHS_FINAL
            gp_epoch_ticks = [(i + 1) * gp_loss_logging_interval for i in range(num_gp_intervals)]
            
            # Plot training loss
            utils.plot_mean_gp_loss_over_intervals(
                mean_interval_losses=gp_interval_losses,
                epoch_intervals_x_axis=gp_epoch_ticks,
                output_dir=config.OUTPUT_DIR,
                roi_name=roi_name,
                loss_logging_interval=gp_loss_logging_interval
            )
            print("GP training loss plot saved.")
        else:
            print("No GP interval losses found for plotting.")
            
    except Exception as e:
        print(f"Error plotting GP training loss: {e}")


def main():
    """Main function for GP model training."""
    parser = argparse.ArgumentParser(description="Train GP Model for DELAG LST Reconstruction")
    parser.add_argument(
        "--roi_name", 
        required=True, 
        dest='roi_name',
        help="Name of the ROI to process"
    )
    parser.add_argument(
        "--spatial-sample-percentage",
        type=float,
        default=None,
        help="Override spatial sampling percentage for training (0.0-1.0)"
    )
    
    args = parser.parse_args()
    roi_name = args.roi_name
    
    print("="*60)
    print(f"DELAG GP Model Training Pipeline")
    print(f"ROI: {roi_name}")
    print("="*60)
    
    try:
        # Override spatial sampling if provided
        if args.spatial_sample_percentage is not None:
            if 0.0 <= args.spatial_sample_percentage <= 1.0:
                config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE = args.spatial_sample_percentage
                print(f"Spatial sampling percentage set to: {args.spatial_sample_percentage}")
            else:
                raise ValueError("Spatial sample percentage must be between 0.0 and 1.0")
        
        # Setup directories
        model_weights_dir = setup_output_directories(roi_name)
        print(f"Model weights will be saved to: {model_weights_dir}")
        
        # Check if GP model is enabled
        if not config.USE_GP_MODEL:
            print("GP model is disabled in config (USE_GP_MODEL=False). Exiting.")
            return
        
        # Load training data
        train_data = load_training_data(roi_name)
        
        # Apply spatial masking for GP training
        train_data = apply_spatial_masking_for_gp(train_data, model_weights_dir)
        
        # Load ATC predictions on training data
        atc_predictions_train = load_atc_predictions_on_training_data(roi_name)
        
        # Train GP model
        train_gp_model_pipeline(train_data, atc_predictions_train, roi_name)
        
        print("\n" + "="*60)
        print("GP Model Training Pipeline Completed Successfully!")
        print(f"Model saved to: {os.path.join(model_weights_dir, config.GP_MODEL_WEIGHT_FILENAME)}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during GP model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 