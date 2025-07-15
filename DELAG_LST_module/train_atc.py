"""
Script to train ATC models using preprocessed data.
This script will:
1. Load preprocessed training data from output/ROI/data_train/
2. Train ATC models and collect snapshots
3. Save trained models to output_models/ROI/
4. Generate training loss plots
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
    """Main ATC training function."""
    print("Starting DELAG ATC Training Pipeline...")
    
    # Set config attributes from args
    config.ROI_NAME = args.roi_name
    
    # Set up directories
    # The training data follows the nested ROI structure
    train_data_dir = os.path.join(config.OUTPUT_DIR_BASE, config.ROI_NAME, 'data_train')
    # The models are saved in the non-nested structure
    models_output_dir = os.path.join(config.OUTPUT_DIR_BASE, config.ROI_NAME, "output_models")
    
    # Create models output directory
    os.makedirs(models_output_dir, exist_ok=True)
    
    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 1. Load Pre-processed Training Data
    print("\nStep 1: Loading Pre-processed Training Data")
    try:
        train_data = utils.load_processed_data(train_data_dir)
        print("Training data loading completed.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading training data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Create spatial sampling mask for training
    print("\nApplying spatial sampling for ATC training...")
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
            
            print(f"Spatially sampling {np.sum(training_pixel_mask)} pixels ({config.SPATIAL_TRAINING_SAMPLE_PERCENTAGE*100:.2f}%) for ATC training.")
        else:
            training_pixel_mask = np.ones((height, width), dtype=bool)
            print("Using all pixels for ATC training (SPATIAL_TRAINING_SAMPLE_PERCENTAGE is 1.0).")

        train_data['training_pixel_mask'] = training_pixel_mask
        
        # Also save the mask to the models directory for the prediction script to use
        mask_filepath = os.path.join(models_output_dir, 'spatial_training_mask.npy')
        np.save(mask_filepath, training_pixel_mask)
        print(f"Saved spatial training mask to {mask_filepath}")
    except Exception as e:
        print(f"Error during spatial sampling: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Train ATC Model and Collect Snapshots
    print("\nStep 2: ATC Model Training and Snapshot Collection")
    try:
        # Train ATC models and save snapshots using TRAINING DATA
        print("  Training ATC models and collecting snapshots/losses on training data...")
        all_pixel_snapshots, interval_loss_maps = atc_model.train_and_collect_all_atc_snapshots(
            train_data, config
        )

        # Plot Mean ATC Training and Validation Loss
        if interval_loss_maps and 'train' in interval_loss_maps and 'val' in interval_loss_maps:
            train_loss_maps_array = interval_loss_maps['train']
            val_loss_maps_array = interval_loss_maps['val']

            mean_train_losses = np.nanmean(train_loss_maps_array, axis=(1, 2))
            mean_val_losses = np.nanmean(val_loss_maps_array, axis=(1, 2))
            
            num_intervals = train_loss_maps_array.shape[0]
            loss_logging_interval = getattr(config, 'ATC_LOSS_LOGGING_INTERVAL', 100)
            epoch_ticks = [(i + 1) * loss_logging_interval for i in range(num_intervals)]
            
            utils.plot_mean_atc_loss_over_intervals(
                mean_train_losses=list(mean_train_losses),
                mean_val_losses=list(mean_val_losses),
                epoch_intervals_x_axis=epoch_ticks,
                output_dir=models_output_dir,  # Save plots to ROI directory
                roi_name=train_data.get('roi_name', 'UnknownROI'),
                loss_logging_interval=loss_logging_interval
            )
        else:
            print("  Skipping ATC mean loss plot as interval_loss_maps dict is incomplete or empty.")

        # Define path for saving snapshots in models directory
        snapshots_filename = f"atc_snapshots_{train_data.get('roi_name', 'all')}.npz"
        snapshots_filepath = os.path.join(models_output_dir, snapshots_filename)
        
        print(f"  Saving ATC snapshots to {snapshots_filepath}...")
        # Get image dimensions from one of the stacks, e.g., LST stack
        _, height, width = train_data['lst_stack'].shape 
        atc_model.save_atc_snapshots(
            all_pixel_snapshots, 
            snapshots_filepath,
            image_height=height,
            image_width=width,
            num_snapshots_expected=config.ATC_ENSEMBLE_SNAPSHOTS
        )
        print("  ATC snapshots saved.")

    except Exception as e:
        print(f"Error during ATC model training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("ATC model training completed successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DELAG ATC Training Pipeline")
    parser.add_argument('--roi_name', type=str, required=True, help="Name of the Region of Interest (ROI).")
    
    args = parser.parse_args()
    main(args) 