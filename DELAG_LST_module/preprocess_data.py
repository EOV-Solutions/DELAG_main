"""
Script to run the data preprocessing and splitting part of the DELAG LST pipeline.
This script will:
1. Preprocess all input data (LST, ERA5, S2, etc.).
2. Split the data into training (2015-2024) and testing (2024-2025) sets.
3. Save the processed training and testing data dictionaries to the output folder.
"""
import numpy as np
import torch
import os
import json
import pandas as pd
from typing import Dict, Tuple
import argparse

# Import project modules
import config
import utils
import data_preprocessing

def split_data_by_date(
    preprocessed_data: Dict, 
    train_start_date: str, 
    train_end_date: str, 
    test_start_date: str, 
    test_end_date: str
) -> Tuple[Dict, Dict]:
    """
    Split data into train and test sets based on dates.
    This implements the same temporal split used in data_amount_analysis.py.
    
    Args:
        preprocessed_data: Dictionary containing preprocessed data
        train_start_date (str): Start date for training data (YYYY-MM-DD).
        train_end_date (str): End date for training data (YYYY-MM-DD).
        test_start_date (str): Start date for test data (YYYY-MM-DD).
        test_end_date (str): End date for test data (YYYY-MM-DD).
        
    Returns:
        Tuple of (train_data, test_data) dictionaries
    """
    print("Splitting data by date...")
    
    common_dates = preprocessed_data['common_dates']
    
    # Define date boundaries from arguments
    train_start = pd.to_datetime(train_start_date)
    train_end = pd.to_datetime(train_end_date)
    test_start = pd.to_datetime(test_start_date)
    test_end = pd.to_datetime(test_end_date)
    
    # Find indices for train and test sets
    train_indices = []
    test_indices = []
    
    for i, date in enumerate(common_dates):
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        if train_start <= date < train_end:
            train_indices.append(i)
        if test_start <= date < test_end:
            test_indices.append(i)
    
    print(f"Train dates: {len(train_indices)} ({train_start} to {train_end})")
    print(f"Test dates: {len(test_indices)} ({test_start} to {test_end})")
    
    if len(train_indices) == 0:
        raise ValueError("No training data found in the specified date range")
    if len(test_indices) == 0:
        raise ValueError("No test data found in the specified date range")
    
    # Create train data subset
    train_data = {}
    for key, value in preprocessed_data.items():
        if key == 'common_dates':
            train_data[key] = [common_dates[i] for i in train_indices]
        elif key in ['lst_stack', 'era5_stack', 's2_reflectance_stack', 'doy_stack']:
            if hasattr(value, 'shape') and len(value.shape) >= 1:
                train_data[key] = value[train_indices]
            else:
                train_data[key] = value
        elif key == 'ndvi_stack' and value is not None:
            train_data[key] = value[train_indices]
        else:
            train_data[key] = value
    
    # Create test data subset
    test_data = {}
    for key, value in preprocessed_data.items():
        if key == 'common_dates':
            test_data[key] = [common_dates[i] for i in test_indices]
        elif key in ['lst_stack', 'era5_stack', 's2_reflectance_stack', 'doy_stack']:
            if hasattr(value, 'shape') and len(value.shape) >= 1:
                test_data[key] = value[test_indices]
            else:
                test_data[key] = value
        elif key == 'ndvi_stack' and value is not None:
            test_data[key] = value[test_indices]
        else:
            test_data[key] = value
    
    return train_data, test_data


def main(args):
    """Main data preprocessing and saving function."""
    print("Starting DELAG Data Preprocessing Pipeline...")
    
    # Set config attributes from args
    config.ROI_NAME = args.roi_name
    print(f"ROI name: {config.ROI_NAME}")
    # Override OUTPUT_DIR to be based on ROI name, not a timestamp
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR_BASE, config.ROI_NAME)

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 0. Create output directories (idempotent)
    utils.create_output_directories(config)

    # 1. Data Preprocessing
    print("\nStep 1: Data Preprocessing")
    try:
        preprocessed_data = data_preprocessing.preprocess_all_data(config)
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return # Stop pipeline if preprocessing fails
    print("Data preprocessing completed.")

    # --- START DIAGNOSTIC BLOCK FOR PREPROCESSED DATA ---
    if 'era5_stack' in preprocessed_data:
        era5_nan_count = np.isnan(preprocessed_data['era5_stack']).sum()
        era5_total_count = preprocessed_data['era5_stack'].size
        era5_nan_percentage = (era5_nan_count / era5_total_count) * 100 if era5_total_count > 0 else 0
        print(f"  DIAGNOSTIC (PREPROCESSING): ERA5 stack has {era5_nan_count} NaNs out of {era5_total_count} values ({era5_nan_percentage:.2f}%).")
        if era5_nan_percentage > 0:
            print("    INFO: These NaNs in ERA5 will likely propagate to ATC predictions if the 'b' coefficient is non-zero.")
    if 's2_reflectance_stack' in preprocessed_data:
        s2_nan_count = np.isnan(preprocessed_data['s2_reflectance_stack']).sum()
        s2_total_count = preprocessed_data['s2_reflectance_stack'].size
        s2_nan_percentage = (s2_nan_count / s2_total_count) * 100 if s2_total_count > 0 else 0
        print(f"  DIAGNOSTIC (PREPROCESSING): S2 reflectance stack has {s2_nan_count} NaNs out of {s2_total_count} values ({s2_nan_percentage:.2f}%).")
    # --- END DIAGNOSTIC BLOCK FOR PREPROCESSED DATA ---

    # 2. Split data into train and test sets
    print("\nStep 2: Splitting data into train and test sets")
    try:
        train_data, test_data = split_data_by_date(
            preprocessed_data,
            train_start_date=args.train_start,
            train_end_date=args.train_end,
            test_start_date=args.test_start,
            test_end_date=args.test_end
        )
    except Exception as e:
        print(f"Error during data splitting: {e}")
        import traceback
        traceback.print_exc()
        return
    print("Data splitting completed.")

    # 3: Save processed train and test data
    print("\nStep 3: Saving processed train and test data")
    try:
        # Define base directories for train and test data
        train_data_dir = os.path.join(config.OUTPUT_DIR, 'data_train')
        test_data_dir = os.path.join(config.OUTPUT_DIR, 'data_test')
        
        # Save training data
        utils.save_processed_data(
            data_dict=train_data,
            output_dir=train_data_dir,
        )
        
        # Save test data
        utils.save_processed_data(
            data_dict=test_data,
            output_dir=test_data_dir,
        )
        print("Processed data saving completed.")
    except Exception as e:
        print(f"Error during saving of processed data: {e}")
        import traceback
        traceback.print_exc()

    # Save data split information
    print("\nStep 4: Saving Data Split Information")
    try:
        split_info = {
            'train_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in train_data['common_dates']],
            'test_dates': [date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date) for date in test_data['common_dates']],
            'train_count': len(train_data['common_dates']),
            'test_count': len(test_data['common_dates']),
            'train_date_range': f"{args.train_start} to {args.train_end}",
            'test_date_range': f"{args.test_start} to {args.test_end}"
        }
        split_info_filename = os.path.join(config.OUTPUT_DIR, 'data_split_info.json')
        with open(split_info_filename, 'w') as f:
            json.dump(split_info, f, indent=4)
        print(f"Data split information saved to {split_info_filename}")
    except Exception as e:
        print(f"Error saving data split information: {e}")
        import traceback
        traceback.print_exc()

    print("\nDELAG Data Preprocessing Pipeline Completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DELAG Data Preprocessing Pipeline")
    parser.add_argument('--roi_name', type=str, required=True, help="Name of the Region of Interest (ROI). This name is used for data loading and output directory naming.")
    parser.add_argument('--train_start', type=str, default='2015-01-01', help="Start date for training data (YYYY-MM-DD).")
    parser.add_argument('--train_end', type=str, default='2021-01-01', help="End date for training data (YYYY-MM-DD).")
    parser.add_argument('--test_start', type=str, default='2021-01-01', help="Start date for test data (YYYY-MM-DD).")
    parser.add_argument('--test_end', type=str, default='2023-01-01', help="End date for test data (YYYY-MM-DD).")
    
    args = parser.parse_args()
    main(args) 