#!/usr/bin/env python3
"""
Example script to run the DELAG pipeline with ERA5 as the primary timeline.

This script demonstrates how to use the new ERA5-primary approach where:
1. ERA5 dates define the primary timeline
2. LST data is loaded for available dates and synthetic LST is created for missing dates
3. Reconstruction images will cover ALL ERA5 dates (not just LST dates)

Usage:
    python run_era5_primary_pipeline.py --roi_name YOUR_ROI_NAME
"""

import subprocess
import argparse
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("STDERR:", e.stderr)
        print("STDOUT:", e.stdout)
        return False

def main():
    parser = argparse.ArgumentParser(description="Run DELAG Pipeline with ERA5 Primary Timeline")
    parser.add_argument('--roi_name', type=str, required=True, help="Name of the ROI to process")
    parser.add_argument('--train_start', type=str, default='2015-01-01', help="Start date for training data")
    parser.add_argument('--train_end', type=str, default='2021-01-01', help="End date for training data")
    parser.add_argument('--test_start', type=str, default='2021-01-01', help="Start date for test data")
    parser.add_argument('--test_end', type=str, default='2023-01-01', help="End date for test data")
    parser.add_argument('--data_split', type=str, default='test', choices=['train', 'test'], help="Which data split to generate predictions for")
    parser.add_argument('--skip_training', action='store_true', help="Skip training steps (only run prediction/reconstruction)")
    
    args = parser.parse_args()
    
    print(f"""
🚀 DELAG ERA5-Primary Pipeline Runner
=====================================
ROI: {args.roi_name}
Timeline Mode: ERA5 Primary (with synthetic LST gap-filling)
Data Split for Predictions: {args.data_split}
Skip Training: {args.skip_training}

This pipeline will:
1. Use ERA5 dates as the primary timeline
2. Create synthetic LST for missing dates via temporal interpolation
3. Generate reconstruction images for ALL ERA5 dates
""")

    # Pipeline steps
    pipeline_commands = []
    
    # 1. Data Preprocessing (ERA5 Primary)
    preprocess_cmd = [
        'python', 'preprocess_data.py',
        '--roi_name', args.roi_name,
        '--timeline_mode', 'era5_primary',  # This is the key flag!
        '--train_start', args.train_start,
        '--train_end', args.train_end,
        '--test_start', args.test_start,
        '--test_end', args.test_end
    ]
    pipeline_commands.append((preprocess_cmd, "Data Preprocessing (ERA5 Primary Timeline)"))
    
    if not args.skip_training:
        # 2. Train ATC Model
        train_atc_cmd = [
            'python', 'train_atc.py',
            '--roi_name', args.roi_name
        ]
        pipeline_commands.append((train_atc_cmd, "Train ATC Model"))
        
        # 3. Train GP Model  
        train_gp_cmd = [
            'python', 'train_gp.py',
            '--roi_name', args.roi_name
        ]
        pipeline_commands.append((train_gp_cmd, "Train GP Model"))
    
    # 4. Predict ATC
    predict_atc_cmd = [
        'python', 'predict_atc.py',
        '--roi_name', args.roi_name,
        '--data_split', args.data_split
    ]
    pipeline_commands.append((predict_atc_cmd, f"Predict ATC on {args.data_split} data"))
    
    # 5. Predict GP
    predict_gp_cmd = [
        'python', 'predict_gp.py',
        '--roi_name', args.roi_name,
        '--data_split', args.data_split
    ]
    pipeline_commands.append((predict_gp_cmd, f"Predict GP on {args.data_split} data"))
    
    # 6. Final Reconstruction
    reconstruction_cmd = [
        'python', 'reconstruction.py',
        '--roi_name', args.roi_name,
        '--data_split', args.data_split
    ]
    pipeline_commands.append((reconstruction_cmd, f"Final Reconstruction for {args.data_split} data"))
    
    # Execute pipeline
    failed_steps = []
    completed_steps = []
    
    for cmd, description in pipeline_commands:
        success = run_command(cmd, description)
        if success:
            completed_steps.append(description)
        else:
            failed_steps.append(description)
            print(f"\n❌ Pipeline failed at step: {description}")
            print("You can retry from this step or check the error messages above.")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed steps: {len(completed_steps)}")
    for step in completed_steps:
        print(f"   - {step}")
    
    if failed_steps:
        print(f"\n❌ Failed steps: {len(failed_steps)}")
        for step in failed_steps:
            print(f"   - {step}")
        sys.exit(1)
    else:
        print(f"\n🎉 ERA5-Primary Pipeline completed successfully!")
        
        # Show output information
        output_base = f"output/{args.roi_name}"
        reconstruction_dir = f"{output_base}/reconstructed_lst{'_'+args.data_split if args.data_split != 'test' else ''}"
        uncertainty_dir = f"{output_base}/uncertainty_maps{'_'+args.data_split if args.data_split != 'test' else ''}"
        
        print(f"""
📁 Output Files:
   - Reconstructed LST: {reconstruction_dir}/
   - Uncertainty Maps: {uncertainty_dir}/
   - Data Split Info: {output_base}/data_split_info.json
   
🗓️  Timeline Information:
   The reconstruction images now cover ALL ERA5 dates for your ROI.
   This includes both actual LST observation dates and synthetic LST dates
   created via temporal interpolation.
   
🔍 Next Steps:
   1. Check the data_split_info.json for timeline details
   2. Examine the reconstruction images - you should see images for all ERA5 dates
   3. Compare with the original LST-primary approach to see the difference in coverage
""")

if __name__ == "__main__":
    main() 