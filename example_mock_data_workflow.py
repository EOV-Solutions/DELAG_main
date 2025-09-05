#!/usr/bin/env python3
"""
Example Mock Data Workflow

This script demonstrates the complete workflow:
1. Generate mock data from GEE
2. Test ETL module with mock data  
3. Clean up

Usage:
    python example_mock_data_workflow.py
"""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(cmd, description):
    """Run a command and handle errors."""
    logging.info(f"🔄 {description}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logging.info(f"✅ {description} - SUCCESS")
        if result.stdout.strip():
            print("Output:", result.stdout.strip())
        return True
    else:
        logging.error(f"❌ {description} - FAILED")
        if result.stderr.strip():
            print("Error:", result.stderr.strip())
        return False

def main():
    # Configuration
    ROI_NAME = "49-49-A-c-2-1"
    START_DATE = "2023-01-01"
    END_DATE = "2023-01-15"  # Shorter range for faster testing
    MOCK_DATA_DIR = "example_mock_data"
    GRID_FILE = "data/Grid_50K_MatchedDates.geojson"
    
    print("🚀 MOCK DATA WORKFLOW EXAMPLE")
    print("=" * 50)
    print(f"ROI: {ROI_NAME}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Mock Data Directory: {MOCK_DATA_DIR}")
    print("=" * 50)
    
    try:
        # Step 1: Check if grid file exists
        if not Path(GRID_FILE).exists():
            logging.error(f"Grid file not found: {GRID_FILE}")
            logging.info("Please ensure the grid file exists before running this example")
            return False
        
        # Step 2: Generate mock data (start with just ERA5 for quick testing)
        logging.info("📥 STEP 1: Generating mock data from GEE...")
        
        mock_cmd = [
            "python", "generate_mock_server_data.py",
            "--roi_name", ROI_NAME,
            "--start_date", START_DATE,
            "--end_date", END_DATE,
            "--output_dir", MOCK_DATA_DIR,
            "--grid_file", GRID_FILE
        ]
        
        if not run_command(mock_cmd, "Generate mock data"):
            return False
        
        # Step 3: Check generated data
        mock_path = Path(MOCK_DATA_DIR)
        summary_manifest = mock_path / "summary_manifest.json"
        
        if summary_manifest.exists():
            logging.info("📋 Generated datasets:")
            import json
            with open(summary_manifest, 'r') as f:
                summary = json.load(f)
            
            for dataset_type, info in summary["datasets"].items():
                folder_path = mock_path / info["folder_path"]
                tif_count = len(list(folder_path.glob("*.tif")))
                logging.info(f"  📁 {dataset_type.upper()}: {tif_count} files in {info['folder_id']}")
        
        # Step 4: Test ETL module with mock data
        logging.info("🧪 STEP 2: Testing ETL module with mock data...")
        
        etl_cmd = [
            "python", "test_mock_data_with_etl.py",
            "--mock_data_dir", MOCK_DATA_DIR,
            "--roi_name", ROI_NAME,
            "--datasets", "era5", "s2",  # Test with a subset for speed
            "--server_port", "8001"  # Use different port to avoid conflicts
        ]
        
        if not run_command(etl_cmd, "Test ETL module"):
            return False
        
        # Step 5: Success message
        print("\n" + "=" * 50)
        print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"✅ Mock data generated in: {MOCK_DATA_DIR}")
        print("✅ ETL module tested successfully")
        print("\nNext steps:")
        print("1. Examine the generated mock data structure")
        print("2. Use this data to develop your API endpoints")
        print("3. Test your ETL module with different datasets")
        print("=" * 50)
        
        return True
        
    except KeyboardInterrupt:
        logging.info("Workflow interrupted by user")
        return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False
    
    finally:
        # Ask user if they want to clean up
        try:
            response = input(f"\nDo you want to remove the mock data directory '{MOCK_DATA_DIR}'? (y/N): ")
            if response.lower().startswith('y'):
                if Path(MOCK_DATA_DIR).exists():
                    shutil.rmtree(MOCK_DATA_DIR)
                    logging.info(f"🗑️  Cleaned up mock data directory: {MOCK_DATA_DIR}")
                else:
                    logging.info("Mock data directory not found (already cleaned up)")
        except (KeyboardInterrupt, EOFError):
            logging.info("\nSkipping cleanup")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
