#!/usr/bin/env python3
"""
ETL Integration Test

Quick integration test for the ETL_data_retrieval_module that can be run from project root.
Tests the download endpoints with mock folder IDs.

Usage from project root:
    python test_etl_integration.py --folder_ids abc123,def456,ghi789,jkl012
"""

import sys
import os
import argparse

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ETL_data_retrieval_module.test_simple_download import main as test_main


def main():
    parser = argparse.ArgumentParser(description="ETL Integration Test")
    parser.add_argument("--folder_ids", required=True,
                       help="Comma-separated folder IDs on your server (e.g., folder1,folder2,folder3,folder4)")
    parser.add_argument("--api_base_url", default="http://localhost:8000",
                       help="Your server API base URL")
    parser.add_argument("--output_folder", default="data/test_retrieved_data",
                       help="Test output folder (relative to project root)")
    
    args = parser.parse_args()
    
    print("🧪 ETL Data Retrieval Module - Integration Test")
    print("=" * 60)
    print(f"📂 Server folder IDs: {args.folder_ids}")
    print(f"🌐 API URL: {args.api_base_url}")
    print(f"📁 Output: {args.output_folder}")
    print("")
    
    # Set up arguments for the test module
    test_args = [
        "--folder_ids", args.folder_ids,
        "--api_base_url", args.api_base_url,
        "--output_folder", args.output_folder,
        "--roi_name", "integration_test_roi"
    ]
    
    # Temporarily modify sys.argv to pass arguments to test
    original_argv = sys.argv
    try:
        sys.argv = ["test_simple_download.py"] + test_args
        result = test_main()
        
        if result == 0:
            print("\n🎉 Integration test PASSED!")
            print(f"✓ ETL module can successfully download and process data")
            print(f"✓ Output files created in expected format")
            print(f"✓ Ready for production use with your server endpoints")
        else:
            print("\n❌ Integration test FAILED!")
            print("Check the error messages above for issues")
            
        return result
        
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
