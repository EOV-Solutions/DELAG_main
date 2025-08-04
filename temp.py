import rasterio
from osgeo import gdal
import os

# Path to your downloaded TIFF file
tiff_path = 'data/retrieved_data/F-48-93-B/lst/L8_lst16days_2015-01-15.tif'

def read_timestamp_metadata(tiff_path):
    """
    Reads the timestamp metadata from images downloaded by the updated crawling code.
    The metadata includes DATETIME and ACQUISITION_TYPE tags.
    """
    print(f"Reading metadata from: {tiff_path}")
    
    if not os.path.exists(tiff_path):
        print(f"Error: File does not exist at {tiff_path}")
        return
    
    try:
        with rasterio.open(tiff_path) as src:
            tags = src.tags()
            
            print("\n--- Image Metadata ---")
            print(f"File: {os.path.basename(tiff_path)}")
            
            # Read the new timestamp metadata
            if 'DATETIME' in tags:
                datetime_str = tags['DATETIME']
                print(f"Acquisition Datetime: {datetime_str}")
            else:
                print("DATETIME metadata not found")
            
            if 'ACQUISITION_TYPE' in tags:
                acquisition_type = tags['ACQUISITION_TYPE']
                print(f"Acquisition Type: {acquisition_type}")
            else:
                print("ACQUISITION_TYPE metadata not found")
            
            # Display all available tags for debugging
            print(f"\nAll available tags ({len(tags)} total):")
            for key, value in tags.items():
                print(f"  {key}: {value}")
                
            # Basic image information
            print(f"\nImage Information:")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Bands: {src.count}")
            print(f"  CRS: {src.crs}")
            print(f"  Data type: {src.dtypes[0]}")
            
    except Exception as e:
        print(f"Error reading metadata: {e}")

def test_multiple_files(base_folder):
    """
    Test reading metadata from multiple downloaded files to verify consistency.
    """
    print("="*60)
    print("TESTING MULTIPLE FILES")
    print("="*60)
    
    # Test different types of downloaded files
    test_paths = [
        os.path.join(base_folder, "lst", "L8_lst16days_2015-01-15.tif"),
        os.path.join(base_folder, "lst", "L9_lst16days_2015-01-15.tif"),
        os.path.join(base_folder, "era5", "ERA5_data_2015-01-15.tif"),
        os.path.join(base_folder, "modis", "MODIS_LST_2015-01-15.tif"),
        os.path.join(base_folder, "s2_images", "S2_8days_2015-01-15.tif")
    ]
    
    for test_path in test_paths:
        if os.path.exists(test_path):
            read_timestamp_metadata(test_path)
        else:
            print(f"File not found (skipping): {test_path}")
        print("-" * 40)

# Read metadata from the specified file
read_timestamp_metadata(tiff_path)

# Test multiple file types if the base folder exists
base_folder = 'data/retrieved_data/F-48-93-B'
if os.path.exists(base_folder):
    test_multiple_files(base_folder)