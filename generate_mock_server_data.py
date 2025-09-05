#!/usr/bin/env python3
"""
Mock Server Data Generator

This script uses the existing GEE-based data_retrival_module to download
all base datasets (ASTER GED, Landsat 8/9 L1/L2, Sentinel-2, ERA5) and
stores them in folders with unique IDs to simulate API server responses.

Usage:
    python generate_mock_server_data.py --roi_name "49-49-A-c-2-1" \
                                       --start_date "2023-01-01" \
                                       --end_date "2023-01-31" \
                                       --output_dir "mock_server_data"
"""

import argparse
import json
import logging
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import requests
import time

import ee
import rasterio
import numpy as np

# Import the existing GEE-based retrieval modules
try:
    import sys
    sys.path.append('data_retrival_module')
    
    from data_retrival_module.lst_retrieval import lst_retrive
    from data_retrival_module.s2_retrieval import main_s2_retrieval
    from data_retrival_module.era5_retriever import main as era5_main_retrieval
    from data_retrival_module.lst_module.Landsat_LST import COLLECTION
except ImportError as e:
    print(f"Error importing retrieval modules: {e}")
    print("Make sure you're running this from the project root directory")
    print("And that the virtual environment is activated")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mock_data_generation.log')
    ]
)

class MockDataGenerator:
    """Generates mock server data by downloading from GEE and organizing into UUID folders."""
    
    def __init__(self, output_dir: str, roi_name: str, start_date: str, end_date: str, grid_file: str):
        self.output_dir = Path(output_dir)
        self.roi_name = roi_name
        self.start_date = start_date
        self.end_date = end_date
        self.grid_file = grid_file
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Temporary directory for GEE downloads
        self.temp_dir = self.output_dir / "temp_gee_downloads"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Earth Engine
        self._init_earth_engine()
        
        # Get ROI geometry
        self.roi_geometry = self._get_roi_geometry()
        
    def _init_earth_engine(self):
        """Initialize Earth Engine with authentication."""
        try:
            ee.Initialize(project='ee-hadat-461702-p4')
            logging.info("Earth Engine initialized successfully")
        except Exception:
            logging.info("Authenticating to Earth Engine...")
            ee.Authenticate()
            ee.Initialize(project='ee-hadat-461702-p4')
            
    def _get_roi_geometry(self):
        """Get ROI geometry from grid file."""
        try:
            with open(self.grid_file, 'r') as f:
                grid_data = json.load(f)
            
            for feature in grid_data.get('features', []):
                if feature.get('properties', {}).get('PhienHieu') == self.roi_name:
                    roi_geometry_data = feature['geometry']
                    return ee.Geometry.Polygon(roi_geometry_data['coordinates'])
            
            raise ValueError(f"ROI '{self.roi_name}' not found in grid file")
        except Exception as e:
            logging.error(f"Failed to get ROI geometry: {e}")
            raise
            
    def generate_uuid_folder(self, dataset_type: str) -> str:
        """Generate a UUID folder for a dataset type."""
        folder_id = str(uuid.uuid4())
        folder_path = self.output_dir / dataset_type / folder_id
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
        
    def check_existing_data(self, dataset_type: str) -> Path:
        """Check if data already exists for a dataset type and return the folder path."""
        dataset_dir = self.output_dir / dataset_type
        if dataset_dir.exists():
            # Find existing folders with data
            for folder in dataset_dir.iterdir():
                if folder.is_dir():
                    # Check if folder has any .tif files
                    tif_files = list(folder.glob("*.tif"))
                    if tif_files:
                        logging.info(f"📁 Found existing {dataset_type} data with {len(tif_files)} files in {folder.name}")
                        return folder
        return None
        
    def create_manifest(self, folder_path: Path, dataset_type: str, metadata: dict):
        """Create a manifest file for the dataset folder."""
        manifest = {
            "dataset_type": dataset_type,
            "roi_name": self.roi_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "generated_at": datetime.now().isoformat(),
            "files": [],
            **metadata
        }
        
        # List all files in the folder
        for file_path in folder_path.rglob("*.tif"):
            relative_path = file_path.relative_to(folder_path)
            file_info = {
                "filename": str(relative_path),
                "size_bytes": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            manifest["files"].append(file_info)
        
        # Write manifest
        manifest_path = folder_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logging.info(f"Created manifest for {dataset_type}: {len(manifest['files'])} files")
        
    def download_era5_data(self) -> Path:
        """Download ERA5 data using existing retrieval module."""
        logging.info("🌍 Downloading ERA5 data from GEE...")
        
        # Check if data already exists
        existing_folder = self.check_existing_data("era5")
        if existing_folder:
            logging.info("✅ Skipping ERA5 download - data already exists")
            return existing_folder
        
        uuid_folder = self.generate_uuid_folder("era5")
        temp_lst_folder = self.temp_dir / "lst_for_era5"
        temp_era5_folder = self.temp_dir / "era5"
        
        try:
            temp_lst_folder.mkdir(parents=True, exist_ok=True)
            temp_era5_folder.mkdir(parents=True, exist_ok=True)
            
            # Check if LST data was already downloaded and copy it as reference
            lst_folders = list(self.output_dir.glob("lst/*/"))
            if lst_folders:
                # Copy LST files to temp folder for ERA5 to use as reference
                lst_source = lst_folders[0]  # Use the first LST folder
                for lst_file in lst_source.glob("*.tif"):
                    shutil.copy(str(lst_file), str(temp_lst_folder / lst_file.name))
                logging.info(f"Using {len(list(lst_source.glob('*.tif')))} LST files as reference for ERA5")
            else:
                # Fallback: create dummy reference if no LST files available
                logging.warning("No LST files found, creating dummy reference for ERA5")
                dummy_tif_path = temp_lst_folder / "dummy_reference.tif"
                self._create_dummy_reference_tif(dummy_tif_path)
            
            # Generate target dates (every 16 days in the date range)
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
            
            target_dates = []
            current_dt = start_dt
            while current_dt <= end_dt:
                target_dates.append(current_dt.strftime('%Y-%m-%d'))
                current_dt += timedelta(days=16)
            
            # Call the existing ERA5 retrieval function
            era5_main_retrieval(
                input_folder=str(temp_lst_folder),
                output_folder=str(temp_era5_folder),
                specific_dates=target_dates
            )
            
            # Move downloaded files to UUID folder
            if temp_era5_folder.exists():
                for file_path in temp_era5_folder.glob("*.tif"):
                    shutil.move(str(file_path), str(uuid_folder / file_path.name))
            
            # Create manifest
            self.create_manifest(uuid_folder, "era5", {
                "variables": ["skin_temperature"],
                "resolution": "30m",
                "target_dates": target_dates
            })
            
            return uuid_folder
            
        except Exception as e:
            logging.error(f"Failed to download ERA5 data: {e}")
            raise
            
    def download_s2_data(self) -> Path:
        """Download Sentinel-2 data using existing retrieval module."""
        logging.info("🛰️ Downloading Sentinel-2 data from GEE...")
        
        # Check if data already exists
        existing_folder = self.check_existing_data("s2")
        if existing_folder:
            logging.info("✅ Skipping S2 download - data already exists")
            return existing_folder
        
        uuid_folder = self.generate_uuid_folder("s2")
        temp_s2_folder = self.temp_dir / "s2_temp"
        
        try:
            temp_s2_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate target dates (every 8 days)
            start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
            
            target_dates = []
            current_dt = start_dt
            while current_dt <= end_dt:
                target_dates.append(ee.Date(current_dt.strftime('%Y-%m-%d')))
                current_dt += timedelta(days=8)
            
            # Call the existing S2 retrieval function
            main_s2_retrieval(
                target_composite_dates=target_dates,
                roi=self.roi_geometry,
                roi_name=self.roi_name,
                big_folder=str(temp_s2_folder)
            )
            
            # Move downloaded files to UUID folder
            s2_source_folder = temp_s2_folder / self.roi_name / "s2_images"
            if s2_source_folder.exists():
                for file_path in s2_source_folder.glob("*.tif"):
                    shutil.move(str(file_path), str(uuid_folder / file_path.name))
            
            # Create manifest
            self.create_manifest(uuid_folder, "s2", {
                "bands": ["B2", "B3", "B4", "B8"],
                "resolution": "30m",
                "composite_type": "8-day_median"
            })
            
            return uuid_folder
            
        except Exception as e:
            logging.error(f"Failed to download S2 data: {e}")
            raise
            
    def download_landsat_data(self, satellite: str, level: str) -> Path:
        """Download Landsat data (L8/L9, L1/L2) using Earth Engine directly."""
        dataset_type = f"landsat{satellite.lower()}_{level.lower()}"
        logging.info(f"🛰️ Downloading Landsat {satellite} {level.upper()} data from GEE...")
        
        # Check if data already exists
        existing_folder = self.check_existing_data(dataset_type)
        if existing_folder:
            logging.info(f"✅ Skipping Landsat {satellite} {level.upper()} download - data already exists")
            return existing_folder
        
        uuid_folder = self.generate_uuid_folder(dataset_type)
        
        try:
            # Get collection info from Landsat_LST module - use direct string IDs
            if satellite == "L8":
                if level.lower() == "l1":
                    collection_name = 'LANDSAT/LC08/C02/T1_TOA'
                    bands = ['B10', 'B11']  # TIR bands for L1
                else:  # L2
                    collection_name = 'LANDSAT/LC08/C02/T1_L2'
                    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']  # Visible/SWIR bands for L2
            elif satellite == "L9":
                if level.lower() == "l1":
                    collection_name = 'LANDSAT/LC09/C02/T1_TOA'
                    bands = ['B10', 'B11']  # TIR bands for L1
                else:  # L2
                    collection_name = 'LANDSAT/LC09/C02/T1_L2'
                    bands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']  # Visible/SWIR bands for L2
            else:
                raise ValueError(f"Unsupported satellite: {satellite}")
            
            # Filter the collection
            collection = ee.ImageCollection(collection_name) \
                .filterDate(self.start_date, self.end_date) \
                .filterBounds(self.roi_geometry) \
                .filter(ee.Filter.lt('CLOUD_COVER', 70))
            
            # Download images
            image_list = collection.toList(collection.size())
            image_count = collection.size().getInfo()
            
            logging.info(f"Found {image_count} {satellite} {level.upper()} images")
            
            downloaded_files = []
            for i in range(min(image_count, 5)):  # Limit to 5 images to avoid timeouts
                image = ee.Image(image_list.get(i))
                image_date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
                time_start_ms = image.get('system:time_start').getInfo()
                
                # Download image
                filename = f"{satellite}_{level.upper()}_{image_date}.tif"
                output_path = uuid_folder / filename
                
                if self._download_ee_image(image, bands, output_path, time_start_ms, f"{satellite}_{level.upper()}"):
                    downloaded_files.append(filename)
            
            # Create manifest
            self.create_manifest(uuid_folder, f"landsat{satellite.lower()}_{level.lower()}", {
                "satellite": satellite,
                "level": level.upper(),
                "bands": bands,
                "resolution": "30m",
                "downloaded_files": downloaded_files
            })
            
            return uuid_folder
            
        except Exception as e:
            logging.error(f"Failed to download Landsat {satellite} {level} data: {e}")
            raise
            
    def download_aster_data(self) -> Path:
        """Download ASTER GED data using Earth Engine directly."""
        logging.info("🏔️ Downloading ASTER GED data from GEE...")
        
        # Check if data already exists
        existing_folder = self.check_existing_data("aster")
        if existing_folder:
            logging.info("✅ Skipping ASTER download - data already exists")
            return existing_folder
        
        uuid_folder = self.generate_uuid_folder("aster")
        
        try:
            # ASTER GED is a single image, not a collection
            image = ee.Image("NASA/ASTER_GED/AG100_003")
            
            # ASTER bands for emissivity and NDVI
            bands = ['emissivity_band10', 'emissivity_band11', 'emissivity_band12', 
                    'emissivity_band13', 'emissivity_band14', 'ndvi']
            
            downloaded_files = []
            for band in bands:
                filename = f"ASTER_{band}.tif"
                output_path = uuid_folder / filename
                
                if self._download_ee_image(image.select(band), [band], output_path, None, "ASTER_GED"):
                    downloaded_files.append(filename)
            
            # Create manifest
            self.create_manifest(uuid_folder, "aster", {
                "bands": bands,
                "resolution": "100m",
                "dataset": "ASTER_GED_v003",
                "downloaded_files": downloaded_files
            })
            
            return uuid_folder
            
        except Exception as e:
            logging.error(f"Failed to download ASTER data: {e}")
            raise
            
    def _download_ee_image(self, image, bands, output_path, timestamp_ms=None, acquisition_type=None):
        """Download an Earth Engine image to local path."""
        try:
            # Get download URL
            download_params = {
                'scale': 30,
                'region': self.roi_geometry,
                'fileFormat': 'GeoTIFF',
                'crs': 'EPSG:4326'
            }
            
            if len(bands) == 1:
                # Single band download
                download_url = image.select(bands[0]).getDownloadURL(download_params)
            else:
                # Multi-band download
                download_url = image.select(bands).getDownloadURL(download_params)
            
            # Download
            response = requests.get(download_url, timeout=300)
            response.raise_for_status()
            
            # Handle ZIP response
            if response.headers.get('Content-Type', '').find('zip') != -1:
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                    temp_zip.write(response.content)
                    temp_zip_path = temp_zip.name
                
                # Extract ZIP
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    tif_files = [f for f in zip_ref.namelist() if f.endswith('.tif')]
                    if tif_files:
                        # Extract the first TIF file
                        with tempfile.TemporaryDirectory() as temp_dir:
                            zip_ref.extract(tif_files[0], temp_dir)
                            extracted_path = os.path.join(temp_dir, tif_files[0])
                            shutil.move(extracted_path, output_path)
                
                os.unlink(temp_zip_path)
            else:
                # Direct TIFF response
                with open(output_path, 'wb') as f:
                    f.write(response.content)
            
            # Write metadata
            if timestamp_ms or acquisition_type:
                self._write_metadata_to_tiff(output_path, timestamp_ms, acquisition_type)
            
            logging.info(f"Successfully downloaded: {output_path.name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to download {output_path.name}: {e}")
            return False
            
    def _write_metadata_to_tiff(self, tif_path, timestamp_ms=None, acquisition_type=None):
        """Write metadata to a GeoTIFF file."""
        try:
            with rasterio.open(tif_path, 'r+') as dst:
                tags = {}
                if timestamp_ms:
                    dt_object = datetime.fromtimestamp(timestamp_ms / 1000)
                    datetime_str = dt_object.strftime('%Y:%m:%d %H:%M:%S')
                    tags['DATETIME'] = datetime_str
                
                if acquisition_type:
                    tags['ACQUISITION_TYPE'] = acquisition_type
                
                if tags:
                    dst.update_tags(**tags)
        except Exception as e:
            logging.warning(f"Failed to write metadata to {tif_path}: {e}")
            
    def _create_dummy_reference_tif(self, tif_path):
        """Create a dummy reference TIF file for ERA5 alignment."""
        try:
            # Get ROI bounds
            roi_coords = self.roi_geometry.getInfo()['coordinates'][0]
            
            # Calculate bounds
            lons = [coord[0] for coord in roi_coords]
            lats = [coord[1] for coord in roi_coords]
            west, east = min(lons), max(lons)
            south, north = min(lats), max(lats)
            
            # Create a simple raster
            width, height = 100, 100
            transform = rasterio.transform.from_bounds(west, south, east, north, width, height)
            
            # Create dummy data
            data = np.ones((height, width), dtype=np.float32) * 300.0  # 300K temperature
            
            # Write the file
            with rasterio.open(
                tif_path, 'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=data.dtype,
                crs='EPSG:4326',
                transform=transform,
                nodata=np.nan
            ) as dst:
                dst.write(data, 1)
                
        except Exception as e:
            logging.error(f"Failed to create dummy reference TIF: {e}")
            raise
            
    def download_lst_data(self) -> Path:
        """Download LST data using existing retrieval module."""
        logging.info("🌡️ Downloading LST data from GEE...")
        
        # Check if data already exists
        existing_folder = self.check_existing_data("lst")
        if existing_folder:
            logging.info("✅ Skipping LST download - data already exists")
            return existing_folder
        
        uuid_folder = self.generate_uuid_folder("lst")
        temp_lst_folder = self.temp_dir / "lst_download"
        
        try:
            temp_lst_folder.mkdir(parents=True, exist_ok=True)
            
            # Call the existing LST retrieval function
            lst_retrive(
                date_start=self.start_date,
                date_end=self.end_date,
                geometry=self.roi_geometry,
                ROI=self.roi_name,
                main_folder=str(temp_lst_folder)
            )
            
            # Move downloaded files to UUID folder
            lst_source_folder = temp_lst_folder / self.roi_name / "lst"
            if lst_source_folder.exists():
                for file_path in lst_source_folder.glob("*.tif"):
                    shutil.move(str(file_path), str(uuid_folder / file_path.name))
            
            # Create manifest
            self.create_manifest(uuid_folder, "lst", {
                "satellites": ["L8", "L9"],
                "algorithm": "Single_Mono_Window",
                "resolution": "30m"
            })
            
            return uuid_folder
            
        except Exception as e:
            logging.error(f"Failed to download LST data: {e}")
            raise
            
    def generate_all_datasets(self):
        """Generate all mock datasets."""
        logging.info(f"🚀 Starting mock data generation for ROI: {self.roi_name}")
        logging.info(f"📅 Date range: {self.start_date} to {self.end_date}")
        
        generated_folders = {}
        
        try:
            # Download datasets in dependency order
            
            # 1. Sentinel-2 data (independent)
            generated_folders['s2'] = self.download_s2_data()
            
            # 2. ASTER GED data (static global dataset, independent)  
            generated_folders['aster'] = self.download_aster_data()
            
            # 3. Landsat data (needed for LST calculation)
            for satellite in ['L8', 'L9']:
                for level in ['l1', 'l2']:
                    key = f'landsat{satellite.lower()}_{level}'
                    generated_folders[key] = self.download_landsat_data(satellite, level)
            
            # 4. LST data (uses Landsat data, needed as reference for ERA5)
            generated_folders['lst'] = self.download_lst_data()
            
            # 5. ERA5 data (uses LST files as reference for alignment)
            generated_folders['era5'] = self.download_era5_data()
            
        except Exception as e:
            logging.error(f"Error during data generation: {e}")
            raise
        finally:
            # Cleanup temporary directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        
        # Create summary manifest
        self._create_summary_manifest(generated_folders)
        
        logging.info("✅ Mock data generation completed successfully!")
        return generated_folders
        
    def _create_summary_manifest(self, generated_folders):
        """Create a summary manifest of all generated datasets."""
        summary = {
            "roi_name": self.roi_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "generated_at": datetime.now().isoformat(),
            "datasets": {}
        }
        
        for dataset_type, folder_path in generated_folders.items():
            folder_id = folder_path.name
            summary["datasets"][dataset_type] = {
                "folder_id": folder_id,
                "folder_path": str(folder_path.relative_to(self.output_dir)),
                "manifest_path": str((folder_path / "manifest.json").relative_to(self.output_dir))
            }
        
        summary_path = self.output_dir / "summary_manifest.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Created summary manifest: {summary_path}")
        
        # Print results
        print("\n" + "="*60)
        print("🎉 MOCK DATA GENERATION SUMMARY")
        print("="*60)
        print(f"ROI: {self.roi_name}")
        print(f"Date Range: {self.start_date} to {self.end_date}")
        print(f"Output Directory: {self.output_dir}")
        print("\nGenerated Datasets:")
        for dataset_type, info in summary["datasets"].items():
            print(f"  📁 {dataset_type.upper()}: {info['folder_id']}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Generate mock server data using GEE")
    parser.add_argument("--roi_name", required=True, help="ROI PhienHieu identifier")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output_dir", default="mock_server_data", help="Output directory")
    parser.add_argument("--grid_file", default="data/Grid_50K_MatchedDates.geojson", help="Grid file path")
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = MockDataGenerator(
        output_dir=args.output_dir,
        roi_name=args.roi_name,
        start_date=args.start_date,
        end_date=args.end_date,
        grid_file=args.grid_file
    )
    
    generated_folders = generator.generate_all_datasets()
    
    print(f"\n✅ Mock data generation completed successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📄 Summary manifest: {args.output_dir}/summary_manifest.json")


if __name__ == "__main__":
    main()
