# Comprehensive Google Earth Engine Data Downloader

This script provides a comprehensive solution for downloading satellite data from multiple sources via Google Earth Engine (GEE) and organizing them into structured ZIP files with task ID mapping. It supports both grid-based (recommended) and region-based (legacy) approaches for defining areas of interest.

## Features

- **Multiple Dataset Support**: ERA5, ASTER, Sentinel-2, Landsat 8/9 (L1 and L2)
- **Grid-Based ROI Support**: Use GeoJSON grid files with PhienHieu identifiers
- **Region-Based ROI Support**: Legacy bounding box approach
- **Native Resolution**: Each dataset downloads at its native resolution
- **Cloud Filtering**: Automatic cloud masking for optical data (Sentinel-2)
- **Organized Output**: Individual ZIP files with task IDs as filenames
- **JSON Mapping**: Complete task mapping for tracking downloads
- **Date Range Processing**: Download data for any date range
- **Proper Band Selection**: Specific bands for each dataset type
- **Error Handling**: Robust error handling with retry mechanisms

## Supported Datasets

### ERA5 Land Reanalysis
- **Source**: `ECMWF/ERA5_LAND/HOURLY`
- **Bands**: `skin_temperature`, `temperature_2m`
- **Output**: `{task_id}.zip`
- **Files**: 
  - `skin_temperature_era5_{YYYYMMDD_HHMMSS}Z.tif`
  - `temperature_2m_era5_{YYYYMMDD_HHMMSS}Z.tif`

### ASTER Global Emissivity Dataset
- **Source**: `NASA/ASTER_GED/AG100_003`
- **Bands**: `emissivity_band10`, `emissivity_band11`, `emissivity_band12`, `emissivity_band13`, `emissivity_band14`, `ndvi`
- **Output**: `{task_id}.zip`
- **Files**:
  - `ASTER_emissivity_band10.tif` (8.125-8.475 μm)
  - `ASTER_emissivity_band11.tif` (8.475-8.825 μm)
  - `ASTER_emissivity_band12.tif` (8.925-9.275 μm)
  - `ASTER_emissivity_band13.tif` (10.25-10.95 μm)
  - `ASTER_emissivity_band14.tif` (10.95-11.65 μm)
  - `ASTER_ndvi.tif`

### Sentinel-2 Surface Reflectance
- **Source**: `COPERNICUS/S2_SR_HARMONIZED`
- **Bands**: `B4`, `B3`, `B2`, `B8` (Red, Green, Blue, NIR)
- **Output**: `sentinel2_{task_id}.zip`
- **Files**:
  - `S2_SR_B4_{YYYYMMDD}_00.tif` (Red band - 665nm)
  - `S2_SR_B3_{YYYYMMDD}_00.tif` (Green band - 560nm)
  - `S2_SR_B2_{YYYYMMDD}_00.tif` (Blue band - 490nm)
  - `S2_SR_B8_{YYYYMMDD}_00.tif` (NIR band - 842nm)
- **Processing**: Multi-band download attempt with fallback to individual band downloads if size limits are hit. Filenames include the MGRS tile ID to prevent duplicates.

### Landsat 8/9 Level 1 (TOA)
- **Sources**: 
  - `LANDSAT/LC08/C02/T1_TOA` (Landsat 8)
  - `LANDSAT/LC09/C02/T1_TOA` (Landsat 9)
- **Bands**: `B10`, `B11` (Thermal Infrared)
- **Output**: `landsat8_l1_{task_id}.zip` / `landsat9_l1_{task_id}.zip`
- **Files**:
  - `L8_L1_B10_{YYYYMMDD}_{scene_id}.tif` (Thermal Infrared Band 10)
  - `L8_L1_B11_{YYYYMMDD}_{scene_id}.tif` (Thermal Infrared Band 11)

### Landsat 8/9 Level 2 (Surface Reflectance)
- **Sources**:
  - `LANDSAT/LC08/C02/T1_L2` (Landsat 8)
  - `LANDSAT/LC09/C02/T1_L2` (Landsat 9)
- **Bands**: `SR_B1`, `SR_B2`, `SR_B3`, `SR_B4`, `SR_B5`, `SR_B6`, `SR_B7`, `QA_PIXEL`
- **Output**: `landsat8_l2_{task_id}.zip` / `landsat9_l2_{task_id}.zip`
- **Files**:
  - `L8_L2_SR_B1_{YYYYMMDD}_{scene_id}.tif` through `L8_L2_SR_B7_{YYYYMMDD}_{scene_id}.tif`
  - `L8_L2_QA_PIXEL_{YYYYMMDD}_{scene_id}.tif`

## Installation

### Prerequisites

```bash
# Install required Python packages
pip install earthengine-api
pip install rasterio
pip install numpy
pip install requests

# Authenticate with Google Earth Engine
earthengine authenticate
```

### Google Earth Engine Setup

1. **Create a GEE Account**: Sign up at [earthengine.google.com](https://earthengine.google.com)
2. **Enable Earth Engine API**: Visit [console.cloud.google.com](https://console.cloud.google.com)
3. **Update Project ID**: Change the project ID in the script:
   ```python
   ee.Initialize(project='your-project-id-here')
   ```

## ROI Definition Approaches

### Grid-Based Approach (Recommended)

The grid-based approach uses GeoJSON files containing predefined grid features with `PhienHieu` (grid identifier) properties. This is the recommended approach for systematic data collection.

**Benefits:**
- Consistent, standardized grid coverage
- Easy integration with existing grid systems
- Automated output organization by grid ID
- Supports complex polygon geometries

### Region-Based Approach (Legacy)

The region-based approach uses simple bounding box coordinates. This is maintained for backward compatibility.

## Usage

### Command Line Interface

#### Grid-Based Usage (Recommended)

```bash
# Basic grid-based download
python comprehensive_gee_downloader.py \
  --start_date 2023-01-01 \
  --end_date 2023-01-07 \
  --grid_file "data/Grid_50K_MatchedDates.geojson" \
  --phien_hieu "D-49-49-A" \
  --output_dir "./downloads"

# Download specific datasets for a grid
python comprehensive_gee_downloader.py \
  --start_date 2023-06-15 \
  --end_date 2023-06-20 \
  --grid_file "data/Grid_50K_MatchedDates.geojson" \
  --phien_hieu "D-49-49-A" \
  --output_dir "./grid_data" \
  --datasets era5 aster sentinel2 landsat8_l2
```

#### Region-Based Usage (Legacy)

```bash
# Basic region-based download
python comprehensive_gee_downloader.py \
  --start_date 2023-01-01 \
  --end_date 2023-01-07 \
  --region "-122.5,37.5,-122.0,38.0" \
  --output_dir "./downloads"

# Download specific datasets for a region
python comprehensive_gee_downloader.py \
  --start_date 2023-06-15 \
  --end_date 2023-06-20 \
  --region "105.0,10.0,106.0,11.0" \
  --output_dir "./vietnam_data" \
  --datasets era5 aster sentinel2 landsat8_l2
```

### Python API

#### Grid-Based API Usage

```python
import ee
from datetime import datetime
from comprehensive_gee_downloader import (
    GEEDataDownloader, 
    find_grid_feature, 
    get_roi_geometry_from_geojson
)

# Initialize Earth Engine
ee.Initialize(project='your-project-id')

# Find grid feature
grid_file = "data/Grid_50K_MatchedDates.geojson"
phien_hieu = "D-49-49-A"
feature = find_grid_feature(phien_hieu, grid_file)
roi_geometry = get_roi_geometry_from_geojson(feature['geometry'])

# Initialize downloader
downloader = GEEDataDownloader(f"./output/{phien_hieu}", roi_geometry)

# Download data for date range
start_date = datetime(2023, 6, 15)
end_date = datetime(2023, 6, 20)

task_mapping = downloader.download_all_for_date_range(
    start_date, 
    end_date,
    datasets=['era5', 'aster', 'sentinel2', 'landsat8_l2']
)
```

#### Region-Based API Usage (Legacy)

```python
import ee
from datetime import datetime
from comprehensive_gee_downloader import GEEDataDownloader, get_roi_geometry

# Initialize Earth Engine
ee.Initialize(project='your-project-id')

# Define region of interest (bounding box as polygon)
region = [
    [105.0, 10.0],   # Southwest corner (lon, lat)
    [106.0, 10.0],   # Southeast corner
    [106.0, 11.0],   # Northeast corner
    [105.0, 11.0],   # Northwest corner
    [105.0, 10.0]    # Close polygon
]

roi_geometry = get_roi_geometry(region)

# Initialize downloader
downloader = GEEDataDownloader("./output", roi_geometry)

# Download data for date range
start_date = datetime(2023, 6, 15)
end_date = datetime(2023, 6, 20)

task_mapping = downloader.download_all_for_date_range(
    start_date, 
    end_date,
    datasets=['era5', 'aster', 'sentinel2', 'landsat8_l2']
)
```

### Command Line Arguments

- `--start_date`: Start date in YYYY-MM-DD format
- `--end_date`: End date in YYYY-MM-DD format  
- `--region`: Region as "minLon,minLat,maxLon,maxLat"
- `--output_dir`: Output directory (default: ./gee_downloads)
- `--datasets`: Space-separated list of datasets to download

**Available datasets**: `era5`, `aster`, `sentinel2`, `landsat8_l1`, `landsat8_l2`, `landsat9_l1`, `landsat9_l2`

## Output Structure

```
output_directory/
├── downloaded_zips/
│   ├── {task_id}.zip (ERA5 data)
│   ├── {task_id}.zip (ASTER data)
│   ├── {task_id}.zip (Sentinel-2 data)
│   ├── {task_id}.zip (Landsat 8 L1 data)
│   ├── {task_id}.zip (Landsat 8 L2 data)
│   ├── {task_id}.zip (Landsat 9 L1 data)
│   └── {task_id}.zip (Landsat 9 L2 data)
└── task_mapping.json
```

### Task Mapping JSON Structure

```json
{
  "summary": {
    "total_tasks": 12,
    "datasets": {
      "ERA5": 2,
      "ASTER": 2,
      "Landsat8_L1": 2,
      "Landsat8_L2": 2,
      "Landsat9_L1": 2,
      "Landsat9_L2": 2
    },
    "date_range": {
      "start": "2023-06-15",
      "end": "2023-06-20"
    },
    "generated_at": "2024-01-01T12:00:00"
  },
  "task_mapping": {
    "uuid-1234-5678": {
      "dataset_type": "ERA5",
      "date": "2023-06-15",
      "bands": ["skin_temperature", "temperature_2m"],
      "metadata": {
        "source": "ECMWF/ERA5_LAND/DAILY",
        "zip_path": "/path/to/downloaded_era5_uuid-1234-5678.zip"
      }
    }
  }
}
```

## Advanced Configuration

### Custom Region Definition

```python
# Define custom region as polygon coordinates
custom_region = [
    [lon1, lat1],  # Point 1
    [lon2, lat2],  # Point 2
    [lon3, lat3],  # Point 3
    [lon4, lat4],  # Point 4
    [lon1, lat1]   # Close polygon (same as first point)
]
```

### Processing Settings

The script uses the following default settings:
- **Target CRS**: EPSG:4326 (WGS84)
- **Export Scales**: Native resolutions per dataset
  - ERA5: 11km (~11,000m) - preserves native 0.1° grid
  - ASTER GED: 100m - native resolution
  - Sentinel-2: 10m - native resolution for optical bands
  - Landsat L1: 100m - native thermal band resolution
  - Landsat L2: 30m - native surface reflectance resolution
- **Cloud Cover Threshold**: 80% (for Landsat)
- **Download Timeout**: 600 seconds

## Error Handling

The script includes comprehensive error handling:
- **Network timeouts**: Automatic retry with exponential backoff
- **GEE quota limits**: Graceful handling of API limits
- **Missing data**: Skip dates/regions with no available data
- **File I/O errors**: Robust file handling with cleanup

## Limitations

1. **GEE Quotas**: Subject to Google Earth Engine usage quotas
2. **File Size**: Large regions may hit download size limits
3. **Date Availability**: Not all datasets available for all dates
4. **Processing Time**: Downloads can be slow for large regions/date ranges

## Examples

See `example_usage.py` for complete working examples:
- Basic usage with specific datasets
- Full dataset download for multiple dates
- Custom region definitions

## Troubleshooting

### Common Issues

1. **Authentication Error**:
   ```bash
   earthengine authenticate
   ```

2. **Project ID Error**:
   Update the project ID in the script to match your GEE project

3. **Quota Exceeded**:
   Reduce the date range or region size, or wait for quota reset

4. **No Data Found**:
   Check if the date range and region have available data for the selected datasets

### Debug Mode

Add verbose logging by modifying the script:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This script is provided as-is for research and educational purposes. Please respect Google Earth Engine terms of service and data licensing requirements.
