# ETL Module Configuration Guide

The ETL Data Retrieval Module now uses a centralized configuration system that manages all API endpoints, dataset parameters, and processing settings.

## Configuration File Structure

All configuration is centralized in `config.py` which provides:
- **API Endpoints**: Separate endpoints for each satellite and processing level
- **Dataset Settings**: Default parameters for each data type
- **Output Configuration**: File naming patterns and directory structure
- **Processing Parameters**: LST calculation coefficients, nodata values, etc.

## API Endpoints Configuration

### Search Endpoints
```python
SEARCH_ENDPOINTS = {
    "era5": "/v1/era5_search",
    "s2": "/v1/s2_search", 
    "landsat8_l1": "/v1/landsat8_l1_search",
    "landsat8_l2": "/v1/landsat8_l2_search", 
    "landsat9_l1": "/v1/landsat9_l1_search",
    "landsat9_l2": "/v1/landsat9_l2_search",
    "aster": "/v1/aster_search"
}
```

### Download Endpoints
```python
DOWNLOAD_ENDPOINTS = {
    "era5": "/v1/era5_download",
    "s2": "/v1/s2_download",
    "landsat8_l1": "/v1/landsat8_l1_download",
    "landsat8_l2": "/v1/landsat8_l2_download",
    "landsat9_l1": "/v1/landsat9_l1_download", 
    "landsat9_l2": "/v1/landsat9_l2_download",
    "aster": "/v1/aster_download"
}
```

## Dataset Configuration

### ERA5 Settings
```python
ERA5_CONFIG = {
    "default_variables": ["skin_temperature"],
    "default_utc_hours": [10],  # Typical LST acquisition time
    "default_limit": 50
}
```

### Sentinel-2 Settings
```python
S2_CONFIG = {
    "default_bands": ["B2", "B3", "B4", "B8"],  # Blue, Green, Red, NIR
    "default_cloud_cover": 85.0,
    "composite_window_days": 4,  # ±4 days for 8-day composites
    "default_limit": 50
}
```

### Landsat Settings
```python
LANDSAT_CONFIG = {
    "L8": {
        "l1_endpoint_key": "landsat8_l1",
        "l2_endpoint_key": "landsat8_l2", 
        "tir_bands_l1": ["B10", "B11"],
        "tir_bands_l2": ["ST_B10"],
        "optical_bands_l2": ["SR_B4", "SR_B5"],
        # ... more settings
    },
    "L9": {
        # Similar structure for Landsat 9
    }
}
```

### ASTER Settings
```python
ASTER_CONFIG = {
    "default_bands": ["emissivity_band10", "emissivity_band11", "ndvi"],
    "emissivity_scaling": 0.001,
    "ndvi_scaling": 0.01
}
```

## Output Configuration

### File Naming Patterns
```python
OUTPUT_PATTERNS = {
    "era5": "ERA5_data_{date}.tif",
    "s2": "S2_8days_{date}.tif",
    "lst_l8": "L8_lst16days_{date}.tif",
    "lst_l9": "L9_lst16days_{date}.tif"
}
```

### Directory Structure
```python
OUTPUT_DIRS = {
    "era5": "era5",
    "s2": "s2_images", 
    "lst": "lst"
}
```

### Nodata Values
```python
NODATA_VALUES = {
    "era5": "nan",  # Use NaN for ERA5
    "s2": -100,     # Use -100 for S2 to match preprocessing
    "lst": "nan"    # Use NaN for LST
}
```

## Using the Configuration

### Accessing Config Values
```python
from ETL_data_retrieval_module.config import config

# Get endpoints
era5_search_endpoint = config.get_search_endpoint("era5")
landsat8_l1_download = config.get_download_endpoint("landsat8_l1")

# Get dataset settings
era5_variables = config.ERA5_CONFIG["default_variables"]
s2_bands = config.S2_CONFIG["default_bands"]

# Get output settings
era5_output_dir = config.get_output_dir("era5")
s2_nodata = config.get_nodata_value("s2")
```

### ServerClient Integration
The ServerClient now automatically uses the config for endpoint resolution:

```python
from ETL_data_retrieval_module import ServerClient

client = ServerClient()  # Uses config.DEFAULT_API_BASE_URL

# Create tasks using configured endpoints
era5_task = client.create_era5_task(bbox, datetime_range)
landsat_task = client.create_landsat_task("L8", "l1", bbox, datetime_range)
aster_task = client.create_aster_task(bbox, datetime_range)
```

## Customizing Configuration

### Method 1: Modify config.py
Edit the configuration values directly in `config.py` for permanent changes.

### Method 2: Runtime Configuration
```python
from ETL_data_retrieval_module.config import config

# Modify at runtime
config.ERA5_CONFIG["default_variables"] = ["2m_temperature", "skin_temperature"]
config.S2_CONFIG["default_cloud_cover"] = 70.0

# Add custom endpoints
config.SEARCH_ENDPOINTS["custom_satellite"] = "/v1/custom_search"
config.DOWNLOAD_ENDPOINTS["custom_satellite"] = "/v1/custom_download"
```

## Server Implementation Guide

Your server should implement the following endpoints according to the configuration:

### Required Endpoints
1. **ERA5**: `/v1/era5_search`, `/v1/era5_download/{task_id}`
2. **Sentinel-2**: `/v1/s2_search`, `/v1/s2_download/{task_id}`
3. **Landsat 8 L1**: `/v1/landsat8_l1_search`, `/v1/landsat8_l1_download/{task_id}`
4. **Landsat 8 L2**: `/v1/landsat8_l2_search`, `/v1/landsat8_l2_download/{task_id}`
5. **Landsat 9 L1**: `/v1/landsat9_l1_search`, `/v1/landsat9_l1_download/{task_id}`
6. **Landsat 9 L2**: `/v1/landsat9_l2_search`, `/v1/landsat9_l2_download/{task_id}`
7. **ASTER**: `/v1/aster_search`, `/v1/aster_download/{task_id}`

### Search Request Format
All search endpoints should accept:
```json
{
    "bbox": [minx, miny, maxx, maxy],
    "datetime": "YYYY-MM-DDTHH:MM:SSZ/YYYY-MM-DDTHH:MM:SSZ",
    "bands": ["band1", "band2", ...],
    "cloud_cover": 85.0,
    "limit": 50
}
```

### Search Response Format
```json
{
    "task_id": "unique-task-identifier",
    "items_processed": 10,
    "files_created": 25
}
```

### Download Response
- Returns ZIP file containing requested GeoTIFF files
- Files should be organized by date/scene
- File naming should include band identifiers

## Migration from Old System

The new configuration system is backwards compatible. Existing code will continue to work, but you can gradually adopt the new config-based approach:

### Before
```python
client = ServerClient("http://localhost:8000", 120)
endpoint = "/v1/era5_search"
```

### After  
```python
client = ServerClient()  # Uses config defaults
endpoint = config.get_search_endpoint("era5")
```

This centralized configuration makes the ETL module more maintainable and easier to adapt to different server implementations.
