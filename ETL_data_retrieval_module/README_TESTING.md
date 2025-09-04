# ETL Data Retrieval Module - Testing Guide

This guide shows how to test the ETL workflow using mock data folders on your server before implementing the actual satellite search endpoints.

## Quick Test Setup

### 1. Prepare Test Data on Server

Put 4 folders with unique IDs on your server, each containing sample GeoTIFF files:
- `folder_id_1/` - ERA5-like data (any single-band GeoTIFF)
- `folder_id_2/` - Sentinel-2-like data (4 GeoTIFFs for bands B2,B3,B4,B8)
- `folder_id_3/` - Landsat 8 data (any GeoTIFF files)
- `folder_id_4/` - Landsat 9 data (any GeoTIFF files)

### 2. Ensure Download Endpoints Work

Your server should support these download endpoints:
```
GET /v1/era5_download/{folder_id}      # Returns ZIP with folder_id_1 contents
GET /v1/s2_download/{folder_id}        # Returns ZIP with folder_id_2 contents  
GET /v1/download/{folder_id}           # Generic endpoint for other datasets
```

## Running Tests

### Simple Integration Test (Recommended)

From the project root:
```bash
python test_etl_integration.py --folder_ids folder1,folder2,folder3,folder4 --api_base_url http://localhost:8000
```

### Detailed Download Test

```bash
cd ETL_data_retrieval_module
python test_simple_download.py --folder_ids abc123,def456,ghi789,jkl012 --api_base_url http://your-server:8000
```

### Advanced Workflow Test

```bash
cd ETL_data_retrieval_module  
python test_workflow.py --test_folder_ids folder1,folder2,folder3,folder4 --datasets era5 s2 lst
```

## Expected Output

After successful testing, you should see:
```
data/test_retrieved_data/test_roi/
├── era5/
│   └── ERA5_data_2024-01-15.tif
├── s2_images/
│   └── S2_8days_2024-01-15.tif
└── lst/
    ├── L8_lst16days_2024-01-15.tif
    └── L9_lst16days_2024-01-15.tif
```

## Test Parameters

- `--folder_ids`: Comma-separated list of your server folder IDs
- `--api_base_url`: Your server URL (default: http://localhost:8000)
- `--output_folder`: Where to save test results (default: test_retrieved_data)
- `--roi_name`: Test ROI identifier (default: test_roi)
- `--keep_temp`: Keep downloaded files for inspection

## Troubleshooting

1. **Import Errors**: Make sure you're running from the correct directory
2. **Download Fails**: Check your server is running and folder IDs exist
3. **No TIF Files**: Ensure test folders contain .tif/.TIF files
4. **Permission Errors**: Check write permissions on output folder

## Next Steps

Once testing passes:
1. Your download endpoints are working correctly
2. The ETL module can process your data format
3. Output matches the expected structure for preprocessing
4. Ready to implement actual search endpoints (`/v1/era5_search`, `/v1/s2_search`, etc.)

## Production Usage

After implementing search endpoints, use the full module:
```bash
python -m ETL_data_retrieval_module.main \
  --roi_name D-49-49-A-c-2 \
  --start_date 2020-01-01 \
  --end_date 2020-12-31 \
  --datasets era5 s2 lst \
  --api_base_url http://your-server:8000
```
