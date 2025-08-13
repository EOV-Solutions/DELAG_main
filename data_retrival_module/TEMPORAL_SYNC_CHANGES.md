# Temporal Synchronization Changes

## Problem Statement

The original code had temporal mismatch issues between different data sources:

1. **ERA5 Data**: Used daily aggregated data, which represents average temperatures over 24 hours
2. **LST Data**: Taken at specific times during the day (typically around 10:30 UTC for Landsat)
3. **MODIS Data**: Available as day/night data, but was always using day data regardless of LST acquisition time

This mismatch could lead to inaccurate comparisons and analysis.

## Solution Implemented

### 1. LST Acquisition Time Extraction

Added utility functions to extract the actual acquisition time from LST data:

- `get_lst_acquisition_time()`: Reads DATETIME metadata from LST GeoTIFF files
- `get_lst_file_for_date()`: Finds the corresponding LST file for a given date

### 2. ERA5 Time-Aware Retrieval

**New Function**: `get_era5_for_date_with_time()`

**Changes**:
- Uses ERA5 hourly data instead of daily aggregated data
- Extracts LST acquisition time to determine the target hour
- Finds the closest hourly ERA5 data to the LST acquisition time
- Maintains backward compatibility with the original function

**Key Features**:
- Automatically selects the ERA5 hour closest to the LST acquisition time
- Falls back to 10:30 UTC if no LST metadata is available
- Provides detailed logging of time selection decisions

### 3. MODIS Time-Aware Retrieval

**New Function**: `get_modis_for_date_with_time()`

**Changes**:
- Determines day/night selection based on LST acquisition time
- Uses LST_Day_1km for acquisitions between 6:00-18:00 UTC
- Uses LST_Night_1km for acquisitions between 18:00-6:00 UTC
- Maintains backward compatibility with the original function

**Key Features**:
- Automatically selects appropriate day/night MODIS data
- Falls back to day data if no LST metadata is available
- Provides detailed logging of day/night selection decisions

### 4. Updated Main Functions

Both `era5_retriever.py` and `modis_retriever.py` main functions now use the time-aware retrieval functions by default.

## Files Modified

### 1. `era5_retriever.py`
- Added utility functions for LST time extraction
- Added `get_era5_for_date_with_time()` function
- Updated main function to use time-aware retrieval
- Maintained backward compatibility

### 2. `modis_retriever.py`
- Added utility functions for LST time extraction
- Added `get_modis_for_date_with_time()` function
- Updated main function to use time-aware retrieval
- Maintained backward compatibility

### 3. `main_shapefile.py`
- Already had correct imports for the new functions
- No changes needed - automatically uses time-aware functions

## Usage

The changes are backward compatible. Existing code will continue to work, but new runs will automatically use the time-aware retrieval:

```python
# This will now use time-aware retrieval automatically
era5_main_retrieval(input_folder, output_folder, specific_dates)
modis_main_retrieval(input_folder, output_folder, specific_dates)
```

## Benefits

1. **Temporal Consistency**: ERA5 and MODIS data now match the LST acquisition time
2. **Improved Accuracy**: Better temporal alignment leads to more accurate comparisons
3. **Automatic Selection**: No manual intervention required - system automatically determines appropriate times
4. **Backward Compatibility**: Existing workflows continue to work without modification
5. **Detailed Logging**: Clear information about time selection decisions

## Technical Details

### LST Time Extraction
- Reads DATETIME metadata from GeoTIFF tags
- Handles multiple datetime formats
- Falls back to default time (10:30 UTC) if metadata unavailable

### ERA5 Hourly Selection
- Uses Earth Engine's ERA5 collection
- Calculates hour difference between target and available times
- Selects the closest available hour

### MODIS Day/Night Logic
- Day: 6:00-18:00 UTC (LST_Day_1km)
- Night: 18:00-6:00 UTC (LST_Night_1km)
- Based on LST acquisition time, not local time

## Error Handling

- Graceful fallbacks when LST metadata is unavailable
- Detailed warning messages for debugging
- Continues processing even if individual files fail
- Maintains existing error handling patterns 