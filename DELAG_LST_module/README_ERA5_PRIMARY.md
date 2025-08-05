# ERA5-Primary Timeline for DELAG Pipeline

## Overview

This document describes the new **ERA5-primary timeline** approach for the DELAG LST reconstruction pipeline. Instead of using LST observation dates as the primary timeline, this approach uses **ERA5 dates** as the primary timeline and creates **synthetic LST images** for missing dates via temporal interpolation.

## Key Differences

| Aspect | Original (LST Primary) | New (ERA5 Primary) |
|--------|----------------------|-------------------|
| **Primary Timeline** | LST observation dates | ERA5 dates |
| **Missing LST Dates** | Skipped (no reconstruction) | Filled with synthetic LST via interpolation |
| **Reconstruction Coverage** | Only LST observation dates | ALL ERA5 dates |
| **Timeline Source** | Sparse (LST observations) | Dense (ERA5 daily data) |

## Benefits

1. **Complete Temporal Coverage**: Reconstruction images cover ALL ERA5 dates, providing continuous time series
2. **No Missing Days**: No gaps in the reconstruction timeline due to missing LST observations
3. **Better Model Training**: More training data available due to synthetic LST gap-filling
4. **Consistent Time Series**: Uniform temporal sampling for analysis

## Implementation

### New Functions Added

#### 1. `load_era5_as_primary_timeline()`
- Loads ERA5 data to establish the primary timeline
- Applies date filtering based on config
- Handles ERA5 interpolation if enabled

#### 2. `create_synthetic_lst_for_era5_timeline()`
- Loads available LST data for ERA5 dates
- Creates synthetic LST for missing dates via temporal interpolation
- Applies LST outlier removal
- Returns LST stack aligned to ERA5 timeline

#### 3. `preprocess_all_data_era5_primary()`
- Alternative preprocessing function using ERA5 as primary timeline
- Integrates all data types to ERA5 timeline
- Adds metadata indicating timeline source

## Usage

### Option 1: Direct Command Line

```bash
# Use ERA5 as primary timeline
python preprocess_data.py --roi_name YOUR_ROI --timeline_mode era5_primary

# Use original LST-based timeline  
python preprocess_data.py --roi_name YOUR_ROI --timeline_mode lst
```

### Option 2: Full Pipeline Script

```bash
# Run complete pipeline with ERA5 primary timeline
python run_era5_primary_pipeline.py --roi_name YOUR_ROI --data_split test

# Skip training (if models already trained)
python run_era5_primary_pipeline.py --roi_name YOUR_ROI --data_split test --skip_training
```

## Technical Details

### Synthetic LST Creation Process

1. **Load Available LST**: Load actual LST observations for dates where they exist
2. **Temporal Interpolation**: Use pandas linear interpolation to fill missing dates:
   ```python
   pixel_series = pd.Series(lst_stack_aligned[:, r, c])
   lst_stack_aligned[:, r, c] = pixel_series.interpolate(method='linear', limit_direction='both')
   ```
3. **Quality Control**: Apply outlier removal after interpolation
4. **Spatial Coherence**: Interpolation preserves spatial patterns

### Data Flow Changes

```
Original Pipeline:
ERA5 → Interpolate to LST dates → LST timeline (sparse)

ERA5-Primary Pipeline:  
ERA5 → ERA5 timeline (dense) ← LST + Synthetic LST
```

### Configuration Options

The approach respects existing config options:

- `START_DATE`/`END_DATE`: Filter ERA5 dates
- `INTERPOLATE_ERA5`: Apply ERA5 interpolation
- `INTERPOLATE_S2`: Apply S2 interpolation
- `LST_OUTLIER_METHOD`: Apply to real + synthetic LST
- `SPATIAL_TRAINING_SAMPLE_PERCENTAGE`: Spatial sampling

## Output Differences

### Metadata Changes

The preprocessed data includes additional metadata:

```json
{
  "timeline_source": "era5_primary",
  "common_dates": ["ERA5", "dates", "list"],
  ...
}
```

### Reconstruction Files

**Original approach**: 
```
LST_RECON_TEST_20210105.tif  # Only LST observation dates
LST_RECON_TEST_20210112.tif
LST_RECON_TEST_20210225.tif
```

**ERA5-primary approach**:
```
LST_RECON_TEST_20210101.tif  # ALL ERA5 dates
LST_RECON_TEST_20210102.tif
LST_RECON_TEST_20210103.tif
LST_RECON_TEST_20210104.tif
LST_RECON_TEST_20210105.tif  # Includes both real and synthetic LST dates
...
```

## Validation Considerations

### Synthetic LST Quality

1. **Temporal Smoothness**: Linear interpolation creates smooth temporal transitions
2. **Physical Realism**: May not capture sudden weather events between observations
3. **Spatial Consistency**: Preserves spatial patterns from neighboring observations

### Model Training Impact

1. **More Training Data**: Synthetic LST provides additional training samples
2. **Potential Overfitting**: Models might learn interpolation patterns
3. **Real vs Synthetic**: Consider weighting real observations higher during training

## Best Practices

### When to Use ERA5-Primary

✅ **Recommended for**:
- Applications requiring complete temporal coverage
- Time series analysis needing regular intervals
- Studies where missing data is problematic
- Regions with frequent cloud cover

❌ **Consider alternatives for**:
- Studies focused on specific observation quality
- Applications where synthetic data introduces bias
- Short time periods with dense LST coverage

### Quality Assessment

1. **Compare Timelines**: Check date counts between approaches
2. **Validate Synthetic LST**: Compare interpolated vs real LST where available
3. **Model Performance**: Evaluate reconstruction quality on real observation dates
4. **Temporal Analysis**: Assess continuity of time series

## Example Usage

```bash
# 1. Preprocess with ERA5 primary timeline
python preprocess_data.py \
    --roi_name "D-49-49-A-c-2" \
    --timeline_mode era5_primary \
    --train_start "2017-03-01" \
    --train_end "2025-04-12" \
    --test_start "2021-01-01" \
    --test_end "2022-01-01"

# 2. Run standard training and prediction pipeline
python train_atc.py --roi_name "test_region"
python train_gp.py --roi_name "test_region"
python predict_atc.py --roi_name "test_region" --data_split test
python predict_gp.py --roi_name "test_region" --data_split test
python reconstruction.py --roi_name "test_region" --data_split test

# 3. Check results
ls output/test_region/reconstructed_lst/  # Should see ALL ERA5 dates
cat output/test_region/data_split_info.json  # Check timeline_mode: "era5_primary"
```

## Troubleshooting

### Common Issues

1. **No ERA5 Data**: Ensure ERA5 files exist in the expected directory
2. **Memory Issues**: ERA5-primary creates larger datasets - monitor memory usage
3. **Interpolation Failures**: Check for sufficient LST observations for interpolation
4. **Date Parsing**: Ensure ERA5 filenames contain parseable dates (YYYYMMDD format)

### Debugging

```bash
# Check timeline information
python -c "
import json
with open('output/YOUR_ROI/data_split_info.json') as f:
    info = json.load(f)
    print(f'Timeline mode: {info[\"timeline_mode\"]}')
    print(f'Test date count: {info[\"test_count\"]}')
    print(f'First date: {info[\"test_dates\"][0]}')
    print(f'Last date: {info[\"test_dates\"][-1]}')
"
```

## Conclusion

The ERA5-primary timeline approach provides complete temporal coverage for LST reconstruction by intelligently filling missing LST observation dates with synthetic data. This ensures that reconstruction images are available for all ERA5 dates, enabling continuous time series analysis while maintaining the scientific integrity of the reconstruction process. 