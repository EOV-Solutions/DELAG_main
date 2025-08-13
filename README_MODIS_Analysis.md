# MODIS Reconstruction Analysis Tool

A comprehensive Python tool for analyzing MODIS ground truth images against reconstructed images, designed for Land Surface Temperature (LST) validation and quality assessment.

## Features

### 🔍 **Image Filtering & Sorting**
- Automatically filters ground truth images with >25% non-NaN pixels (configurable)
- Sorts images by date for chronological analysis
- Handles different date formats between GT and reconstructed images

### 📊 **Visualization Generation**
- **Side-by-side comparisons**: GT vs reconstructed images in configurable batches
- **Time series plots**: Random pixel indices tracked over time
- **High-quality outputs**: 300 DPI PNG files with proper color scaling
- **Batch processing**: Configurable number of images per plot

### 📈 **Statistical Analysis**
- **RMSE calculation**: Root Mean Square Error for all valid pixels
- **MAE calculation**: Mean Absolute Error for all valid pixels
- **Per-date metrics**: Individual statistics for each date
- **Overall metrics**: Aggregated statistics across the entire dataset

### 💾 **Data Export**
- **JSON metrics**: Comprehensive results in structured format
- **Organized outputs**: Separate directories for visualizations and metrics
- **Reproducible analysis**: Configurable random seeds for consistent results

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Dependencies
- `rasterio` - Geospatial raster data handling
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars
- `pathlib` - Path handling

## Usage

### Basic Usage
```bash
python modis_reconstruction_analysis.py \
    --roi vietnam \
    --gt-folder data/retrieved_data/vietnam/movis \
    --recon-folder data/output/vietnam/reconstructed_lst_train \
    --output-dir results/vietnam
```

### Advanced Usage with Custom Parameters
```bash
python modis_reconstruction_analysis.py \
    --roi thailand \
    --gt-folder data/retrieved_data/thailand/movis \
    --recon-folder data/output/thailand/reconstructed_lst_train \
    --output-dir results/thailand \
    --min-percentage 30.0 \
    --images-per-plot 5 \
    --num-pixels 3
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--roi` | str | Required | Region of Interest identifier |
| `--gt-folder` | str | Required | Path to ground truth MODIS images |
| `--recon-folder` | str | Required | Path to reconstructed images |
| `--output-dir` | str | Required | Output directory for results |
| `--min-percentage` | float | 25.0 | Minimum % of non-NaN pixels for GT images |
| `--images-per-plot` | int | 10 | Number of images per comparison plot |
| `--num-pixels` | int | 5 | Number of random pixels for time series |

## Data Structure

### Expected Folder Structure
```
data/
├── retrieved_data/
│   └── {ROI}/
│       └── movis/
│           └── *_YYYY-MM-DD.tif          # Ground truth images
└── output/
    └── {ROI}/
        └── reconstructed_lst_train/
            └── *_YYYYMMDD.tif            # Reconstructed images
```

### File Naming Conventions
- **Ground Truth**: `*_YYYY-MM-DD.tif` (e.g., `lst16days_2020-01-01.tif`)
- **Reconstructed**: `*_YYYYMMDD.tif` (e.g., `LST_RECON_20200101.tif`)

## Output Structure

### Generated Directory Structure
```
results/{ROI}/
├── visualizations/
│   ├── comparison_batch_001.png
│   ├── comparison_batch_002.png
│   └── time_series_random_pixels.png
└── metrics/
    └── metrics_{ROI}.json
```

### Visualization Outputs

#### 1. Comparison Plots (`comparison_batch_*.png`)
- Side-by-side GT vs reconstructed images
- Configurable number of images per plot
- Consistent color scaling within each batch
- High-resolution (300 DPI) output

#### 2. Time Series Plots (`time_series_random_pixels.png`)
- Multiple subplots for different pixel locations
- GT and reconstructed values over time
- Proper date formatting on x-axis
- Grid lines and legends for clarity

### Metrics Output (`metrics_{ROI}.json`)

```json
{
  "roi": "vietnam",
  "analysis_date": "2024-01-15T10:30:00",
  "overall_metrics": {
    "overall_rmse": 2.3456,
    "overall_mae": 1.8765,
    "total_valid_pixels": 1500000,
    "total_dates_processed": 45
  },
  "date_metrics": {
    "2020-01-01": {
      "rmse": 2.1234,
      "mae": 1.6543,
      "valid_pixels": 50000,
      "total_pixels": 60000
    }
  },
  "filtered_dates_count": 50,
  "common_dates_count": 60
}
```

## Analysis Pipeline

### Step 1: File Discovery
- Scans GT and reconstructed folders
- Extracts dates from filenames using regex patterns
- Maps dates to file paths

### Step 2: Image Filtering
- Calculates non-NaN pixel percentage for each GT image
- Filters images above the minimum threshold (default: 25%)
- Sorts images chronologically

### Step 3: Visualization Generation
- Creates comparison plots in batches
- Generates time series for random pixel indices
- Applies consistent color scaling and formatting

### Step 4: Metrics Calculation
- Computes RMSE and MAE for each date
- Aggregates metrics across the entire dataset
- Handles NaN values appropriately

### Step 5: Results Export
- Saves visualizations as high-quality PNG files
- Exports comprehensive metrics to JSON format
- Provides summary statistics

## Example Use Cases

### 1. Quality Assessment
```bash
# High-quality analysis with strict filtering
python modis_reconstruction_analysis.py \
    --roi cambodia \
    --gt-folder data/retrieved_data/cambodia/movis \
    --recon-folder data/output/cambodia/reconstructed_lst_train \
    --output-dir results/cambodia \
    --min-percentage 50.0
```

### 2. Quick Overview
```bash
# Quick analysis with more images per plot
python modis_reconstruction_analysis.py \
    --roi vietnam \
    --gt-folder data/retrieved_data/vietnam/movis \
    --recon-folder data/output/vietnam/reconstructed_lst_train \
    --output-dir results/vietnam \
    --images-per-plot 15
```

### 3. Detailed Time Series
```bash
# Detailed time series analysis
python modis_reconstruction_analysis.py \
    --roi thailand \
    --gt-folder data/retrieved_data/thailand/movis \
    --recon-folder data/output/thailand/reconstructed_lst_train \
    --output-dir results/thailand \
    --num-pixels 10
```

## Performance Considerations

### Memory Usage
- Images are loaded one at a time to minimize memory usage
- Large datasets are processed in batches
- Progress bars show processing status

### Processing Time
- Depends on number of images and image size
- Typical processing: 100 images ≈ 5-10 minutes
- Parallel processing not implemented (single-threaded)

### Output Size
- High-resolution plots: ~2-5 MB each
- JSON metrics: ~10-50 KB
- Total output: ~100-500 MB for large datasets

## Troubleshooting

### Common Issues

#### 1. No Images Found
```
Error: No valid image files found!
```
**Solution**: Check file paths and naming conventions

#### 2. No Common Dates
```
Error: No common dates found between GT and reconstructed images!
```
**Solution**: Verify date formats in filenames

#### 3. Memory Errors
```
MemoryError: Unable to allocate array
```
**Solution**: Reduce `--images-per-plot` parameter

#### 4. Missing Dependencies
```
ModuleNotFoundError: No module named 'rasterio'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

### Debug Mode
For detailed debugging, modify the script to add more verbose output:
```python
# Add to the script for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Code Style
- Follow PEP 8 guidelines
- Use type hints for function parameters
- Include comprehensive docstrings
- Add error handling for robustness

### Testing
- Test with sample data before running on large datasets
- Verify output formats and metrics calculations
- Check memory usage with different parameter combinations

## License

This tool is part of the DELAG project for Land Surface Temperature analysis and reconstruction.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Verify data structure and file naming
3. Review command line parameters
4. Check system requirements and dependencies 