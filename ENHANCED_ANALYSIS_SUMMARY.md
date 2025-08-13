# Enhanced MODIS Reconstruction Analysis

## Overview

The `modis_reconstruction_analysis.py` script has been enhanced with comprehensive advanced visualizations that provide deep insights into MODIS LST reconstruction performance. All visualizations now use **actual data** from your analysis pipeline instead of example data.

## 🆕 **New Advanced Visualizations**

### **1. Error Distribution Analysis** (`error_distribution_YYYY-MM-DD.png`)
For each date, generates a 2x2 panel showing:
- **Error Histogram**: Distribution of GT - Reconstructed errors with mean/median lines
- **Q-Q Plot**: Normality test for error distribution
- **GT vs Reconstructed Scatter**: With R² value and 1:1 line
- **Error vs GT Value**: Shows if errors depend on temperature magnitude

### **2. Spatial Error Maps** (`error_maps_YYYY-MM-DD.png`)
Three-panel spatial analysis:
- **Error Map**: Spatial distribution of errors (red-blue diverging colormap)
- **Absolute Error Map**: Magnitude of errors (viridis colormap)
- **Valid Pixels Mask**: Shows which pixels have valid data

### **3. Correlation Analysis** (`correlation_analysis_YYYY-MM-DD.png`)
Two-panel correlation study:
- **Scatter Plot with Regression**: GT vs reconstructed with trend line
- **Spatial Correlation Map**: Local correlation coefficients across the image

### **4. Seasonal Performance Analysis** (`seasonal_analysis_{ROI}.png`)
Comprehensive seasonal breakdown:
- **Monthly RMSE Box Plots**: Performance variation by month
- **Seasonal RMSE Box Plots**: Performance by season (Winter/Spring/Summer/Fall)
- **Monthly MAE Trends**: Average MAE over months
- **Valid Pixels by Month**: Data availability patterns

### **5. Performance Dashboard** (`performance_dashboard_{ROI}.png`)
Complete overview dashboard:
- **Overall Metrics Summary**: Key statistics in text box
- **RMSE Time Series**: Performance over time
- **MAE Time Series**: Error trends
- **Valid Pixels Percentage**: Data coverage over time
- **RMSE Distribution**: Histogram of daily RMSE values
- **RMSE vs MAE Correlation**: Relationship between error metrics

## 📁 **Enhanced Output Structure**

```
results/{ROI}/
├── visualizations/                    # Basic visualizations
│   ├── comparison_batch_001.png
│   ├── comparison_batch_002.png
│   └── time_series_random_pixels.png
├── advanced_visualizations/           # NEW: Advanced analysis
│   ├── error_distribution_2020-01-01.png
│   ├── error_maps_2020-01-01.png
│   ├── correlation_analysis_2020-01-01.png
│   ├── seasonal_analysis_{ROI}.png
│   └── performance_dashboard_{ROI}.png
└── metrics/
    └── metrics_{ROI}.json
```

## 🚀 **Usage Examples**

### **Basic Analysis with Advanced Visualizations** (Default)
```bash
python modis_reconstruction_analysis.py \
    --roi vietnam \
    --gt-folder data/retrieved_data/vietnam/movis \
    --recon-folder data/output/vietnam/reconstructed_lst_train \
    --output-dir results/vietnam
```

### **Custom Parameters with Advanced Visualizations**
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

### **Disable Advanced Visualizations** (Faster Processing)
```bash
python modis_reconstruction_analysis.py \
    --roi cambodia \
    --gt-folder data/retrieved_data/cambodia/movis \
    --recon-folder data/output/cambodia/reconstructed_lst_train \
    --output-dir results/cambodia \
    --no-advanced-viz
```

## 🔧 **New Command Line Options**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--advanced-viz` | flag | True | Enable advanced visualizations |
| `--no-advanced-viz` | flag | False | Disable advanced visualizations |

## 📊 **Key Insights from Advanced Visualizations**

### **Error Distribution Analysis**
- **Normality**: Q-Q plots show if errors follow normal distribution
- **Bias Detection**: Histogram mean/median reveal systematic bias
- **Temperature Dependence**: Error vs GT plots show if errors vary with temperature
- **Correlation Strength**: R² values indicate reconstruction quality

### **Spatial Error Maps**
- **Geographic Patterns**: Identify regions with poor reconstruction
- **Error Magnitude**: Absolute error maps highlight problematic areas
- **Data Coverage**: Valid pixel masks show data availability

### **Seasonal Analysis**
- **Seasonal Patterns**: Performance variation across seasons
- **Monthly Trends**: Identify months with better/worse performance
- **Data Availability**: Seasonal patterns in valid pixel coverage

### **Performance Dashboard**
- **Overall Assessment**: Comprehensive performance summary
- **Temporal Trends**: Performance evolution over time
- **Metric Relationships**: Correlation between different error measures

## ⚡ **Performance Considerations**

### **Processing Time**
- **With Advanced Visualizations**: ~2-3x longer than basic analysis
- **Without Advanced Visualizations**: Same speed as original script
- **Per-Date Visualizations**: Generated for each valid date

### **Storage Requirements**
- **Basic Analysis**: ~50-100 MB for typical dataset
- **With Advanced Visualizations**: ~200-500 MB for typical dataset
- **High-Resolution Outputs**: 300 DPI PNG files

### **Memory Usage**
- **Efficient Processing**: Images loaded one at a time
- **Batch Processing**: Visualizations generated in batches
- **Progress Tracking**: tqdm progress bars for all operations

## 🎯 **Scientific Applications**

### **Model Validation**
- **Statistical Rigor**: Comprehensive error analysis
- **Spatial Assessment**: Geographic performance patterns
- **Temporal Analysis**: Seasonal and long-term trends

### **Publication Quality**
- **High-Resolution Outputs**: 300 DPI publication-ready figures
- **Professional Layout**: Consistent formatting and styling
- **Comprehensive Coverage**: Multiple analysis perspectives

### **Research Insights**
- **Error Characterization**: Detailed error distribution analysis
- **Performance Patterns**: Seasonal and spatial performance variations
- **Quality Assessment**: Multi-dimensional performance evaluation

## 🔄 **Integration with Existing Workflow**

The enhanced script is **fully backward compatible**:
- All existing functionality preserved
- New visualizations are optional
- Same command-line interface
- Same output structure (with additions)

## 📈 **Benefits for Your Research**

1. **Comprehensive Analysis**: Multiple perspectives on reconstruction quality
2. **Publication Ready**: High-quality visualizations for papers
3. **Deep Insights**: Statistical and spatial error analysis
4. **Seasonal Understanding**: Performance patterns across time
5. **Quality Assessment**: Multi-dimensional performance evaluation
6. **Flexible Usage**: Enable/disable advanced features as needed

The enhanced script now provides a complete toolkit for MODIS LST reconstruction analysis, combining basic comparisons with advanced statistical and spatial analysis techniques. 