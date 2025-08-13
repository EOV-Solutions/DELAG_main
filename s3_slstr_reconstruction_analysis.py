#!/usr/bin/env python3
"""
S3 SLSTR LST Reconstruction Analysis Script

This script performs comprehensive analysis of S3 SLSTR LST ground truth images vs reconstructed images:
1. Sorts GT images by non-NaN pixel percentage (>25%)
2. Creates side-by-side comparison visualizations
3. Generates time series plots for random pixel indices
4. Calculates RMSE and MAE metrics for all valid pixels
5. Creates time-averaged error distribution, error maps, and correlation analysis

Author: ML Chemistry Expert
Date: 2024
"""

import os
import re
import json
import argparse
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

# Set matplotlib style for better plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
sns.set_palette("husl")


class S3SLSTRReconstructionAnalyzer:
    """Comprehensive analyzer for S3 SLSTR LST ground truth vs reconstructed images with time-averaged analysis."""
    
    def __init__(self, roi: str, gt_folder: str, recon_folder: str, output_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            roi: Region of Interest identifier
            gt_folder: Path to ground truth S3 SLSTR LST images
            recon_folder: Path to reconstructed images
            output_dir: Output directory for results
        """
        self.roi = roi
        self.gt_folder = Path(gt_folder)
        self.recon_folder = Path(recon_folder)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.viz_dir = self.output_dir / "visualizations"
        self.metrics_dir = self.output_dir / "metrics"
        self.advanced_viz_dir = self.output_dir / "advanced_visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.advanced_viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.gt_files = {}
        self.recon_files = {}
        self.common_dates = []
        self.filtered_dates = []
        
        # Regex patterns for S3 SLSTR LST date extraction
        self.gt_pattern = r"S3_SLSTR_LST_(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\.tif"
        self.recon_pattern = r".*_(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.tif"
    
    def find_image_files(self, folder_path: Path, pattern: str) -> Dict[date, Path]:
        """Find image files and extract dates based on regex pattern."""
        date_to_file_map = {}
        date_pattern = re.compile(pattern)
        
        print(f"Scanning folder: {folder_path}")
        if not folder_path.is_dir():
            print(f"Error: Folder not found at {folder_path}")
            return date_to_file_map
        
        for filename in folder_path.glob("*.tif"):
            match = date_pattern.match(filename.name)
            if match:
                parts = match.groupdict()
                try:
                    year = int(parts['year'])
                    month = int(parts['month'])
                    day = int(parts['day'])
                    date_obj = datetime(year, month, day).date()
                    date_to_file_map[date_obj] = filename
                except (ValueError, KeyError) as e:
                    print(f"Warning: Could not parse date from '{filename.name}'. Error: {e}")
        
        print(f"Found {len(date_to_file_map)} valid image files.")
        return date_to_file_map
    
    def calculate_non_nan_percentage(self, file_path: Path) -> float:
        """Calculate percentage of non-NaN pixels in an image."""
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                nodata_val = src.nodata
                if nodata_val is not None:
                    data[data == nodata_val] = np.nan
                
                total_pixels = data.size
                non_nan_pixels = np.sum(~np.isnan(data))
                percentage = (non_nan_pixels / total_pixels) * 100
                return percentage
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return 0.0
    
    def filter_gt_images(self, min_percentage: float = 25.0) -> List[date]:
        """Filter GT images to only include those with >25% non-NaN pixels."""
        print(f"\nFiltering GT images with >{min_percentage}% non-NaN pixels...")
        
        filtered_dates = []
        for date_obj, file_path in tqdm(self.gt_files.items(), desc="Analyzing GT images"):
            percentage = self.calculate_non_nan_percentage(file_path)
            if percentage > min_percentage:
                filtered_dates.append(date_obj)
        
        filtered_dates.sort()
        print(f"Found {len(filtered_dates)} GT images with >{min_percentage}% non-NaN pixels")
        return filtered_dates
    
    def load_image_data(self, file_path: Path) -> np.ndarray:
        """Load image data and handle nodata values."""
        with rasterio.open(file_path) as src:
            data = src.read(1)
            nodata_val = src.nodata
            if nodata_val is not None:
                data[data == nodata_val] = np.nan
            return data
    
    def create_comparison_visualizations(self, images_per_plot: int = 10):
        """Create side-by-side comparison visualizations."""
        print(f"\nCreating comparison visualizations ({images_per_plot} images per plot)...")
        
        # Group dates into batches
        date_batches = [self.filtered_dates[i:i + images_per_plot] 
                       for i in range(0, len(self.filtered_dates), images_per_plot)]
        
        for batch_idx, date_batch in enumerate(tqdm(date_batches, desc="Creating visualization batches")):
            self._create_batch_comparison_plot(date_batch, batch_idx)
    
    def _create_batch_comparison_plot(self, date_batch: List[date], batch_idx: int):
        """Create a single batch comparison plot."""
        n_images = len(date_batch)
        n_cols = 2  # GT and reconstructed side by side
        n_rows = n_images
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3 * n_rows))
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        # Determine global color scale for this batch
        all_valid_values = []
        for date_obj in date_batch:
            if date_obj in self.recon_files:
                gt_data = self.load_image_data(self.gt_files[date_obj])
                recon_data = self.load_image_data(self.recon_files[date_obj])
                
                all_valid_values.extend(gt_data[~np.isnan(gt_data)].flatten())
                all_valid_values.extend(recon_data[~np.isnan(recon_data)].flatten())
        
        if all_valid_values:
            vmin = np.percentile(all_valid_values, 2)
            vmax = np.percentile(all_valid_values, 98)
        else:
            vmin, vmax = 0, 1
        
        for row_idx, date_obj in enumerate(date_batch):
            if date_obj not in self.recon_files:
                continue
                
            # Load data
            gt_data = self.load_image_data(self.gt_files[date_obj])
            recon_data = self.load_image_data(self.recon_files[date_obj])
            
            # Plot GT image
            im1 = axes[row_idx, 0].imshow(gt_data, cmap='YlOrRd', vmin=vmin, vmax=vmax)
            axes[row_idx, 0].set_title(f"GT: {date_obj.strftime('%Y-%m-%d')}", fontsize=10)
            axes[row_idx, 0].set_xticks([])
            axes[row_idx, 0].set_yticks([])
            
            # Plot reconstructed image
            im2 = axes[row_idx, 1].imshow(recon_data, cmap='YlOrRd', vmin=vmin, vmax=vmax)
            axes[row_idx, 1].set_title(f"Reconstructed: {date_obj.strftime('%Y-%m-%d')}", fontsize=10)
            axes[row_idx, 1].set_xticks([])
            axes[row_idx, 1].set_yticks([])
        
        plt.suptitle(f'S3 SLSTR LST Comparison - Batch {batch_idx + 1} ({self.roi})', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        output_path = self.viz_dir / f"comparison_batch_{batch_idx + 1:03d}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def create_time_series_plot(self, num_pixels: int = 5):
        """Create time series plots for random pixel indices."""
        print(f"\nCreating time series plots for {num_pixels} random pixels...")
        
        # Get image dimensions from first GT image
        first_gt_file = next(iter(self.gt_files.values()))
        with rasterio.open(first_gt_file) as src:
            height, width = src.shape
        
        # Generate random pixel indices
        np.random.seed(42)  # For reproducibility
        pixel_indices = []
        for _ in range(num_pixels):
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            pixel_indices.append((row, col))
        
        # Create time series data
        time_series_data = {f"Pixel_{i+1}": {"dates": [], "gt_values": [], "recon_values": []} 
                           for i in range(num_pixels)}
        
        for date_obj in sorted(self.filtered_dates):
            if date_obj not in self.recon_files:
                continue
                
            gt_data = self.load_image_data(self.gt_files[date_obj])
            recon_data = self.load_image_data(self.recon_files[date_obj])
            
            for i, (row, col) in enumerate(pixel_indices):
                gt_val = gt_data[row, col]
                recon_val = recon_data[row, col]
                
                if not np.isnan(gt_val) and not np.isnan(recon_val):
                    time_series_data[f"Pixel_{i+1}"]["dates"].append(date_obj)
                    time_series_data[f"Pixel_{i+1}"]["gt_values"].append(gt_val)
                    time_series_data[f"Pixel_{i+1}"]["recon_values"].append(recon_val)
        
        # Create time series plots
        fig, axes = plt.subplots(num_pixels, 1, figsize=(12, 3 * num_pixels))
        if num_pixels == 1:
            axes = [axes]
        
        for i, (pixel_name, data) in enumerate(time_series_data.items()):
            if not data["dates"]:
                continue
                
            dates = [datetime.combine(d, datetime.min.time()) for d in data["dates"]]
            
            axes[i].plot(dates, data["gt_values"], 'o-', label='Ground Truth', 
                        color='blue', alpha=0.7, markersize=4)
            axes[i].plot(dates, data["recon_values"], 's-', label='Reconstructed', 
                        color='red', alpha=0.7, markersize=4)
            
            axes[i].set_title(f'{pixel_name} (Row: {pixel_indices[i][0]}, Col: {pixel_indices[i][1]})')
            axes[i].set_ylabel('LST (°K)')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Format x-axis
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axes[i].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'S3 SLSTR LST Time Series - Random Pixels ({self.roi})', fontsize=14)
        plt.tight_layout()
        
        # Save plot
        output_path = self.viz_dir / f"time_series_random_pixels.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    def create_time_averaged_error_distribution_plots(self, all_gt_data: List[np.ndarray], all_recon_data: List[np.ndarray]):
        """Create error distribution plots averaged across the entire time series."""
        print("Creating time-averaged error distribution plots...")
        
        # Collect all valid errors across time
        all_errors = []
        all_gt_values = []
        all_recon_values = []
        
        for gt_data, recon_data in zip(all_gt_data, all_recon_data):
            valid_mask = ~(np.isnan(gt_data) | np.isnan(recon_data))
            if np.sum(valid_mask) > 0:
                gt_valid = gt_data[valid_mask]
                recon_valid = recon_data[valid_mask]
                errors = gt_valid - recon_valid
                
                all_errors.extend(errors.tolist())
                all_gt_values.extend(gt_valid.tolist())
                all_recon_values.extend(recon_valid.tolist())
        
        if not all_errors:
            print("No valid data found for error distribution analysis")
            return
        
        all_errors = np.array(all_errors)
        all_gt_values = np.array(all_gt_values)
        all_recon_values = np.array(all_recon_values)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Time-Averaged Error Distribution Analysis ({self.roi})', fontsize=16)
        
        # 1. Error histogram
        axes[0, 0].hist(all_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(all_errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(all_errors):.3f}')
        axes[0, 0].axvline(np.median(all_errors), color='orange', linestyle='--', 
                          label=f'Median: {np.median(all_errors):.3f}')
        axes[0, 0].set_xlabel('Error (GT - Reconstructed)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Time-Averaged Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality
        try:
            stats.probplot(all_errors, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Normality Test)')
        except:
            axes[0, 1].text(0.5, 0.5, 'Q-Q plot not available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Q-Q Plot (Normality Test)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. GT vs Reconstructed scatter
        axes[1, 0].scatter(all_gt_values, all_recon_values, alpha=0.3, s=0.5)
        # Add 1:1 line
        min_val = min(all_gt_values.min(), all_recon_values.min())
        max_val = max(all_gt_values.max(), all_recon_values.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Ground Truth')
        axes[1, 0].set_ylabel('Reconstructed')
        try:
            r2_val = r2_score(all_gt_values, all_recon_values)
            if np.isnan(r2_val):
                r2_val = 0.0
        except:
            r2_val = 0.0
        axes[1, 0].set_title(f'Time-Averaged GT vs Reconstructed (R² = {r2_val:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error vs GT value
        axes[1, 1].scatter(all_gt_values, all_errors, alpha=0.3, s=0.5)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Ground Truth Value')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Time-Averaged Error vs GT Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.advanced_viz_dir / f'time_averaged_error_distribution_{self.roi}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_time_averaged_error_maps(self, all_gt_data: List[np.ndarray], all_recon_data: List[np.ndarray]):
        """Create spatial error maps averaged across the entire time series."""
        print("Creating time-averaged error maps...")
        
        if not all_gt_data:
            print("No data found for error map analysis")
            return
        
        # Get image dimensions
        height, width = all_gt_data[0].shape
        
        # Initialize accumulators
        error_sum = np.zeros((height, width))
        abs_error_sum = np.zeros((height, width))
        valid_count = np.zeros((height, width))
        
        # Accumulate errors across time
        for gt_data, recon_data in zip(all_gt_data, all_recon_data):
            valid_mask = ~(np.isnan(gt_data) | np.isnan(recon_data))
            errors = gt_data - recon_data
            
            error_sum[valid_mask] += errors[valid_mask]
            abs_error_sum[valid_mask] += np.abs(errors[valid_mask])
            valid_count[valid_mask] += 1
        
        # Calculate means
        mean_errors = np.full((height, width), np.nan)
        mean_abs_errors = np.full((height, width), np.nan)
        valid_mask = valid_count > 0
        
        mean_errors[valid_mask] = error_sum[valid_mask] / valid_count[valid_mask]
        mean_abs_errors[valid_mask] = abs_error_sum[valid_mask] / valid_count[valid_mask]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Time-Averaged Spatial Error Analysis ({self.roi})', fontsize=16)
        
        # 1. Mean error map
        error_max = np.nanmax(np.abs(mean_errors))
        im1 = axes[0].imshow(mean_errors, cmap='RdBu_r', vmin=-error_max, vmax=error_max)
        axes[0].set_title('Time-Averaged Error Map (GT - Reconstructed)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.colorbar(im1, ax=axes[0], label='Mean Error (°K)')
        
        # 2. Mean absolute error map
        im2 = axes[1].imshow(mean_abs_errors, cmap='viridis')
        axes[1].set_title('Time-Averaged Absolute Error Map')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(im2, ax=axes[1], label='Mean |Error| (°K)')
        
        # 3. Valid pixel count map
        im3 = axes[2].imshow(valid_count, cmap='plasma')
        axes[2].set_title('Valid Pixel Count Map')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        plt.colorbar(im3, ax=axes[2], label='Number of Valid Observations')
        
        plt.tight_layout()
        plt.savefig(self.advanced_viz_dir / f'time_averaged_error_maps_{self.roi}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_time_averaged_correlation_analysis(self, all_gt_data: List[np.ndarray], all_recon_data: List[np.ndarray]):
        """Create correlation analysis plots averaged across the entire time series."""
        print("Creating time-averaged correlation analysis...")
        
        # Collect all valid data across time
        all_gt_values = []
        all_recon_values = []
        
        for gt_data, recon_data in zip(all_gt_data, all_recon_data):
            valid_mask = ~(np.isnan(gt_data) | np.isnan(recon_data))
            if np.sum(valid_mask) > 0:
                gt_valid = gt_data[valid_mask]
                recon_valid = recon_data[valid_mask]
                
                all_gt_values.extend(gt_valid.tolist())
                all_recon_values.extend(recon_valid.tolist())
        
        if not all_gt_values:
            print("No valid data found for correlation analysis")
            return
        
        all_gt_values = np.array(all_gt_values)
        all_recon_values = np.array(all_recon_values)
        
        # Calculate overall correlation
        try:
            correlation = np.corrcoef(all_gt_values.flatten(), all_recon_values.flatten())[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Time-Averaged Correlation Analysis ({self.roi})', fontsize=16)
        
        # 1. Scatter plot with regression
        axes[0].scatter(all_gt_values, all_recon_values, alpha=0.3, s=0.5)
        
        # Add regression line
        try:
            z = np.polyfit(all_gt_values, all_recon_values, 1)
            p = np.poly1d(z)
            axes[0].plot(all_gt_values, p(all_gt_values), "r--", alpha=0.8, linewidth=2)
        except:
            # If regression fails, just plot the 1:1 line
            pass
        
        # Add 1:1 line
        min_val = min(all_gt_values.min(), all_recon_values.min())
        max_val = max(all_gt_values.max(), all_recon_values.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        axes[0].set_xlabel('Ground Truth (°K)')
        axes[0].set_ylabel('Reconstructed (°K)')
        axes[0].set_title(f'Time-Averaged Correlation: {correlation:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Time-averaged correlation map (spatial correlation)
        # Calculate correlation for each pixel across time
        if all_gt_data:
            height, width = all_gt_data[0].shape
            corr_map = np.full((height, width), np.nan)
            
            for i in range(height):
                for j in range(width):
                    # Collect time series for this pixel
                    pixel_gt = []
                    pixel_recon = []
                    
                    for gt_data, recon_data in zip(all_gt_data, all_recon_data):
                        if not np.isnan(gt_data[i, j]) and not np.isnan(recon_data[i, j]):
                            pixel_gt.append(gt_data[i, j])
                            pixel_recon.append(recon_data[i, j])
                    
                    # Calculate correlation if we have enough data points
                    if len(pixel_gt) > 5:  # Require at least 5 observations
                        try:
                            corr = np.corrcoef(pixel_gt, pixel_recon)[0, 1]
                            if not np.isnan(corr):
                                corr_map[i, j] = corr
                        except:
                            pass
            
            im = axes[1].imshow(corr_map, cmap='RdYlBu_r', vmin=-1, vmax=1)
            axes[1].set_title('Time-Averaged Spatial Correlation Map')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            plt.colorbar(im, ax=axes[1], label='Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(self.advanced_viz_dir / f'time_averaged_correlation_analysis_{self.roi}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_seasonal_analysis(self, date_metrics: Dict):
        """Create seasonal performance analysis."""
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame.from_dict(date_metrics, orient='index')
        df.index = pd.to_datetime(df.index)
        df['month'] = df.index.month
        df['season'] = df.index.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Seasonal Performance Analysis - {self.roi}', fontsize=16)
        
        # 1. Monthly RMSE box plot
        monthly_data = [df[df['month'] == month]['rmse'].values for month in range(1, 13)]
        axes[0, 0].boxplot(monthly_data, labels=[f'M{month}' for month in range(1, 13)])
        axes[0, 0].set_title('RMSE by Month')
        axes[0, 0].set_ylabel('RMSE (°K)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Seasonal RMSE box plot
        seasonal_data = [df[df['season'] == season]['rmse'].values for season in ['Winter', 'Spring', 'Summer', 'Fall']]
        axes[0, 1].boxplot(seasonal_data, labels=['Winter', 'Spring', 'Summer', 'Fall'])
        axes[0, 1].set_title('RMSE by Season')
        axes[0, 1].set_ylabel('RMSE (°K)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly MAE trend
        monthly_mae = df.groupby('month')['mae'].mean()
        axes[1, 0].plot(monthly_mae.index, monthly_mae.values, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Average MAE by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('MAE (°K)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Valid pixels percentage by month
        monthly_valid = df.groupby('month').apply(
            lambda x: (x['valid_pixels'] / x['total_pixels'] * 100).mean()
        )
        axes[1, 1].plot(monthly_valid.index, monthly_valid.values, 's-', 
                       linewidth=2, markersize=8, color='green')
        axes[1, 1].set_title('Average Valid Pixels % by Month')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Valid Pixels (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.advanced_viz_dir / f'seasonal_analysis_{self.roi}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_dashboard(self, overall_metrics: Dict, date_metrics: Dict):
        """Create comprehensive performance dashboard."""
        df = pd.DataFrame.from_dict(date_metrics, orient='index')
        df.index = pd.to_datetime(df.index)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall metrics summary
        ax1 = fig.add_subplot(gs[0, :2])
        metrics_text = f"""
        Overall Performance Summary ({self.roi})
        
        RMSE: {overall_metrics['overall_rmse']:.3f} °K
        MAE: {overall_metrics['overall_mae']:.3f} °K
        Total Valid Pixels: {overall_metrics['total_valid_pixels']:,}
        Dates Processed: {overall_metrics['total_dates_processed']}
        """
        ax1.text(0.1, 0.5, metrics_text, transform=ax1.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.5))
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # 2. RMSE time series
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.plot(df.index, df['rmse'], 'o-', alpha=0.7, markersize=3)
        ax2.set_title('RMSE Over Time')
        ax2.set_ylabel('RMSE (°K)')
        ax2.grid(True, alpha=0.3)
        
        # 3. MAE time series
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(df.index, df['mae'], 's-', alpha=0.7, markersize=3, color='orange')
        ax3.set_title('MAE Over Time')
        ax3.set_ylabel('MAE (°K)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Valid pixels percentage
        ax4 = fig.add_subplot(gs[1, 2:])
        valid_percentage = (df['valid_pixels'] / df['total_pixels'] * 100)
        ax4.plot(df.index, valid_percentage, '^-', alpha=0.7, markersize=3, color='green')
        ax4.set_title('Valid Pixels Percentage Over Time')
        ax4.set_ylabel('Valid Pixels (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Error distribution histogram
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.hist(df['rmse'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_title('RMSE Distribution')
        ax5.set_xlabel('RMSE (°K)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance correlation
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.scatter(df['rmse'], df['mae'], alpha=0.6)
        ax6.set_xlabel('RMSE (°K)')
        ax6.set_ylabel('MAE (°K)')
        ax6.set_title('RMSE vs MAE Correlation')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Performance Dashboard - {self.roi}', fontsize=16)
        plt.savefig(self.advanced_viz_dir / f'performance_dashboard_{self.roi}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_metrics(self, generate_advanced_viz: bool = True) -> Dict:
        """Calculate RMSE and MAE for all valid pixels across the dataset."""
        print("\nCalculating RMSE and MAE metrics...")
        
        all_gt_values = []
        all_recon_values = []
        date_metrics = {}
        
        # Collect all data for time-averaged analysis
        all_gt_data = []
        all_recon_data = []
        
        for date_obj in tqdm(self.filtered_dates, desc="Processing dates"):
            if date_obj not in self.recon_files:
                continue
                
            gt_data = self.load_image_data(self.gt_files[date_obj])
            recon_data = self.load_image_data(self.recon_files[date_obj])
            
            # Find valid pixels (non-NaN in both images)
            valid_mask = ~(np.isnan(gt_data) | np.isnan(recon_data))
            
            if np.sum(valid_mask) == 0:
                continue
            
            gt_valid = gt_data[valid_mask]
            recon_valid = recon_data[valid_mask]
            
            # Calculate metrics for this date
            mse = np.mean((gt_valid - recon_valid) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(gt_valid - recon_valid))
            
            date_str = date_obj.strftime('%Y-%m-%d')
            date_metrics[date_str] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'valid_pixels': int(np.sum(valid_mask)),
                'total_pixels': int(gt_data.size)
            }
            
            # Store data for time-averaged analysis
            all_gt_data.append(gt_data)
            all_recon_data.append(recon_data)
            
            # Store for overall metrics
            all_gt_values.extend(gt_valid.tolist())
            all_recon_values.extend(recon_valid.tolist())
        
        # Generate time-averaged advanced visualizations
        if generate_advanced_viz and all_gt_data:
            print("\nGenerating time-averaged advanced visualizations...")
            self.create_time_averaged_error_distribution_plots(all_gt_data, all_recon_data)
            self.create_time_averaged_error_maps(all_gt_data, all_recon_data)
            self.create_time_averaged_correlation_analysis(all_gt_data, all_recon_data)
        
        # Calculate overall metrics
        if all_gt_values:
            all_gt_array = np.array(all_gt_values)
            all_recon_array = np.array(all_recon_values)
            
            overall_mse = np.mean((all_gt_array - all_recon_array) ** 2)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = np.mean(np.abs(all_gt_array - all_recon_array))
            
            overall_metrics = {
                'overall_rmse': float(overall_rmse),
                'overall_mae': float(overall_mae),
                'total_valid_pixels': len(all_gt_values),
                'total_dates_processed': len(date_metrics)
            }
        else:
            overall_metrics = {
                'overall_rmse': 0.0,
                'overall_mae': 0.0,
                'total_valid_pixels': 0,
                'total_dates_processed': 0
            }
        
        # Combine all metrics
        results = {
            'roi': self.roi,
            'analysis_date': datetime.now().isoformat(),
            'overall_metrics': overall_metrics,
            'date_metrics': date_metrics,
            'filtered_dates_count': len(self.filtered_dates),
            'common_dates_count': len(self.common_dates)
        }
        
        return results
    
    def save_metrics(self, metrics: Dict):
        """Save metrics to JSON file."""
        output_path = self.metrics_dir / f"metrics_{self.roi}.json"
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"\nMetrics saved to: {output_path}")
        
        # Print summary
        overall = metrics['overall_metrics']
        print(f"\n=== SUMMARY FOR {self.roi.upper()} ===")
        print(f"Overall RMSE: {overall['overall_rmse']:.4f} °K")
        print(f"Overall MAE: {overall['overall_mae']:.4f} °K")
        print(f"Total valid pixels: {overall['total_valid_pixels']:,}")
        print(f"Dates processed: {overall['total_dates_processed']}")
        print(f"Filtered dates: {metrics['filtered_dates_count']}")
    
    def run_analysis(self, min_percentage: float = 25.0, images_per_plot: int = 10, 
                    num_pixels: int = 5, advanced_viz: bool = True):
        """Run the complete analysis pipeline."""
        print(f"=== S3 SLSTR LST Reconstruction Analysis for {self.roi} ===")
        
        # Step 1: Find image files
        print("\nStep 1: Finding image files...")
        self.gt_files = self.find_image_files(self.gt_folder, self.gt_pattern)
        self.recon_files = self.find_image_files(self.recon_folder, self.recon_pattern)
        
        if not self.gt_files or not self.recon_files:
            print("Error: No valid image files found!")
            return
        
        # Step 2: Find common dates
        self.common_dates = sorted(list(set(self.gt_files.keys()) & set(self.recon_files.keys())))
        print(f"Found {len(self.common_dates)} common dates between GT and reconstructed images")
        
        if not self.common_dates:
            print("Error: No common dates found between GT and reconstructed images!")
            return
        
        # Step 3: Filter GT images by non-NaN percentage
        self.filtered_dates = self.filter_gt_images(min_percentage)
        
        if not self.filtered_dates:
            print(f"Error: No GT images found with >{min_percentage}% non-NaN pixels!")
            return
        
        # Step 4: Create visualizations
        self.create_comparison_visualizations(images_per_plot)
        self.create_time_series_plot(num_pixels)
        
        # Step 5: Calculate metrics
        metrics = self.calculate_metrics(generate_advanced_viz=advanced_viz)
        self.save_metrics(metrics)
        
        # Step 6: Create advanced summary visualizations
        if advanced_viz:
            print("\nStep 6: Creating advanced summary visualizations...")
            self.create_seasonal_analysis(metrics['date_metrics'])
            self.create_performance_dashboard(metrics['overall_metrics'], metrics['date_metrics'])
        
        print(f"\n=== Analysis Complete ===")
        print(f"Results saved to: {self.output_dir}")
        if advanced_viz:
            print(f"Time-averaged advanced visualizations saved to: {self.advanced_viz_dir}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive S3 SLSTR LST reconstruction analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python s3_slstr_reconstruction_analysis.py --roi vietnam --gt-folder data/retrieved_data/vietnam/s3_slstr --recon-folder data/output/vietnam/reconstructed_lst_train --output-dir results/vietnam
  
  python s3_slstr_reconstruction_analysis.py --roi thailand --min-percentage 30 --images-per-plot 5 --num-pixels 3
  
  python s3_slstr_reconstruction_analysis.py --roi cambodia --no-advanced-viz  # Disable advanced visualizations
        """
    )
    
    parser.add_argument("--roi", required=True, help="Region of Interest identifier")
    parser.add_argument("--gt-folder", required=True, 
                       help="Path to ground truth S3 SLSTR LST images folder")
    parser.add_argument("--recon-folder", required=True,
                       help="Path to reconstructed images folder")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory for results")
    parser.add_argument("--min-percentage", type=float, default=25.0,
                       help="Minimum percentage of non-NaN pixels for GT images (default: 25.0)")
    parser.add_argument("--images-per-plot", type=int, default=10,
                       help="Number of images per comparison plot (default: 10)")
    parser.add_argument("--num-pixels", type=int, default=5,
                       help="Number of random pixels for time series plots (default: 5)")
    parser.add_argument("--advanced-viz", action="store_true", default=True,
                       help="Generate time-averaged advanced visualizations (default: True)")
    parser.add_argument("--no-advanced-viz", action="store_true",
                       help="Disable time-averaged advanced visualizations")
    
    args = parser.parse_args()
    
    # Handle advanced visualization flag
    if args.no_advanced_viz:
        args.advanced_viz = False
    
    # Create analyzer and run analysis
    analyzer = S3SLSTRReconstructionAnalyzer(
        roi=args.roi,
        gt_folder=args.gt_folder,
        recon_folder=args.recon_folder,
        output_dir=args.output_dir
    )
    
    analyzer.run_analysis(
        min_percentage=args.min_percentage,
        images_per_plot=args.images_per_plot,
        num_pixels=args.num_pixels,
        advanced_viz=args.advanced_viz
    )


if __name__ == "__main__":
    main() 