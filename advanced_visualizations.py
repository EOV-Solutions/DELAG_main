#!/usr/bin/env python3
"""
Advanced Visualization Module for MODIS Reconstruction Analysis

This module provides additional visualization capabilities beyond the basic
side-by-side comparisons and time series plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizations:
    """Advanced visualization methods for MODIS reconstruction analysis."""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.viz_dir = output_dir / "advanced_visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def create_error_distribution_plots(self, gt_data, recon_data, date_str, roi):
        """Create comprehensive error distribution plots."""
        # Calculate errors
        valid_mask = ~(np.isnan(gt_data) | np.isnan(recon_data))
        gt_valid = gt_data[valid_mask]
        recon_valid = recon_data[valid_mask]
        errors = gt_valid - recon_valid
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Error Distribution Analysis - {date_str} ({roi})', fontsize=16)
        
        # 1. Error histogram
        axes[0, 0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(errors):.3f}')
        axes[0, 0].axvline(np.median(errors), color='orange', linestyle='--', 
                          label=f'Median: {np.median(errors):.3f}')
        axes[0, 0].set_xlabel('Error (GT - Reconstructed)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot for normality
        stats.probplot(errors, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normality Test)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. GT vs Reconstructed scatter
        axes[1, 0].scatter(gt_valid, recon_valid, alpha=0.5, s=1)
        # Add 1:1 line
        min_val = min(gt_valid.min(), recon_valid.min())
        max_val = max(gt_valid.max(), recon_valid.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 0].set_xlabel('Ground Truth')
        axes[1, 0].set_ylabel('Reconstructed')
        axes[1, 0].set_title(f'GT vs Reconstructed (R² = {r2_score(gt_valid, recon_valid):.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error vs GT value
        axes[1, 1].scatter(gt_valid, errors, alpha=0.5, s=1)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Ground Truth Value')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Error vs GT Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'error_distribution_{date_str}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_error_maps(self, gt_data, recon_data, date_str, roi):
        """Create spatial error maps."""
        errors = gt_data - recon_data
        valid_mask = ~(np.isnan(gt_data) | np.isnan(recon_data))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Spatial Error Analysis - {date_str} ({roi})', fontsize=16)
        
        # 1. Error map
        im1 = axes[0].imshow(errors, cmap='RdBu_r', center=0)
        axes[0].set_title('Error Map (GT - Reconstructed)')
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        plt.colorbar(im1, ax=axes[0], label='Error (°C)')
        
        # 2. Absolute error map
        abs_errors = np.abs(errors)
        abs_errors[~valid_mask] = np.nan
        im2 = axes[1].imshow(abs_errors, cmap='viridis')
        axes[1].set_title('Absolute Error Map')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(im2, ax=axes[1], label='|Error| (°C)')
        
        # 3. Valid pixel mask
        im3 = axes[2].imshow(valid_mask, cmap='binary')
        axes[2].set_title('Valid Pixels Mask')
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        plt.colorbar(im3, ax=axes[2], label='Valid Pixels')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'error_maps_{date_str}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_seasonal_analysis(self, date_metrics, roi):
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
        fig.suptitle(f'Seasonal Performance Analysis - {roi}', fontsize=16)
        
        # 1. Monthly RMSE box plot
        monthly_data = [df[df['month'] == month]['rmse'].values for month in range(1, 13)]
        axes[0, 0].boxplot(monthly_data, labels=[f'M{month}' for month in range(1, 13)])
        axes[0, 0].set_title('RMSE by Month')
        axes[0, 0].set_ylabel('RMSE (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Seasonal RMSE box plot
        seasonal_data = [df[df['season'] == season]['rmse'].values for season in ['Winter', 'Spring', 'Summer', 'Fall']]
        axes[0, 1].boxplot(seasonal_data, labels=['Winter', 'Spring', 'Summer', 'Fall'])
        axes[0, 1].set_title('RMSE by Season')
        axes[0, 1].set_ylabel('RMSE (°C)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Monthly MAE trend
        monthly_mae = df.groupby('month')['mae'].mean()
        axes[1, 0].plot(monthly_mae.index, monthly_mae.values, 'o-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Average MAE by Month')
        axes[1, 0].set_xlabel('Month')
        axes[1, 0].set_ylabel('MAE (°C)')
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
        plt.savefig(self.viz_dir / f'seasonal_analysis_{roi}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_dashboard(self, overall_metrics, date_metrics, roi):
        """Create comprehensive performance dashboard."""
        df = pd.DataFrame.from_dict(date_metrics, orient='index')
        df.index = pd.to_datetime(df.index)
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Overall metrics summary
        ax1 = fig.add_subplot(gs[0, :2])
        metrics_text = f"""
        Overall Performance Summary ({roi})
        
        RMSE: {overall_metrics['overall_rmse']:.3f} °C
        MAE: {overall_metrics['overall_mae']:.3f} °C
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
        ax2.set_ylabel('RMSE (°C)')
        ax2.grid(True, alpha=0.3)
        
        # 3. MAE time series
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(df.index, df['mae'], 's-', alpha=0.7, markersize=3, color='orange')
        ax3.set_title('MAE Over Time')
        ax3.set_ylabel('MAE (°C)')
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
        # Collect all errors (simplified - would need actual data)
        ax5.hist(df['rmse'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.set_title('RMSE Distribution')
        ax5.set_xlabel('RMSE (°C)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance correlation
        ax6 = fig.add_subplot(gs[2, 2:])
        ax6.scatter(df['rmse'], df['mae'], alpha=0.6)
        ax6.set_xlabel('RMSE (°C)')
        ax6.set_ylabel('MAE (°C)')
        ax6.set_title('RMSE vs MAE Correlation')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Performance Dashboard - {roi}', fontsize=16)
        plt.savefig(self.viz_dir / f'performance_dashboard_{roi}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_analysis(self, gt_data, recon_data, date_str, roi):
        """Create correlation analysis plots."""
        valid_mask = ~(np.isnan(gt_data) | np.isnan(recon_data))
        gt_valid = gt_data[valid_mask]
        recon_valid = recon_data[valid_mask]
        
        # Calculate correlation
        correlation = np.corrcoef(gt_valid.flatten(), recon_valid.flatten())[0, 1]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Correlation Analysis - {date_str} ({roi})', fontsize=16)
        
        # 1. Scatter plot with regression
        axes[0].scatter(gt_valid, recon_valid, alpha=0.5, s=1)
        
        # Add regression line
        z = np.polyfit(gt_valid, recon_valid, 1)
        p = np.poly1d(z)
        axes[0].plot(gt_valid, p(gt_valid), "r--", alpha=0.8, linewidth=2)
        
        # Add 1:1 line
        min_val = min(gt_valid.min(), recon_valid.min())
        max_val = max(gt_valid.max(), recon_valid.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        axes[0].set_xlabel('Ground Truth (°C)')
        axes[0].set_ylabel('Reconstructed (°C)')
        axes[0].set_title(f'Correlation: {correlation:.3f}')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Correlation map (spatial correlation)
        # Create correlation map by calculating correlation in sliding windows
        window_size = 10
        height, width = gt_data.shape
        corr_map = np.full((height, width), np.nan)
        
        for i in range(0, height - window_size, window_size // 2):
            for j in range(0, width - window_size, window_size // 2):
                gt_window = gt_data[i:i+window_size, j:j+window_size]
                recon_window = recon_data[i:i+window_size, j:j+window_size]
                valid_window = ~(np.isnan(gt_window) | np.isnan(recon_window))
                
                if np.sum(valid_window) > window_size * window_size // 4:
                    try:
                        corr = np.corrcoef(gt_window[valid_window], 
                                         recon_window[valid_window])[0, 1]
                        corr_map[i:i+window_size, j:j+window_size] = corr
                    except:
                        pass
        
        im = axes[1].imshow(corr_map, cmap='RdYlBu_r', vmin=-1, vmax=1)
        axes[1].set_title('Spatial Correlation Map')
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        plt.colorbar(im, ax=axes[1], label='Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'correlation_analysis_{date_str}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()


def integrate_with_main_analyzer(analyzer_instance):
    """Integrate advanced visualizations with the main analyzer."""
    # This function would be called from the main analyzer
    # to add advanced visualizations to the existing pipeline
    
    advanced_viz = AdvancedVisualizations(analyzer_instance.output_dir)
    
    # Example integration points:
    # 1. After loading each image pair
    # 2. After calculating metrics
    # 3. At the end of the analysis
    
    return advanced_viz


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    output_dir = Path("results/example")
    advanced_viz = AdvancedVisualizations(output_dir)
    
    # Example data (replace with actual data)
    gt_data = np.random.rand(100, 100) * 30 + 10  # 10-40°C
    recon_data = gt_data + np.random.normal(0, 2, (100, 100))  # Add noise
    
    # Create example visualizations
    advanced_viz.create_error_distribution_plots(gt_data, recon_data, "2020-01-01", "vietnam")
    advanced_viz.create_error_maps(gt_data, recon_data, "2020-01-01", "vietnam")
    
    print("Advanced visualizations created successfully!") 