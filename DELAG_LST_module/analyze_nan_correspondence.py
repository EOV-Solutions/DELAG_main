"""
Analysis script to verify if NaN pixels in inferred images correspond 
to NaN pixels in the original label LST images.
"""
import numpy as np
import os
import glob
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple, List
import json

# Import project modules
import config
import utils

def load_lst_stack_from_directory(directory: str, file_pattern: str = "*.tif") -> Tuple[np.ndarray, List[str], List[datetime]]:
    """
    Load LST images from a directory and return as a stack.
    
    Args:
        directory: Path to directory containing LST files
        file_pattern: Pattern to match files
        
    Returns:
        stack: 3D numpy array (n_dates, height, width)
        file_paths: List of file paths
        dates: List of datetime objects
    """
    print(f"Loading LST stack from: {directory}")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    # Find all matching files
    all_files = glob.glob(os.path.join(directory, file_pattern))
    all_files.sort()
    
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern {file_pattern} in {directory}")
    
    # Parse dates from filenames
    dates = []
    valid_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            # Try different date parsing patterns
            if '_' in filename:
                date_str = filename.split('_')[-1].replace('.tif', '')
                if len(date_str) == 8:  # YYYYMMDD format
                    date = datetime.strptime(date_str, '%Y%m%d')
                elif len(date_str) == 10:  # YYYY-MM-DD format
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                else:
                    continue
            else:
                continue
                
            dates.append(date)
            valid_files.append(file_path)
        except (ValueError, IndexError):
            print(f"Warning: Could not parse date from filename: {filename}")
            continue
    
    if not valid_files:
        raise ValueError("No files with parseable dates found")
    
    # Sort by date
    sorted_pairs = sorted(zip(dates, valid_files))
    dates, valid_files = zip(*sorted_pairs)
    
    # Load the first file to get dimensions
    with rasterio.open(valid_files[0]) as src:
        height, width = src.height, src.width
        
    # Initialize stack
    stack = np.full((len(valid_files), height, width), np.nan, dtype=np.float32)
    
    # Load all files
    for i, file_path in enumerate(valid_files):
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1).astype(np.float32)
                # Convert any nodata values to NaN
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                stack[i, :, :] = data
        except Exception as e:
            print(f"Warning: Could not read file {file_path}. Error: {e}")
            
    return stack, list(valid_files), list(dates)

def analyze_nan_correspondence(label_stack: np.ndarray, infer_stack: np.ndarray, 
                             label_dates: List[datetime], infer_dates: List[datetime]) -> Dict:
    """
    Analyze the correspondence between NaN pixels in label and inferred stacks.
    
    Args:
        label_stack: Ground truth LST stack (n_dates, height, width)
        infer_stack: Inferred LST stack (n_dates, height, width)
        label_dates: Dates for label stack
        infer_dates: Dates for infer stack
        
    Returns:
        Dictionary containing analysis results
    """
    results = {}
    
    # Find common dates
    label_date_set = set(label_dates)
    infer_date_set = set(infer_dates)
    common_dates = sorted(label_date_set.intersection(infer_date_set))
    
    print(f"Found {len(common_dates)} common dates between label and infer data")
    
    if not common_dates:
        print("No common dates found!")
        return {"error": "No common dates found"}
    
    # Get indices for common dates
    label_indices = [label_dates.index(date) for date in common_dates]
    infer_indices = [infer_dates.index(date) for date in common_dates]
    
    # Extract common date stacks
    label_common = label_stack[label_indices, :, :]
    infer_common = infer_stack[infer_indices, :, :]
    
    print(f"Label stack shape: {label_common.shape}")
    print(f"Infer stack shape: {infer_common.shape}")
    
    # Analyze NaN patterns for each date
    date_analysis = []
    
    for i, date in enumerate(common_dates):
        label_img = label_common[i, :, :]
        infer_img = infer_common[i, :, :]
        
        # Get NaN masks
        label_nan_mask = np.isnan(label_img)
        infer_nan_mask = np.isnan(infer_img)
        
        # Calculate statistics
        total_pixels = label_img.size
        label_nan_count = np.sum(label_nan_mask)
        infer_nan_count = np.sum(infer_nan_mask)
        
        # Calculate correspondence
        both_nan = np.sum(label_nan_mask & infer_nan_mask)
        only_label_nan = np.sum(label_nan_mask & ~infer_nan_mask)
        only_infer_nan = np.sum(~label_nan_mask & infer_nan_mask)
        both_valid = np.sum(~label_nan_mask & ~infer_nan_mask)
        
        # Calculate correspondence metrics
        if label_nan_count > 0:
            correspondence_rate = both_nan / label_nan_count
        else:
            correspondence_rate = 1.0 if infer_nan_count == 0 else 0.0
            
        precision = both_nan / infer_nan_count if infer_nan_count > 0 else 1.0
        recall = both_nan / label_nan_count if label_nan_count > 0 else 1.0
        
        date_result = {
            'date': date.strftime('%Y-%m-%d'),
            'total_pixels': total_pixels,
            'label_nan_count': int(label_nan_count),
            'infer_nan_count': int(infer_nan_count),
            'both_nan': int(both_nan),
            'only_label_nan': int(only_label_nan),
            'only_infer_nan': int(only_infer_nan),
            'both_valid': int(both_valid),
            'label_nan_percentage': (label_nan_count / total_pixels) * 100,
            'infer_nan_percentage': (infer_nan_count / total_pixels) * 100,
            'correspondence_rate': correspondence_rate,
            'precision': precision,
            'recall': recall
        }
        
        date_analysis.append(date_result)
        
        print(f"Date {date.strftime('%Y-%m-%d')}:")
        print(f"  Label NaN: {label_nan_count}/{total_pixels} ({date_result['label_nan_percentage']:.2f}%)")
        print(f"  Infer NaN: {infer_nan_count}/{total_pixels} ({date_result['infer_nan_percentage']:.2f}%)")
        print(f"  Both NaN: {both_nan} (Correspondence: {correspondence_rate:.4f})")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Overall statistics
    total_label_nan = sum(d['label_nan_count'] for d in date_analysis)
    total_infer_nan = sum(d['infer_nan_count'] for d in date_analysis)
    total_both_nan = sum(d['both_nan'] for d in date_analysis)
    total_pixels_all = sum(d['total_pixels'] for d in date_analysis)
    
    overall_correspondence = total_both_nan / total_label_nan if total_label_nan > 0 else 1.0
    overall_precision = total_both_nan / total_infer_nan if total_infer_nan > 0 else 1.0
    overall_recall = total_both_nan / total_label_nan if total_label_nan > 0 else 1.0
    
    results = {
        'common_dates_count': len(common_dates),
        'common_dates': [d.strftime('%Y-%m-%d') for d in common_dates],
        'date_analysis': date_analysis,
        'overall_stats': {
            'total_pixels': total_pixels_all,
            'total_label_nan': total_label_nan,
            'total_infer_nan': total_infer_nan,
            'total_both_nan': total_both_nan,
            'overall_correspondence_rate': overall_correspondence,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'label_nan_percentage': (total_label_nan / total_pixels_all) * 100,
            'infer_nan_percentage': (total_infer_nan / total_pixels_all) * 100
        }
    }
    
    return results

def create_visualization(results: Dict, output_dir: str, roi_name: str):
    """Create visualizations of the NaN correspondence analysis."""
    
    if 'error' in results:
        print("Cannot create visualization due to error in analysis")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results['date_analysis'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'NaN Correspondence Analysis - ROI: {roi_name}', fontsize=16)
    
    # Panel 1: NaN percentages over time
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['label_nan_percentage'], 'o-', label='Label NaN %', color='blue')
    ax1.plot(df['date'], df['infer_nan_percentage'], 's-', label='Infer NaN %', color='red')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('NaN Percentage (%)')
    ax1.set_title('NaN Percentages Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Panel 2: Correspondence metrics
    ax2 = axes[0, 1]
    ax2.plot(df['date'], df['correspondence_rate'], 'o-', label='Correspondence Rate', color='green')
    ax2.plot(df['date'], df['precision'], 's-', label='Precision', color='orange')
    ax2.plot(df['date'], df['recall'], '^-', label='Recall', color='purple')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Metric Value')
    ax2.set_title('Correspondence Metrics Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Panel 3: Pixel count comparison
    ax3 = axes[1, 0]
    width = 0.35
    x = range(len(df))
    ax3.bar([i - width/2 for i in x], df['label_nan_count'], width, label='Label NaN', alpha=0.7, color='blue')
    ax3.bar([i + width/2 for i in x], df['infer_nan_count'], width, label='Infer NaN', alpha=0.7, color='red')
    ax3.set_xlabel('Date Index')
    ax3.set_ylabel('NaN Pixel Count')
    ax3.set_title('NaN Pixel Counts Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Correspondence statistics summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    overall = results['overall_stats']
    
    summary_text = f"""Overall Statistics:
    
Total Pixels: {overall['total_pixels']:,}
Total Label NaN: {overall['total_label_nan']:,} ({overall['label_nan_percentage']:.2f}%)
Total Infer NaN: {overall['total_infer_nan']:,} ({overall['infer_nan_percentage']:.2f}%)
Total Both NaN: {overall['total_both_nan']:,}

Correspondence Rate: {overall['overall_correspondence_rate']:.4f}
Precision: {overall['overall_precision']:.4f}
Recall: {overall['overall_recall']:.4f}

Dates Analyzed: {results['common_dates_count']}
"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'nan_correspondence_analysis_{roi_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()

def main():
    """Main function to run the NaN correspondence analysis."""
    
    # Set the ROI name
    roi_name = "C-48-56-A"
    
    # Define paths
    base_output_dir = "/mnt/hdd12tb/code/nhatvm/DELAG_main/data/output"  # Adjust this path as needed
    roi_output_dir = os.path.join(base_output_dir, roi_name)
    
    print(f"Analyzing NaN correspondence for ROI: {roi_name}")
    print(f"Output directory: {roi_output_dir}")
    
    try:
        # Path to original LST data (labels)
        label_lst_dir = os.path.join(roi_output_dir, "data_train")  # or wherever your original LST data is stored
        
        # Path to inferred/reconstructed LST data  
        infer_lst_dir = os.path.join(roi_output_dir, "reconstructed_lst_train")
        
        print(f"Label LST directory: {label_lst_dir}")
        print(f"Infer LST directory: {infer_lst_dir}")
        
        # Load label LST stack
        print("\nLoading label LST data...")
        # Try to load from processed data first
        try:
            train_data = utils.load_processed_data(label_lst_dir)
            label_stack = train_data['lst_stack']
            label_dates = train_data['common_dates']
            print(f"Loaded label data from processed files: {label_stack.shape}")
        except:
            # Fallback to loading from TIF files
            print("Could not load from processed data, trying TIF files...")
            label_stack, _, label_dates = load_lst_stack_from_directory(
                label_lst_dir, "LST_*.tif"
            )
        
        # Load inferred LST stack
        print("\nLoading inferred LST data...")
        infer_stack, _, infer_dates = load_lst_stack_from_directory(
            infer_lst_dir, "LST_RECON_TRAIN_*.tif"
        )
        
        print(f"Label stack shape: {label_stack.shape}")
        print(f"Infer stack shape: {infer_stack.shape}")
        print(f"Label dates: {len(label_dates)}")
        print(f"Infer dates: {len(infer_dates)}")
        
        # Run the analysis
        print("\nRunning NaN correspondence analysis...")
        results = analyze_nan_correspondence(label_stack, infer_stack, label_dates, infer_dates)
        
        # Save results to JSON
        output_file = os.path.join(roi_output_dir, f'nan_correspondence_analysis_{roi_name}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Analysis results saved to: {output_file}")
        
        # Create visualization
        print("\nCreating visualization...")
        create_visualization(results, roi_output_dir, roi_name)
        
        # Print summary
        if 'overall_stats' in results:
            overall = results['overall_stats']
            print(f"\n=== SUMMARY ===")
            print(f"Overall Correspondence Rate: {overall['overall_correspondence_rate']:.4f}")
            print(f"Overall Precision: {overall['overall_precision']:.4f}")
            print(f"Overall Recall: {overall['overall_recall']:.4f}")
            
            if overall['overall_correspondence_rate'] > 0.95:
                print("✓ EXCELLENT: Inferred NaN positions highly correspond to label NaN positions")
            elif overall['overall_correspondence_rate'] > 0.8:
                print("✓ GOOD: Inferred NaN positions reasonably correspond to label NaN positions")
            else:
                print("⚠ WARNING: Low correspondence between inferred and label NaN positions")
                
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 