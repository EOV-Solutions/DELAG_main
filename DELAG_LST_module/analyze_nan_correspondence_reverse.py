"""
Analysis script to prove that wherever inferred images have NaN pixels,
the corresponding positions in label images are also NaN.

This addresses the specific hypothesis:
"If there is a NaN pixel at position (i,j) in infer images, 
then all label LST images have NaN values at that same position (i,j)"
"""
import numpy as np
import os
import glob
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import Dict, Tuple, List
import json

# Import project modules
import config
import utils

def load_lst_stack_from_directory(directory: str, file_pattern: str = "*.tif") -> Tuple[np.ndarray, List[str], List[datetime]]:
    """Load LST images from a directory and return as a stack."""
    print(f"Loading LST stack from: {directory}")
    
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    all_files = glob.glob(os.path.join(directory, file_pattern))
    all_files.sort()
    
    if not all_files:
        raise FileNotFoundError(f"No files found matching pattern {file_pattern} in {directory}")
    
    dates = []
    valid_files = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
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

def analyze_reverse_nan_correspondence(label_stack: np.ndarray, infer_stack: np.ndarray, 
                                     label_dates: List[datetime], infer_dates: List[datetime]) -> Dict:
    """
    Analyze if wherever inferred images have NaN, the corresponding label positions are also NaN.
    
    This proves the hypothesis: "If infer[i,j] is NaN, then label[i,j] is NaN for all corresponding dates"
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
    
    # For each date, check if inferred NaN positions correspond to label NaN positions
    date_analysis = []
    
    for i, date in enumerate(common_dates):
        label_img = label_common[i, :, :]
        infer_img = infer_common[i, :, :]
        
        # Get NaN masks
        label_nan_mask = np.isnan(label_img)
        infer_nan_mask = np.isnan(infer_img)
        
        # Count pixels
        total_pixels = infer_img.size
        infer_nan_count = np.sum(infer_nan_mask)
        
        # Key analysis: For pixels that are NaN in inferred image, 
        # how many are also NaN in the label image?
        infer_nan_positions_also_label_nan = np.sum(infer_nan_mask & label_nan_mask)
        infer_nan_positions_but_label_valid = np.sum(infer_nan_mask & ~label_nan_mask)
        
        # Calculate the key metric: What percentage of inferred NaN positions 
        # are also NaN in the label?
        if infer_nan_count > 0:
            hypothesis_support_rate = infer_nan_positions_also_label_nan / infer_nan_count
        else:
            hypothesis_support_rate = 1.0  # No inferred NaN to contradict hypothesis
        
        date_result = {
            'date': date.strftime('%Y-%m-%d'),
            'total_pixels': total_pixels,
            'infer_nan_count': int(infer_nan_count),
            'infer_nan_positions_also_label_nan': int(infer_nan_positions_also_label_nan),
            'infer_nan_positions_but_label_valid': int(infer_nan_positions_but_label_valid),
            'infer_nan_percentage': (infer_nan_count / total_pixels) * 100,
            'hypothesis_support_rate': hypothesis_support_rate,
            'hypothesis_violations': int(infer_nan_positions_but_label_valid)
        }
        
        date_analysis.append(date_result)
        
        print(f"Date {date.strftime('%Y-%m-%d')}:")
        print(f"  Infer NaN pixels: {infer_nan_count}")
        print(f"  Infer NaN → Label also NaN: {infer_nan_positions_also_label_nan}")
        print(f"  Infer NaN → Label valid (violations): {infer_nan_positions_but_label_valid}")
        print(f"  Hypothesis support rate: {hypothesis_support_rate:.4f} ({hypothesis_support_rate*100:.2f}%)")
        
        if infer_nan_positions_but_label_valid > 0:
            print(f"  ⚠ WARNING: Found {infer_nan_positions_but_label_valid} violations of the hypothesis!")
        else:
            print(f"  ✓ No violations found for this date")
    
    # Overall statistics
    total_infer_nan = sum(d['infer_nan_count'] for d in date_analysis)
    total_infer_nan_also_label_nan = sum(d['infer_nan_positions_also_label_nan'] for d in date_analysis)
    total_violations = sum(d['hypothesis_violations'] for d in date_analysis)
    total_pixels_all = sum(d['total_pixels'] for d in date_analysis)
    
    overall_hypothesis_support = total_infer_nan_also_label_nan / total_infer_nan if total_infer_nan > 0 else 1.0
    
    results = {
        'hypothesis': "If infer[i,j] is NaN, then label[i,j] is also NaN",
        'common_dates_count': len(common_dates),
        'common_dates': [d.strftime('%Y-%m-%d') for d in common_dates],
        'date_analysis': date_analysis,
        'overall_stats': {
            'total_pixels': total_pixels_all,
            'total_infer_nan_pixels': total_infer_nan,
            'total_infer_nan_also_label_nan': total_infer_nan_also_label_nan,
            'total_hypothesis_violations': total_violations,
            'overall_hypothesis_support_rate': overall_hypothesis_support,
            'hypothesis_proven': overall_hypothesis_support == 1.0,
            'violation_rate': total_violations / total_infer_nan if total_infer_nan > 0 else 0.0
        }
    }
    
    return results

def create_reverse_visualization(results: Dict, output_dir: str, roi_name: str):
    """Create visualizations for the reverse NaN correspondence analysis."""
    
    if 'error' in results:
        print("Cannot create visualization due to error in analysis")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results['date_analysis'])
    df['date'] = pd.to_datetime(df['date'])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Reverse NaN Correspondence Analysis - ROI: {roi_name}', fontsize=16)
    
    # Panel 1: Hypothesis support rate over time
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['hypothesis_support_rate'], 'o-', color='green', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Perfect Support (100%)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Hypothesis Support Rate')
    ax1.set_title('Hypothesis Support Rate Over Time\n(If infer NaN → label NaN)')
    ax1.set_ylim(0, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Panel 2: Violation counts
    ax2 = axes[0, 1]
    colors = ['red' if v > 0 else 'green' for v in df['hypothesis_violations']]
    ax2.bar(range(len(df)), df['hypothesis_violations'], color=colors, alpha=0.7)
    ax2.set_xlabel('Date Index')
    ax2.set_ylabel('Number of Violations')
    ax2.set_title('Hypothesis Violations per Date')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Infer NaN counts over time
    ax3 = axes[1, 0]
    ax3.plot(df['date'], df['infer_nan_count'], 'o-', color='blue')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Infer NaN Pixel Count')
    ax3.set_title('Inferred NaN Pixels Over Time')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # Panel 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    overall = results['overall_stats']
    
    hypothesis_status = "✓ PROVEN" if overall['hypothesis_proven'] else "✗ DISPROVEN"
    
    summary_text = f"""Hypothesis Analysis Results:

"{results['hypothesis']}"

Status: {hypothesis_status}

Total Infer NaN Pixels: {overall['total_infer_nan_pixels']:,}
Supporting Evidence: {overall['total_infer_nan_also_label_nan']:,}
Violations: {overall['total_hypothesis_violations']:,}

Overall Support Rate: {overall['overall_hypothesis_support_rate']:.4f}
Violation Rate: {overall['violation_rate']:.4f}

Dates Analyzed: {results['common_dates_count']}
"""
    
    color = 'lightgreen' if overall['hypothesis_proven'] else 'lightcoral'
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f'reverse_nan_correspondence_analysis_{roi_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()

def main():
    """Main function to run the reverse NaN correspondence analysis."""
    
    # Set the ROI name
    roi_name = "C-48-56-A"
    
    # Define paths
    base_output_dir = "/mnt/hdd12tb/code/nhatvm/DELAG_main/data/output"
    roi_output_dir = os.path.join(base_output_dir, roi_name)
    
    print(f"Analyzing REVERSE NaN correspondence for ROI: {roi_name}")
    print(f"Hypothesis: If infer[i,j] is NaN, then label[i,j] is also NaN")
    print(f"Output directory: {roi_output_dir}")
    
    try:
        # Path to original LST data (labels)
        label_lst_dir = os.path.join(roi_output_dir, "data_train")
        
        # Path to inferred/reconstructed LST data  
        infer_lst_dir = os.path.join(roi_output_dir, "reconstructed_lst_train")
        
        print(f"Label LST directory: {label_lst_dir}")
        print(f"Infer LST directory: {infer_lst_dir}")
        
        # Load label LST stack
        print("\nLoading label LST data...")
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
        
        # Run the reverse analysis
        print("\nRunning REVERSE NaN correspondence analysis...")
        print("Testing hypothesis: If infer[i,j] is NaN, then label[i,j] is also NaN")
        results = analyze_reverse_nan_correspondence(label_stack, infer_stack, label_dates, infer_dates)
        
        # Save results to JSON
        output_file = os.path.join(roi_output_dir, f'reverse_nan_correspondence_analysis_{roi_name}.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Analysis results saved to: {output_file}")
        
        # Create visualization
        print("\nCreating visualization...")
        create_reverse_visualization(results, roi_output_dir, roi_name)
        
        # Print detailed summary
        if 'overall_stats' in results:
            overall = results['overall_stats']
            print(f"\n" + "="*60)
            print(f"HYPOTHESIS TEST RESULTS")
            print(f"="*60)
            print(f"Hypothesis: {results['hypothesis']}")
            print(f"")
            print(f"Evidence:")
            print(f"  Total inferred NaN pixels: {overall['total_infer_nan_pixels']:,}")
            print(f"  Supporting cases: {overall['total_infer_nan_also_label_nan']:,}")
            print(f"  Violations: {overall['total_hypothesis_violations']:,}")
            print(f"")
            print(f"Results:")
            print(f"  Support Rate: {overall['overall_hypothesis_support_rate']:.6f}")
            print(f"  Violation Rate: {overall['violation_rate']:.6f}")
            print(f"")
            
            if overall['hypothesis_proven']:
                print(f"✓ HYPOTHESIS PROVEN: Every NaN pixel in inferred images")
                print(f"  corresponds to a NaN pixel in the same position in label images.")
            else:
                print(f"✗ HYPOTHESIS DISPROVEN: Found {overall['total_hypothesis_violations']} cases")
                print(f"  where inferred images have NaN but label images have valid values.")
                print(f"  Support rate: {overall['overall_hypothesis_support_rate']*100:.2f}%")
                
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 