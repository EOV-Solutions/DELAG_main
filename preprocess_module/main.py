import os
import argparse
import glob
import shutil
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.windows import Window
from tqdm import tqdm


def process_s2_nodata_pixels(file_path: str) -> None:
    """
    Opens a multi-band Sentinel-2 GeoTIFF and converts pixels to NaN where
    all 4 bands have the value -100 (nodata value).
    
    Args:
        file_path: The full path to the Sentinel-2 GeoTIFF file.
    """
    try:
        with rasterio.open(file_path, 'r+') as src:
            # Ensure the file is writable
            if src.mode != 'r+':
                print(f"Warning: Cannot write to {file_path}. It is not in update mode.")
                return

            # Check if file has at least 4 bands (expected for S2: B2, B3, B4, B8)
            if src.count < 4:
                print(f"Warning: {os.path.basename(file_path)} has {src.count} bands, expected at least 4. Skipping.")
                return

            # Read all bands
            band_data = src.read().astype('float32')  # Shape: (bands, height, width)
            
            # Find pixels where ALL bands being considered have value -100
            # Let's consider the first 4 bands for the nodata check
            nodata_mask = np.all(band_data[:4, :, :] == -100, axis=0)  # Shape: (height, width)
            num_nodata_pixels = np.sum(nodata_mask)
            
            if num_nodata_pixels > 0:
                # Set all bands to NaN for these pixels
                band_data[:, nodata_mask] = np.nan
                
                # Write back all bands
                for band_idx in range(src.count):
                    src.write(band_data[band_idx], band_idx + 1)
                
                # Update nodata value to NaN
                src.nodata = np.nan
                
                print(f"  > {os.path.basename(file_path)}: Converted {num_nodata_pixels} pixels (first 4 bands = -100) to NaN")
            else:
                print(f"  Skipping {os.path.basename(file_path)}: No pixels found where first 4 bands = -100")

    except RasterioIOError as e:
        print(f"Error opening or processing {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {file_path}: {e}")


def process_s2_folder(s2_folder: str) -> None:
    """
    Process all Sentinel-2 images in a folder to convert nodata pixels.
    
    Args:
        s2_folder: Path to the folder containing Sentinel-2 .tif files
    """
    print(f"Processing S2 images in: {s2_folder}")
    
    # Find all .tif files in the folder
    tif_files = glob.glob(os.path.join(s2_folder, '*.tif'))
    
    if not tif_files:
        print("  No .tif files found in the S2 folder.")
        return
    
    print(f"  Found {len(tif_files)} .tif files to process.")
    
    # Process each file with progress bar
    for file_path in tqdm(tif_files, desc="Processing S2 files"):
        process_s2_nodata_pixels(file_path)


def remove_satellite_prefixes(file_path: str) -> str:
    """
    Renames a file by removing "L8_" or "L9_" prefixes from the filename.
    
    Args:
        file_path: The full path to the file to be renamed.
        
    Returns:
        str: The new file path after renaming, or the original path if no prefix was found.
    """
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Check if filename starts with L8_ or L9_
    if filename.startswith('L8_'):
        new_filename = filename[3:]  # Remove "L8_"
        new_path = os.path.join(directory, new_filename)
        try:
            os.rename(file_path, new_path)
            print(f"  Renamed: {filename} -> {new_filename}")
            return new_path
        except OSError as e:
            print(f"  Error renaming {filename}: {e}")
            return file_path
    elif filename.startswith('L9_'):
        new_filename = filename[3:]  # Remove "L9_"
        new_path = os.path.join(directory, new_filename)
        try:
            os.rename(file_path, new_path)
            print(f"  Renamed: {filename} -> {new_filename}")
            return new_path
        except OSError as e:
            print(f"  Error renaming {filename}: {e}")
            return file_path
    else:
        # No prefix found, return original path
        return file_path

def rename_files_remove_satellite_prefixes(folder_path: str) -> None:
    """
    Finds all files in a folder and removes "L8_" or "L9_" prefixes from their names.
    
    Args:
        folder_path: The path to the folder containing files to be renamed.
    """
    print(f"Removing satellite prefixes (L8_, L9_) from files in: {folder_path}")
    
    # Get all files (not just .tif) in case there are other file types
    all_files = glob.glob(os.path.join(folder_path, '*'))
    # Filter to only include files (not directories)
    files = [f for f in all_files if os.path.isfile(f)]
    
    if not files:
        print("  No files found in the specified folder.")
        return
    
    # Filter files that have L8_ or L9_ prefixes
    files_to_rename = [f for f in files if 
                      os.path.basename(f).startswith('L8_') or 
                      os.path.basename(f).startswith('L9_')]
    
    if not files_to_rename:
        print("  No files with L8_ or L9_ prefixes found.")
        return
        
    print(f"  Found {len(files_to_rename)} files with satellite prefixes to rename.")
    
    for file_path in files_to_rename:
        remove_satellite_prefixes(file_path)
    
    print(f"  Completed renaming files in {folder_path}\n")

def filter_image_by_range(file_path: str, lower_bound: float, upper_bound: float) -> None:
    """
    Opens a GeoTIFF, filters its first band based on a valid range,
    and overwrites the file with the filtered data.

    Pixels with values outside the [lower_bound, upper_bound] range are
    set to np.nan.

    Args:
        file_path: The full path to the GeoTIFF file.
        lower_bound: The minimum valid value for a pixel.
        upper_bound: The maximum valid value for a pixel.
    """
    try:
        with rasterio.open(file_path, 'r+') as src:
            # Ensure the file is writable
            if src.mode != 'r+':
                print(f"Warning: Cannot write to {file_path}. It is not in update mode.")
                return

            # Preserve original nodata value if it exists (assumed common for all bands)
            nodata_val = src.nodata

            total_changed = 0  # Track total modified pixels across bands

            for band_idx in range(1, src.count + 1):
                band_data = src.read(band_idx).astype('float32')

                # Mask of values outside allowed range
                mask = (band_data < lower_bound) | (band_data > upper_bound)
                pixels_to_change = np.sum(mask)

                if pixels_to_change > 0:
                    band_data[mask] = np.nan
                    src.write(band_data, band_idx)
                    total_changed += pixels_to_change

            if total_changed > 0:
                # Ensure nodata is set to NaN
                if not (nodata_val is not None and np.isnan(nodata_val)):
                    src.nodata = np.nan
                print(f"  > {os.path.basename(file_path)}: Replaced {total_changed} pixels outside range [{lower_bound}, {upper_bound}] across {src.count} band(s).")
            else:
                print(f"  Skipping {os.path.basename(file_path)}: All pixel values are within the valid range across all bands.")

    except RasterioIOError as e:
        print(f"Error opening or processing {file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with {file_path}: {e}")


def filter_lst_folder(lst_folder: str) -> None:
    """
    Main function to find all .tif files in a folder and apply a value filter.
    """
    print(f"Starting LST value filtering in: {lst_folder}")
    print("  Filtering pixel values to be within the range [260, 340].")
    
    tif_files = glob.glob(os.path.join(lst_folder, '*.tif'))
    
    if not tif_files:
        print("  No .tif files found in the specified folder.")
        return

    print(f"  Found {len(tif_files)} .tif files to process.")
    
    for file_path in tqdm(tif_files, desc="Filtering LST files"):
        filter_image_by_range(file_path, lower_bound=260, upper_bound=340)
        
    print(f"  LST filtering complete for {lst_folder}.\n")


def main_recursive_rename(base_folder: str) -> None:
    """
    Main function to recursively find all files and remove satellite prefixes from their names.
    This function only handles renaming, not LST filtering.
    
    Args:
        base_folder: The base folder to process recursively.
    """
    print(f"Starting satellite prefix removal in: {base_folder}")
    print("Removing 'L8_' and 'L9_' prefixes from all file names.")
    
    if not os.path.exists(base_folder):
        print(f"Error: Folder {base_folder} does not exist.")
        return
    
    total_renamed = 0
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_folder):
        if files:  # Only process if there are files in this directory
            print(f"\nChecking directory: {root}")
            
            # Filter files that have L8_ or L9_ prefixes
            files_to_rename = [f for f in files if f.startswith('L8_') or f.startswith('L9_')]
            
            if files_to_rename:
                print(f"  Found {len(files_to_rename)} files to rename in this directory.")
                for filename in files_to_rename:
                    file_path = os.path.join(root, filename)
                    new_path = remove_satellite_prefixes(file_path)
                    if new_path != file_path:
                        total_renamed += 1
            else:
                print(f"  No files with satellite prefixes found in this directory.")
    
    print(f"\nRenaming complete. Total files renamed: {total_renamed}")


def process_roi_directory(roi_path: str, actions: list):
    """
    Processes a single ROI directory, applying actions to its subfolders (e.g., lst, s2).

    Args:
        roi_path (str): The full path to the ROI directory to process.
        actions (list): A list of actions to perform (e.g., ['rename', 'filter_lst']).
    """
    if not os.path.isdir(roi_path):
        print(f"Skipping non-directory: {roi_path}")
        return

    print(f"\n{'='*50}")
    print(f"Processing ROI directory: {os.path.basename(roi_path)}")
    print(f"{'='*50}")

    # Action 1: Recursive rename across the entire ROI if requested
    if 'rename' in actions:
         main_recursive_rename(roi_path)

    # Action 2 & 3: Folder-specific actions on subdirectories
    for folder_name in os.listdir(roi_path):
        folder_path = os.path.join(roi_path, folder_name)
        if os.path.isdir(folder_path):
            
            # Apply LST filtering if this is an LST folder and action is requested
            if 'filter_lst' in actions and "lst" in folder_name.lower():
                filter_lst_folder(folder_path)
            
            # Process S2 images if this is an S2 folder and action is requested
            if 'process_s2' in actions and "s2" in folder_name.lower():
                process_s2_folder(folder_path)


def split_geotiff(source_path: str, dest_path_left: str, dest_path_right: str):
    """
    Splits a GeoTIFF file spatially into two halves (left and right) by width.
    Args:
        source_path: The full path to the source GeoTIFF.
        dest_path_left: The full path to save the left half.
        dest_path_right: The full path to save the right half.
    """
    try:
        with rasterio.open(source_path) as src:
            mid_width = src.width // 2
            left_window = Window(col_off=0, row_off=0, width=mid_width, height=src.height)
            right_window = Window(col_off=mid_width, row_off=0, width=src.width - mid_width, height=src.height)
            left_transform = src.window_transform(left_window)
            right_transform = src.window_transform(right_window)
            profile_left = src.profile.copy()
            profile_left.update({'width': left_window.width, 'height': left_window.height, 'transform': left_transform})
            profile_right = src.profile.copy()
            profile_right.update({'width': right_window.width, 'height': right_window.height, 'transform': right_transform})
            with rasterio.open(dest_path_left, 'w', **profile_left) as dst_left:
                dst_left.write(src.read(window=left_window))
            with rasterio.open(dest_path_right, 'w', **profile_right) as dst_right:
                dst_right.write(src.read(window=right_window))
    except Exception as e:
        print(f"  Error splitting file {os.path.basename(source_path)}: {e}")

def process_roi_for_splitting(roi_path: str, output_base: str):
    """
    Orchestrates the splitting of all images within a single ROI directory.
    Creates new subdirectories and calls the GeoTIFF splitting function.
    Args:
        roi_path (str): The full path to the source ROI directory.
        output_base (str): The base directory to save the new split ROI folders.
    """
    roi_name = os.path.basename(os.path.normpath(roi_path))
    print(f"--- Splitting ROI: {roi_name} ---")
    dest_roi_1 = os.path.join(output_base, f"{roi_name}_1")
    dest_roi_2 = os.path.join(output_base, f"{roi_name}_2")
    for subfolder_name in os.listdir(roi_path):
        source_subfolder = os.path.join(roi_path, subfolder_name)
        if os.path.isdir(source_subfolder):
            print(f"  Processing subfolder: {subfolder_name}")
            dest_subfolder_1 = os.path.join(dest_roi_1, subfolder_name)
            dest_subfolder_2 = os.path.join(dest_roi_2, subfolder_name)
            os.makedirs(dest_subfolder_1, exist_ok=True)
            os.makedirs(dest_subfolder_2, exist_ok=True)
            tif_files = glob.glob(os.path.join(source_subfolder, '*.tif'))
            if not tif_files:
                print(f"    No .tif files found in {subfolder_name}, skipping.")
                continue
            for source_tif in tqdm(tif_files, desc=f"Splitting {subfolder_name} files", leave=False):
                base_filename = os.path.basename(source_tif)
                dest_tif_left = os.path.join(dest_subfolder_1, base_filename)
                dest_tif_right = os.path.join(dest_subfolder_2, base_filename)
                split_geotiff(source_tif, dest_tif_left, dest_tif_right)


def main():
    parser = argparse.ArgumentParser(
        description="Perform processing tasks on LST and S2 data, and rename files. "
                    "Processes data out-of-place by copying to an output folder first, "
                    "or in-place if no output folder is specified."
    )
    # parser.add_argument(
    #     '--actions',
    #     nargs='+',
    #     choices=['rename', 'filter_lst', 'process_s2'],
    #     required=True,
    #     help="A list of actions to perform. "
    #             " 'rename': remove L8/L9 prefixes. "
    #             " 'filter_lst': filter LST values to [260, 340]. "
    #             " 'process_s2': convert S2 nodata pixels to NaN."
    # )
    parser.add_argument(
        '--roi_name',
        help="Specific source folder to process. If provided, actions are applied to this folder only. "
             "If not provided, script runs on the full source_base data structure."
    )
    parser.add_argument(
        '--source_base',
        default='/mnt/hdd12tb/code/nhatvm/DELAG_main/data/retrieved_data',
        help="Base folder containing source ROI directories (default: %(default)s)"
    )
    parser.add_argument(
        '--output_base',
        default='/mnt/hdd12tb/code/nhatvm/DELAG_main/data/preprocessed_data',
        help="Base folder to store processed data. If provided, data is copied here before processing. "
             "If not provided, processing is done in the default path %(default)s."
    )
    parser.add_argument(
        '--split_rois',
        action='store_true',
        help="As a final step, split the processed ROIs into halves. This modifies the results folder in-place."
    )
    
    args = parser.parse_args()
    
    # Determine the primary source path
    source_path = os.path.join(args.source_base, args.roi_name) if args.roi_name else args.source_base
    if not os.path.exists(source_path):
        print(f"Error: Source path '{source_path}' does not exist.")
        return

    # Determine the target path for processing
    if args.output_base:
        # Non-destructive mode: copy first, then process the copy.
        dest_path = args.output_base
        
        if args.roi_name:
            # For a single folder, copy it into the output base.
            processing_path = os.path.join(dest_path, os.path.basename(os.path.normpath(source_path)))
            print(f"Output mode: Copying single folder from '{source_path}' to '{processing_path}' before processing.")
            
            if os.path.isdir(source_path):
                shutil.copytree(source_path, processing_path, dirs_exist_ok=True)
            else: # It's a single file
                os.makedirs(os.path.dirname(processing_path), exist_ok=True)
                shutil.copy2(source_path, processing_path)
        else:
            # Full mode: Copy each ROI folder from the source_base into the output_base.
            processing_path = dest_path
            print(f"Output mode: Copying all ROIs from '{source_path}' to '{processing_path}' before processing.")

            for roi_name in os.listdir(source_path):
                source_roi_path = os.path.join(source_path, roi_name)
                dest_roi_path = os.path.join(processing_path, roi_name)
                
                if os.path.isdir(source_roi_path):
                    print(f"  Copying ROI: {roi_name}...")
                    shutil.copytree(source_roi_path, dest_roi_path, dirs_exist_ok=True)
                else:
                    print(f"  Skipping non-directory item: {roi_name}")
            
        print("Copy complete. Processing will be performed on the new data.")
    else:
        # Destructive mode: process in-place.
        print("\nWarning: No --output_base provided. Processing will be done IN-PLACE on the source data.\n")
        processing_path = source_path
    
    # --- Step 2: Perform Pre-processing Actions ---
    actions = ['rename', 'filter_lst', 'process_s2']
    
    if args.roi_name:
        # Process the single folder specified
        print(f"\n--- Processing single ROI folder: {processing_path} ---")
        process_roi_directory(processing_path, actions)
    else:
        # Process all ROIs in the base directory
        print("=" * 80)
        print(f"STARTING PRE-PROCESSING from: {processing_path}")
        print(f"ACTIONS: {', '.join(actions).upper()}")
        print("=" * 80)

        if not os.path.exists(processing_path):
            print(f"Error: Processing path '{processing_path}' does not exist.")
            return

        for roi_name in os.listdir(processing_path):
            roi_path = os.path.join(processing_path, roi_name)
            process_roi_directory(roi_path, actions)

    print("\n" + "=" * 80)
    print("--- Pre-processing stage complete ---")
    print("=" * 80)
    
    # --- Step 3: Optional Final ROI Splitting ---
    if args.split_rois:
        print("\n" + "=" * 80)
        print("--- Starting Final ROI Splitting Operation ---")
        
        # The source for splitting is the path where the preprocessing just occurred.
        # The output will also be placed in this same directory, and the original will be removed.
        processing_path = args.output_base if args.output_base else args.source_base
        
        print(f"Splitting ROIs in-place within: {processing_path}")
        print("=" * 80 + "\n")
        
        # Get a list of ROI directories before iterating, as we will be modifying the directory contents
        rois_to_process = [args.roi_name] if args.roi_name else [d for d in os.listdir(processing_path) if os.path.isdir(os.path.join(processing_path, d))]

        for roi_name in rois_to_process:
            roi_path = os.path.join(processing_path, roi_name)
            
            # The output base is the directory containing the current ROI folder
            process_roi_for_splitting(roi_path, processing_path)
            
            # After successful splitting, remove the original ROI directory
            try:
                print(f"  Removing original ROI directory: {roi_path}")
                shutil.rmtree(roi_path)
                print(f"  Successfully removed {roi_name}.")
            except OSError as e:
                print(f"  Error removing original ROI directory {roi_name}: {e}")
        
        print("\n" + "=" * 80)
        print("--- ROI Splitting Operation Complete ---")
        print("=" * 80)

    print("\nALL WORKFLOWS COMPLETE.")

if __name__ == '__main__':
    main()