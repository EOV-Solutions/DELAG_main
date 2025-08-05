import argparse
import os
import shutil
from datetime import datetime, timedelta

def check_image_timeseries(folder_path, start_date_str, end_date_str, remove_unexpected=False, shift_days=0):
    """
    Checks if a folder contains a time series of images with a 16-day interval.
    Optionally removes images that are not expected in the time series.
    Optionally shifts all remaining dates by a specified number of days.

    Args:
        folder_path (str): The path to the folder containing the images.
        start_date_str (str): The start date of the time series in YYYY-MM-DD format.
        end_date_str (str): The end date of the time series in YYYY-MM-DD format.
        remove_unexpected (bool): If True, removes images that are not expected in the time series.
        shift_days (int): Number of days to add to all dates in the series (can be negative).
    """
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        return

    print(f"Checking for 16-day time series in '{folder_path}'")
    print(f"From {start_date_str} to {end_date_str}")
    if remove_unexpected:
        print("⚠️  REMOVE MODE: Unexpected images will be deleted!")
    if shift_days != 0:
        print(f"📅 SHIFT MODE: All dates will be shifted by {shift_days:+d} days!")
    print()

    # 1. Generate expected dates
    expected_dates = set()
    current_date = start_date
    while current_date <= end_date:
        expected_dates.add(current_date.date())
        current_date += timedelta(days=16)

    # 2. Find existing dates from filenames and track files to remove
    found_dates = set()
    files_to_remove = []
    files_to_keep = []
    file_date_mapping = {}  # Map filename to its date for shifting
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            try:
                # Extract date from filename like '*_YYYY-MM-DD.tif' or '*_YYYYMMDD.tif'
                date_part = filename.rsplit('_', 1)[1].split('.')[0]
                
                # Try different date formats
                found_date = None
                
                # Try YYYY-MM-DD format first
                try:
                    found_date = datetime.strptime(date_part, "%Y-%m-%d").date()
                except ValueError:
                    pass
                
                # Try YYYYMMDD format if the first one failed
                if found_date is None:
                    try:
                        found_date = datetime.strptime(date_part, "%Y%m%d").date()
                    except ValueError:
                        pass
                
                # If we successfully parsed a date and it's in range
                if found_date and start_date.date() <= found_date <= end_date.date():
                    found_dates.add(found_date)
                    if found_date in expected_dates:
                        files_to_keep.append(filename)
                        file_date_mapping[filename] = found_date
                    else:
                        files_to_remove.append(filename)
                        print(f"Marked for removal: {filename} (date {found_date} not in 16-day series)")
                else:
                    # Date is out of range or couldn't be parsed
                    files_to_remove.append(filename)
                    if found_date:
                        print(f"Marked for removal: {filename} (date {found_date} out of range)")
                    else:
                        print(f"Marked for removal: {filename} (could not parse date)")
                    
            except (IndexError, ValueError):
                # Files that don't match the format
                files_to_remove.append(filename)
                print(f"Marked for removal: {filename} (doesn't match expected format)")

    # 3. Compare and report
    all_dates = sorted(list(expected_dates.union(found_dates)))
    missing_dates = sorted(list(expected_dates - found_dates))
    
    print("\n--- Time Series Report ---")
    for date in all_dates:
        status = "FOUND" if date in found_dates else "MISSING"
        is_expected = " (Expected)" if date in expected_dates else " (Not Expected)"
        print(f"{date.strftime('%Y-%m-%d')}: {status}{is_expected}")

    print("\n--- Summary ---")
    if not missing_dates:
        print("✅ All expected images were found.")
    else:
        print(f"❌ Missing {len(missing_dates)} expected images:")
        for date in missing_dates:
            print(f"  - {date.strftime('%Y-%m-%d')}")
    
    print(f"\n📊 File Statistics:")
    print(f"  Expected files: {len(expected_dates)}")
    print(f"  Found files: {len(found_dates)}")
    print(f"  Files to keep: {len(files_to_keep)}")
    print(f"  Files to remove: {len(files_to_remove)}")

    # 4. Remove unexpected files if requested
    if remove_unexpected and files_to_remove:
        print(f"\n🗑️  Removing {len(files_to_remove)} unexpected files...")
        
        # Ask for confirmation if removing many files
        if len(files_to_remove) > 10:
            response = input(f"Are you sure you want to delete {len(files_to_remove)} files? (yes/no): ")
            if response.lower() != 'yes':
                print("Operation cancelled.")
                return
        
        removed_count = 0
        for filename in files_to_remove:
            file_path = os.path.join(folder_path, filename)
            try:
                os.remove(file_path)
                print(f"  ✅ Removed: {filename}")
                removed_count += 1
            except OSError as e:
                print(f"  ❌ Failed to remove {filename}: {e}")
        
        print(f"\n🎯 Cleanup complete: {removed_count}/{len(files_to_remove)} files removed.")
        
        # Final verification
        remaining_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
        print(f"📁 Remaining .tif files: {len(remaining_files)}")
        
        if len(remaining_files) == len(files_to_keep):
            print("✅ Verification successful: Only expected files remain.")
        else:
            print("⚠️  Warning: File count mismatch. Some files may not have been processed correctly.")

    # 5. Shift dates if requested
    if shift_days != 0 and files_to_keep:
        print(f"\n📅 Shifting all dates by {shift_days:+d} days...")
        
        # Ask for confirmation if shifting many files
        if len(files_to_keep) > 10:
            response = input(f"Are you sure you want to rename {len(files_to_keep)} files? (yes/no): ")
            if response.lower() != 'yes':
                print("Date shifting cancelled.")
                return
        
        shifted_count = 0
        failed_count = 0
        
        for filename in files_to_keep:
            original_date = file_date_mapping[filename]
            new_date = original_date + timedelta(days=shift_days)
            
            # Create new filename with shifted date
            try:
                # Split filename to preserve prefix
                name_parts = filename.rsplit('_', 1)
                if len(name_parts) == 2:
                    prefix = name_parts[0]
                    extension = '.tif'
                    
                    # Create new filename with shifted date
                    new_date_str = new_date.strftime('%Y%m%d')
                    new_filename = f"{prefix}_{new_date_str}{extension}"
                    
                    # Check if new filename already exists
                    new_file_path = os.path.join(folder_path, new_filename)
                    if os.path.exists(new_file_path):
                        print(f"  ⚠️  Skipped: {filename} -> {new_filename} (target already exists)")
                        failed_count += 1
                        continue
                    
                    # Rename the file
                    old_file_path = os.path.join(folder_path, filename)
                    os.rename(old_file_path, new_file_path)
                    print(f"  ✅ Shifted: {filename} -> {new_filename} ({original_date} -> {new_date})")
                    shifted_count += 1
                    
                else:
                    print(f"  ❌ Failed to parse filename format: {filename}")
                    failed_count += 1
                    
            except OSError as e:
                print(f"  ❌ Failed to rename {filename}: {e}")
                failed_count += 1
        
        print(f"\n🎯 Date shifting complete: {shifted_count}/{len(files_to_keep)} files renamed.")
        if failed_count > 0:
            print(f"⚠️  {failed_count} files could not be renamed.")
        
        # Show new timeline
        if shifted_count > 0:
            print(f"\n📅 New timeline (shifted by {shift_days:+d} days):")
            shifted_dates = sorted([date + timedelta(days=shift_days) for date in expected_dates])
            for date in shifted_dates:
                print(f"  - {date.strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check for a 16-day interval image time series in a folder, optionally remove unexpected files, and shift dates.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "folder_path", 
        help="Path to the folder to check."
    )
    parser.add_argument(
        "start_date", 
        help="Start date of the time series (YYYY-MM-DD)."
    )
    parser.add_argument(
        "end_date", 
        help="End date of the time series (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--remove", 
        action="store_true",
        help="Remove all images that are not expected in the 16-day time series."
    )
    parser.add_argument(
        "--shift", 
        type=int,
        default=0,
        help="Number of days to add to all dates in the series (can be negative)."
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder_path):
        print(f"Error: Folder not found at '{args.folder_path}'")
    else:
        check_image_timeseries(args.folder_path, args.start_date, args.end_date, args.remove, args.shift) 