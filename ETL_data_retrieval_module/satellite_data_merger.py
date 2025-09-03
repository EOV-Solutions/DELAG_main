#!/usr/bin/env python3
"""
Satellite Data Merger - Standalone Utility
Hợp nhất dữ liệu từ nhiều vệ tinh dựa trên dictionary task_ids

Usage:
    python satellite_data_merger.py

Features:
    - Nhận dictionary task_ids (satellite_name -> task_id)
    - Giải nén dữ liệu từ các ZIP files
    - Nhóm files theo datetime/timestamp
    - Merge bands cùng thời điểm thành multi-band images
    - Xuất ra folder có tổ chức chứa merged data
"""

import os
import shutil
import tempfile
import zipfile
import rasterio
import numpy as np
import requests
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.windows import Window
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import re

# Configuration
CLIPPED_OUTPUT_DIR = "ETL_data_retrieval_module/data/"


def extract_datetime_from_filename(filename: str) -> Optional[str]:
    """
    Trích xuất datetime từ tên file của các vệ tinh khác nhau
    
    Args:
        filename: Tên file (ví dụ: 2m_temperature_era5_20240601_040000Z.tif)
    
    Returns:
        Datetime string hoặc None nếu không parse được
    """
    # Pattern cho ERA5: variable_era5_YYYYMMDD_HHMMSSZ.tif
    era5_pattern = r'era5_(\d{8})_(\d{6})Z\.tif'
    
    # Pattern cho ERA5 legacy: variable_item_id.tif với item_id chứa datetime
    era5_legacy_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
    
    # Pattern cho Sentinel-2: S2A_MSIL2A_YYYYMMDDTHHMMSS_...
    s2_pattern = r'S2[AB]_MSIL2A_(\d{8}T\d{6})_'
    
    # Pattern cho Sentinel-1: S1A_IW_GRD_YYYYMMDDTHHMMSS_...
    s1_pattern = r'S1[AB]_\w+_\w+_(\d{8}T\d{6})_'
    
    # Pattern tổng quát cho timestamp
    general_pattern = r'(\d{8}T\d{6}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
    
    # Try ERA5 current format first
    match = re.search(era5_pattern, filename)
    if match:
        date_part = match.group(1)  # YYYYMMDD
        time_part = match.group(2)  # HHMMSS
        try:
            dt = datetime.strptime(f"{date_part}T{time_part}", '%Y%m%dT%H%M%S')
            return dt.isoformat()
        except ValueError:
            pass
    
    # Try other patterns
    patterns = [era5_legacy_pattern, s2_pattern, s1_pattern, general_pattern]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(1)
            try:
                # Chuẩn hóa format datetime
                if 'T' in date_str and len(date_str) == 15:  # YYYYMMDDTHHMMSS
                    dt = datetime.strptime(date_str, '%Y%m%dT%H%M%S')
                    return dt.isoformat()
                elif 'T' in date_str and ':' in date_str:  # ISO format
                    return date_str
            except ValueError:
                continue
    
    print(f"⚠️  Không thể parse datetime từ filename: {filename}")
    return None


def download_task_data(task_id: str, satellite_name: str, api_base_url: str = "http://localhost:8000") -> str:
    """
    Download dữ liệu task từ API server bằng GET request
    
    Args:
        task_id: ID của task
        satellite_name: Tên vệ tinh (để xác định endpoint)
        api_base_url: Base URL của API server
    
    Returns:
        Đường dẫn đến file ZIP đã download
        
    Raises:
        Exception: Nếu download thất bại
    """
    # Xác định endpoint download dựa trên satellite name
    if satellite_name.lower() == "era5":
        download_url = f"{api_base_url}/v1/era5_download/{task_id}"
    elif satellite_name.lower().startswith("sentinel"):
        download_url = f"{api_base_url}/v1/s2_download/{task_id}"  # Giả định có endpoint tương tự
    else:
        # Generic download endpoint hoặc fallback
        download_url = f"{api_base_url}/v1/download/{task_id}"
    
    print(f"📡 Downloading {satellite_name} data từ: {download_url}")
    
    try:
        # Gửi GET request để download file
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Tạo thư mục tạm để lưu file ZIP
        download_dir = tempfile.mkdtemp(prefix=f"download_{satellite_name}_")
        zip_filename = f"{task_id}.zip"
        zip_path = os.path.join(download_dir, zip_filename)
        
        # Lưu file ZIP
        with open(zip_path, 'wb') as f:
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)
        
        print(f"   ✓ Downloaded {total_size} bytes to {zip_path}")
        return zip_path
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Lỗi download {satellite_name} task {task_id}: {e}")
    except Exception as e:
        raise Exception(f"Lỗi lưu file {satellite_name}: {e}")


def extract_task_data(task_id: str, satellite_name: str, temp_dir: str, api_base_url: str = "http://localhost:8000") -> str:
    """
    Download và giải nén dữ liệu từ API server
    
    Args:
        task_id: ID của task
        satellite_name: Tên vệ tinh
        temp_dir: Thư mục tạm để giải nén
        api_base_url: Base URL của API server
    
    Returns:
        Đường dẫn đến thư mục chứa dữ liệu đã giải nén
    """
    # Download ZIP file từ API
    zip_path = download_task_data(task_id, satellite_name, api_base_url)
    extract_dir = os.path.join(temp_dir, f"{satellite_name}_{task_id}")
    
    print(f"📂 Giải nén {os.path.basename(zip_path)} vào {satellite_name} folder...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Đếm số files đã giải nén
        tif_count = sum(1 for root, dirs, files in os.walk(extract_dir) 
                       for file in files if file.endswith('.tif'))
        print(f"   ✓ Đã giải nén {tif_count} TIF files từ {satellite_name}")
        
        return extract_dir
        
    finally:
        # Cleanup downloaded ZIP file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            # Cũng xóa download directory nếu rỗng
            download_dir = os.path.dirname(zip_path)
            try:
                os.rmdir(download_dir)
            except OSError:
                pass  # Directory không rỗng hoặc đã bị xóa


def group_files_by_datetime(extract_dirs: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    """
    Nhóm các file theo datetime từ tất cả satellites
    
    Args:
        extract_dirs: Dict mapping satellite_name -> extracted_directory_path
    
    Returns:
        Dict mapping datetime -> satellite_name -> list_of_files
    """
    datetime_groups = defaultdict(lambda: defaultdict(list))
    
    print("\n🔍 Phân tích và nhóm files theo datetime...")
    
    for satellite_name, extract_dir in extract_dirs.items():
        print(f"   Satellite: {satellite_name}")
        
        # Tìm tất cả .tif files trong thư mục đã giải nén
        file_count = 0
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.tif'):
                    file_path = os.path.join(root, file)
                    datetime_str = extract_datetime_from_filename(file)
                    
                    if datetime_str:
                        datetime_groups[datetime_str][satellite_name].append(file_path)
                        file_count += 1
                    else:
                        print(f"     ⚠️  Bỏ qua {file} - không parse được datetime")
        
        print(f"     ✓ Đã phân loại {file_count} files")
    
    print(f"\n📊 Tổng kết: {len(datetime_groups)} datetime groups được tạo")
    return dict(datetime_groups)


def get_common_bounds_and_resolution(file_paths: List[str]) -> Tuple[List[float], float, str]:
    """
    Tính toán bounds chung và resolution cho danh sách files
    
    Args:
        file_paths: Danh sách đường dẫn đến files
    
    Returns:
        Tuple (bounds, resolution, crs) chung
    """
    all_bounds = []
    all_resolutions = []
    all_crs = set()
    
    for file_path in file_paths:
        try:
            with rasterio.open(file_path) as src:
                bounds = src.bounds
                all_bounds.append([bounds.left, bounds.bottom, bounds.right, bounds.top])
                all_resolutions.append(abs(src.transform[0]))  # pixel size
                all_crs.add(src.crs.to_string())
        except Exception as e:
            print(f"⚠️  Lỗi đọc {file_path}: {e}")
            continue
    
    if not all_bounds:
        raise ValueError("Không có file nào hợp lệ để tính bounds")
    
    # Tính union của tất cả bounds
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    
    common_bounds = [min_x, min_y, max_x, max_y]
    
    # Sử dụng resolution thô nhất (lớn nhất) để tránh oversample
    common_resolution = max(all_resolutions)
    
    # Sử dụng CRS phổ biến nhất hoặc EPSG:4326 nếu có nhiều CRS
    if len(all_crs) == 1:
        common_crs = all_crs.pop()
    else:
        print(f"🌍 Phát hiện nhiều CRS: {all_crs}, sử dụng EPSG:4326")
        common_crs = "EPSG:4326"
    
    print(f"🗺️  Common bounds: {[round(x, 6) for x in common_bounds]}")
    print(f"📏 Common resolution: {common_resolution:.6f}°")
    print(f"🌐 Common CRS: {common_crs}")
    
    return common_bounds, common_resolution, common_crs


def reproject_to_common_grid(
    file_path: str,
    target_bounds: List[float],
    target_resolution: float,
    target_crs: str
) -> Tuple[np.ndarray, rasterio.transform.Affine, dict]:
    """
    Reproject file về common grid
    
    Args:
        file_path: Đường dẫn đến file input
        target_bounds: Bounds mục tiêu [minx, miny, maxx, maxy]
        target_resolution: Resolution mục tiêu
        target_crs: CRS mục tiêu
    
    Returns:
        Tuple (data_array, transform, profile)
    """
    with rasterio.open(file_path) as src:
        # Tính transform và dimensions cho target grid
        minx, miny, maxx, maxy = target_bounds
        target_transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy,
            int((maxx - minx) / target_resolution),
            int((maxy - miny) / target_resolution)
        )
        
        target_width = int((maxx - minx) / target_resolution)
        target_height = int((maxy - miny) / target_resolution)
        
        # Tạo array để chứa dữ liệu reprojected
        target_data = np.full((target_height, target_width), src.nodata or -9999, dtype=src.dtypes[0])
        
        # Reproject dữ liệu
        reproject(
            source=rasterio.band(src, 1),
            destination=target_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )
        
        # Tạo profile cho output
        profile = src.profile.copy()
        profile.update({
            'crs': target_crs,
            'transform': target_transform,
            'width': target_width,
            'height': target_height
        })
        
        return target_data, target_transform, profile


def merge_datetime_bands(
    datetime_files: Dict[str, List[str]],
    target_bounds: List[float],
    target_resolution: float,
    target_crs: str,
    output_dir: str,
    datetime_key: str
) -> str:
    """
    Merge tất cả bands từ các satellites cho một datetime cụ thể
    
    Args:
        datetime_files: Dict mapping satellite_name -> list_of_files cho datetime này
        target_bounds: Bounds mục tiêu
        target_resolution: Resolution mục tiêu
        target_crs: CRS mục tiêu
        output_dir: Thư mục output
        datetime_key: Key datetime để đặt tên file
    
    Returns:
        Đường dẫn đến merged file
    """
    all_bands = []
    band_descriptions = []
    
    print(f"🔄 Merging datetime: {datetime_key}")
    
    # Xử lý từng satellite
    for satellite_name, files in datetime_files.items():
        print(f"   📡 {satellite_name}: {len(files)} files")
        
        for file_path in files:
            try:
                # Reproject và clip file về common grid
                data, transform, profile = reproject_to_common_grid(
                    file_path, target_bounds, target_resolution, target_crs
                )
                
                all_bands.append(data)
                
                # Tạo description cho band
                filename = os.path.basename(file_path)
                variable_name = filename.replace('.tif', '').split('_')[0]  # Extract variable name
                band_desc = f"{satellite_name}_{variable_name}"
                band_descriptions.append(band_desc)
                
            except Exception as e:
                print(f"     ❌ Lỗi xử lý {os.path.basename(file_path)}: {e}")
                continue
    
    if not all_bands:
        raise ValueError(f"Không có bands nào được xử lý cho datetime {datetime_key}")
    
    # Tạo output filename
    safe_datetime = datetime_key.replace(':', '').replace('-', '').replace('T', '_')
    output_filename = f"merged_{safe_datetime}.tif"
    output_path = os.path.join(output_dir, output_filename)
    
    # Cập nhật profile cho multi-band output
    profile.update({
        'count': len(all_bands),
        'compress': 'lzw'
    })
    
    # Lưu multi-band file
    with rasterio.open(output_path, "w", **profile) as dst:
        for i, (band_data, description) in enumerate(zip(all_bands, band_descriptions), 1):
            dst.write(band_data, i)
            dst.set_band_description(i, description)
        
        # Thêm metadata
        dst.update_tags(
            DATETIME=datetime_key,
            BANDS_COUNT=len(all_bands),
            SATELLITE_SOURCES=','.join(datetime_files.keys()),
            CREATION_TIME=datetime.now().isoformat()
        )
    
    print(f"   ✅ Tạo {output_filename} với {len(all_bands)} bands")
    return output_path


def merge_satellite_data(
    task_ids: Dict[str, str],
    output_folder: str = "merged_satellite_data",
    target_crs: str = "EPSG:4326",
    api_base_url: str = "http://localhost:8000"
) -> str:
    """
    Main function để merge dữ liệu từ nhiều satellites
    
    Args:
        task_ids: Dict mapping satellite_name -> task_id
        output_folder: Tên folder output (sẽ tạo trong thư mục hiện tại)
        target_crs: Target CRS cho output
        api_base_url: Base URL của API server (mặc định localhost:8000)
    
    Returns:
        Đường dẫn đến folder output
        
    Example:
        task_ids = {
            "era5": "12345678-1234-1234-1234-123456789abc",
            "sentinel2": "87654321-4321-4321-4321-cba987654321"
        }
        output_dir = merge_satellite_data(task_ids, "my_merged_data", api_base_url="http://localhost:8000")
    """
    print("🚀 Bắt đầu merge satellite data")
    print("=" * 50)
    print(f"📝 Input satellites: {list(task_ids.keys())}")
    print(f"📂 Output folder: {output_folder}")
    print(f"🌐 Target CRS: {target_crs}")
    print()
    
    # Tạo thư mục output
    output_dir = os.path.abspath(output_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    # Tạo thư mục tạm
    temp_dir = tempfile.mkdtemp(prefix="satellite_merger_")
    
    try:
        # 1. Download và giải nén dữ liệu từ tất cả tasks
        print("🔓 Bước 1: Download và giải nén dữ liệu từ task IDs")
        extract_dirs = {}
        for satellite_name, task_id in task_ids.items():
            try:
                extract_dir = extract_task_data(task_id, satellite_name, temp_dir, api_base_url)
                extract_dirs[satellite_name] = extract_dir
            except Exception as e:
                print(f"❌ {e}")
                continue
        
        if not extract_dirs:
            raise ValueError("Không có task nào được giải nén thành công")
        
        # 2. Nhóm files theo datetime
        print("\n🔄 Bước 2: Nhóm files theo datetime")
        datetime_groups = group_files_by_datetime(extract_dirs)
        
        if not datetime_groups:
            raise ValueError("Không tìm thấy dữ liệu để merge")
        
        # 3. Tính toán common bounds và resolution
        print("\n📐 Bước 3: Tính toán common grid")
        all_files = []
        for dt_files in datetime_groups.values():
            for sat_files in dt_files.values():
                all_files.extend(sat_files)
        
        common_bounds, common_resolution, detected_crs = get_common_bounds_and_resolution(all_files)
        target_crs = target_crs or detected_crs
        
        # 4. Merge từng datetime group
        print(f"\n🔗 Bước 4: Merge {len(datetime_groups)} datetime groups")
        merged_files = []
        
        for i, (datetime_key, datetime_files) in enumerate(datetime_groups.items(), 1):
            try:
                print(f"\n[{i}/{len(datetime_groups)}]", end=" ")
                merged_file = merge_datetime_bands(
                    datetime_files,
                    common_bounds,
                    common_resolution,
                    target_crs,
                    output_dir,
                    datetime_key
                )
                merged_files.append(merged_file)
            except Exception as e:
                print(f"❌ Lỗi merge datetime {datetime_key}: {e}")
                continue
        
        if not merged_files:
            raise ValueError("Không có file nào được merge thành công")
        
        # 5. Tạo summary file
        summary_path = os.path.join(output_dir, "merge_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("SATELLITE DATA MERGER SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Merge time: {datetime.now().isoformat()}\n")
            f.write(f"Input satellites: {', '.join(task_ids.keys())}\n")
            f.write(f"Task IDs: {task_ids}\n")
            f.write(f"Target CRS: {target_crs}\n")
            f.write(f"Resolution: {common_resolution:.6f}\n")
            f.write(f"Bounds: {common_bounds}\n")
            f.write(f"Datetime groups processed: {len(datetime_groups)}\n")
            f.write(f"Merged files created: {len(merged_files)}\n\n")
            f.write("MERGED FILES:\n")
            for file_path in merged_files:
                f.write(f"  - {os.path.basename(file_path)}\n")
        
        print(f"\n🎉 Hoàn thành! Đã tạo {len(merged_files)} merged files")
        print(f"📂 Output folder: {output_dir}")
        print(f"📄 Summary: {summary_path}")
        
        return output_dir
        
    finally:
        # Cleanup thư mục tạm
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """
    Example usage
    """
    # Example task IDs - thay thế bằng task IDs thực tế
    task_ids = {
        "era5": "12345678-1234-1234-1234-123456789abc",
        "sentinel2": "87654321-4321-4321-4321-cba987654321"
        # Thêm các satellites khác nếu cần
    }
    
    try:
        output_dir = merge_satellite_data(
            task_ids=task_ids,
            output_folder="merged_satellite_data",
            target_crs="EPSG:4326",
            api_base_url="http://localhost:8000"
        )
        
        print(f"\n✅ Merge completed successfully!")
        print(f"📁 Check output folder: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during merge: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
