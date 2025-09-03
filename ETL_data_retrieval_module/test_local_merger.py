#!/usr/bin/env python3
"""
Test script để kiểm tra Satellite Data Merger với local API server
"""

import requests
import time
import sys
from satellite_data_merger import merge_satellite_data


def check_api_server(base_url="http://localhost:8000"):
    """Kiểm tra API server có đang chạy không"""
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API server đang chạy")
            return True
        else:
            print(f"❌ API server trả về status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Không thể kết nối đến API server")
        print("   Hãy chạy: python restapi/main.py")
        return False
    except Exception as e:
        print(f"❌ Lỗi kiểm tra API server: {e}")
        return False


def create_test_era5_task(base_url="http://localhost:8000"):
    """Tạo test task cho ERA5 data"""
    print("\n🔄 Tạo ERA5 test task...")
    
    # Payload cho ERA5 search
    era5_payload = {
        "bbox": [105.0, 20.0, 106.0, 21.0],  # Small bbox for testing
        "datetime": "2024-06-01T03:00:00Z/2024-06-01T05:00:00Z",  # Short time range
        "variables": ["2m_temperature", "surface_pressure"],  # 2 variables only
        "utc_hours": [3, 4],  # 2 hours only
        "limit": 10
    }
    
    try:
        response = requests.post(f"{base_url}/v1/era5_search", json=era5_payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            task_id = result.get("task_id")
            print(f"✅ ERA5 task created: {task_id}")
            print(f"   Items processed: {result.get('items_processed', 0)}")
            print(f"   Files created: {result.get('files_created', 0)}")
            return task_id
        else:
            print(f"❌ ERA5 search failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Lỗi tạo ERA5 task: {e}")
        return None


def test_download_endpoint(task_id, base_url="http://localhost:8000"):
    """Test download endpoint"""
    print(f"\n🔄 Test download endpoint cho task: {task_id}")
    
    download_url = f"{base_url}/v1/era5_download/{task_id}"
    
    try:
        # Test với stream=True và chỉ đọc headers để kiểm tra file
        response = requests.get(download_url, stream=True, timeout=10)
        
        if response.status_code == 200:
            print("✅ Download endpoint accessible")
            content_length = response.headers.get('content-length')
            if content_length:
                print(f"   File size: {int(content_length)} bytes")
            
            # Check content type
            content_type = response.headers.get('content-type')
            if content_type:
                print(f"   Content type: {content_type}")
            
            # Close connection without downloading full content
            response.close()
            return True
        else:
            print(f"❌ Download endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Lỗi test download: {e}")
        return False


def test_merger_with_real_data(task_ids, base_url="http://localhost:8000"):
    """Test merger với real task IDs"""
    print(f"\n🚀 Testing merger với task IDs: {task_ids}")
    
    try:
        output_dir = merge_satellite_data(
            task_ids=task_ids,
            output_folder="test_merged_output",
            target_crs="EPSG:4326",
            api_base_url=base_url
        )
        
        print(f"✅ Merger test completed!")
        print(f"📁 Output folder: {output_dir}")
        
        # Kiểm tra output files
        import os
        tif_files = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
        print(f"   Created {len(tif_files)} TIF files:")
        for tif_file in tif_files[:5]:  # Show first 5
            print(f"     - {tif_file}")
        if len(tif_files) > 5:
            print(f"     ... and {len(tif_files) - 5} more")
            
        return True
        
    except Exception as e:
        print(f"❌ Merger test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Satellite Data Merger - Local Test")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # 1. Check API server
    if not check_api_server(base_url):
        print("\n❌ Cannot proceed without API server")
        print("\nTo start API server:")
        print("1. cd /media/ekai2/data2tb/datmh/etl_eovplatform")
        print("2. source venv/bin/activate")
        print("3. python restapi/main.py")
        return 1
    
    # 2. Create test ERA5 task
    era5_task_id = create_test_era5_task(base_url)
    if not era5_task_id:
        print("\n❌ Cannot create ERA5 test task")
        return 1
    
    # 3. Wait a bit for task to complete (if needed)
    print("\n⏳ Waiting 5 seconds for task to complete...")
    time.sleep(5)
    
    # 4. Test download endpoint
    if not test_download_endpoint(era5_task_id, base_url):
        print("\n❌ Download endpoint not working")
        return 1
    
    # 5. Test merger with real data
    test_task_ids = {
        "era5": era5_task_id
        # Add more satellites if you have their endpoints working
    }
    
    if not test_merger_with_real_data(test_task_ids, base_url):
        print("\n❌ Merger test failed")
        return 1
    
    print("\n🎉 All tests passed!")
    print("\nNow you can use the merger with real task IDs:")
    print("```python")
    print("from satellite_data_merger import merge_satellite_data")
    print()
    print("task_ids = {")
    print(f'    "era5": "{era5_task_id}",')
    print('    # "sentinel2": "your-s2-task-id"')
    print("}")
    print()
    print("output_dir = merge_satellite_data(")
    print("    task_ids=task_ids,")
    print("    output_folder='my_merged_data',")
    print("    api_base_url='http://localhost:8000'")
    print(")")
    print("```")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
