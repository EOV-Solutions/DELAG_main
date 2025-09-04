import os
import tempfile
import zipfile
import shutil
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import config


class ServerClient:
    """
    Minimal HTTP client for the ETL retrieval server.

    Responsibilities:
    - Create tasks (e.g., era5_search)
    - Download task artifacts as ZIPs and extract to a temp directory
    - Provide simple helpers for endpoint path resolution
    """

    def __init__(self, api_base_url: str = None, timeout: int = None):
        self.api_base_url = (api_base_url or config.DEFAULT_API_BASE_URL).rstrip("/")
        self.timeout = timeout or config.DEFAULT_TIMEOUT

    # -----------------------------
    # Core HTTP helpers
    # -----------------------------
    def _url(self, path: str) -> str:
        return f"{self.api_base_url}{path}"

    def post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(self._url(path), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_stream(self, path: str) -> requests.Response:
        response = requests.get(self._url(path), stream=True, timeout=self.timeout)
        response.raise_for_status()
        return response

    # -----------------------------
    # Task creation
    # -----------------------------
    def create_era5_task(
        self,
        bbox: List[float],
        datetime_range_iso: str,
        variables: Optional[List[str]] = None,
        utc_hours: Optional[List[int]] = None,
        limit: Optional[int] = None,
    ) -> str:
        """
        Create an ERA5 task on the server and return task_id.

        Args:
            bbox: [minx, miny, maxx, maxy] in EPSG:4326
            datetime_range_iso: ISO interval, e.g. "2024-06-01T00:00:00Z/2024-06-02T00:00:00Z"
            variables: e.g. ["2m_temperature", "skin_temperature"]
            utc_hours: list of hours to include, e.g. [10, 11]
            limit: optional maximum items
        """
        payload: Dict[str, Any] = {
            "bbox": bbox,
            "datetime": datetime_range_iso,
        }
        if variables:
            payload["variables"] = variables
        if utc_hours:
            payload["utc_hours"] = utc_hours
        if limit is not None:
            payload["limit"] = limit

        endpoint = config.get_search_endpoint("era5")
        result = self.post_json(endpoint, payload)
        task_id = result.get("task_id")
        if not task_id:
            raise RuntimeError(f"ERA5 search returned no task_id. Response: {result}")
        return task_id

    def create_s2_task(
        self,
        bbox: List[float],
        datetime_range_iso: str,
        limit: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create S2 search task"""
        payload: Dict[str, Any] = {
            "bbox": bbox,
            "datetime": datetime_range_iso,
        }
        if limit is not None:
            payload["limit"] = limit
        if extra:
            payload.update(extra)

        endpoint = config.get_search_endpoint("s2")
        result = self.post_json(endpoint, payload)
        task_id = result.get("task_id")
        if not task_id:
            raise RuntimeError(f"S2 search returned no task_id. Response: {result}")
        return task_id
    
    def create_landsat_task(
        self,
        satellite: str,  # "L8" or "L9" 
        level: str,      # "l1" or "l2"
        bbox: List[float],
        datetime_range_iso: str,
        bands: Optional[List[str]] = None,
        cloud_cover: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Create Landsat search task for specific satellite and processing level"""
        dataset_key = f"landsat{satellite[1].lower()}_{level.lower()}"
        
        payload: Dict[str, Any] = {
            "bbox": bbox,
            "datetime": datetime_range_iso,
        }
        if bands:
            payload["bands"] = bands
        if cloud_cover is not None:
            payload["cloud_cover"] = cloud_cover
        if limit is not None:
            payload["limit"] = limit
        
        endpoint = config.get_search_endpoint(dataset_key)
        if not endpoint:
            raise ValueError(f"Unknown Landsat dataset: {dataset_key}")
            
        result = self.post_json(endpoint, payload)
        task_id = result.get("task_id")
        if not task_id:
            raise RuntimeError(f"Landsat {satellite} {level} search returned no task_id. Response: {result}")
        return task_id
    
    def create_aster_task(
        self,
        bbox: List[float],
        datetime_range_iso: str,
        bands: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Create ASTER search task"""
        payload: Dict[str, Any] = {
            "bbox": bbox,
            "datetime": datetime_range_iso,
        }
        if bands:
            payload["bands"] = bands
        if limit is not None:
            payload["limit"] = limit
        
        endpoint = config.get_search_endpoint("aster")
        result = self.post_json(endpoint, payload)
        task_id = result.get("task_id")
        if not task_id:
            raise RuntimeError(f"ASTER search returned no task_id. Response: {result}")
        return task_id

    # -----------------------------
    # Download helpers
    # -----------------------------
    def _download_endpoint_for(self, dataset_name: str, task_id: str) -> str:
        """Get download endpoint URL for a dataset and task_id"""
        endpoint_base = config.get_download_endpoint(dataset_name.lower())
        if endpoint_base:
            return f"{endpoint_base}/{task_id}"
        # fallback generic path
        return f"/v1/download/{task_id}"

    def download_task_zip(self, dataset_name: str, task_id: str) -> str:
        """
        Download a task ZIP to a temp dir and return its file path.
        Caller is responsible for removing the temp dir or zip if desired.
        """
        endpoint = self._download_endpoint_for(dataset_name, task_id)
        response = self.get_stream(endpoint)

        temp_dir = tempfile.mkdtemp(prefix=f"download_{dataset_name}_")
        zip_path = os.path.join(temp_dir, f"{task_id}.zip")

        total_size = 0
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    total_size += len(chunk)

        print(f"   ✓ Downloaded {total_size} bytes to {zip_path}")
        return zip_path

    def extract_zip(self, zip_path: str, extract_dir: Optional[str] = None) -> str:
        """
        Extract a ZIP file into a new or provided directory and return the directory path.
        """
        if extract_dir is None:
            extract_dir = tempfile.mkdtemp(prefix="extracted_")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        return extract_dir

    def download_and_extract(self, dataset_name: str, task_id: str) -> str:
        """
        Convenience to download a task ZIP and extract it.
        Returns extracted directory path.
        """
        zip_path = self.download_task_zip(dataset_name, task_id)
        try:
            extracted_dir = self.extract_zip(zip_path)
        finally:
            # Attempt to clean the downloaded zip and its parent folder
            try:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                    parent = os.path.dirname(zip_path)
                    if os.path.isdir(parent):
                        shutil.rmtree(parent, ignore_errors=True)
            except Exception:
                pass
        return extracted_dir


