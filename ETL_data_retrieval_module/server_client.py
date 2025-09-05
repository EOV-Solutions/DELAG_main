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
    - Download task artifacts as ZIPs using predefined task IDs
    - Extract downloaded ZIPs to temporary directories
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

    # Note: Task creation methods removed since we now use predefined task IDs
    # All tasks are created externally and task IDs are provided directly

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


