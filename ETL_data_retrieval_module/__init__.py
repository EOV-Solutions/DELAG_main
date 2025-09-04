"""
ETL Data Retrieval Module

Server-based satellite data retrieval system that replaces the GEE-based workflow.
Supports ERA5, Landsat LST, and Sentinel-2 data downloads and processing.

Main entry point: python -m ETL_data_retrieval_module.main
"""

from .server_client import ServerClient
from .era5_from_server import retrieve_era5_from_server
from .lst_from_server import retrieve_lst_from_server
from .s2_from_server import retrieve_s2_from_server
from .aster_from_server import retrieve_aster_from_server
from .utils import find_grid_feature, bbox_from_feature
from .config import config

__all__ = [
    'ServerClient',
    'retrieve_era5_from_server',
    'retrieve_lst_from_server', 
    'retrieve_s2_from_server',
    'retrieve_aster_from_server',
    'find_grid_feature',
    'bbox_from_feature',
    'config'
]
