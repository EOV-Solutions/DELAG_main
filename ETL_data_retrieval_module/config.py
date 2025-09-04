"""
ETL Data Retrieval Module Configuration

Centralized configuration for all API endpoints and module settings.
"""

from typing import Dict, List, Optional


class ETLConfig:
    """Configuration class for ETL data retrieval endpoints and settings"""
    
    # ============================
    # API Endpoints Configuration
    # ============================
    
    # Base API settings
    DEFAULT_API_BASE_URL = "http://localhost:8000"
    DEFAULT_TIMEOUT = 120
    
    # Search endpoints
    SEARCH_ENDPOINTS = {
        "era5": "/v1/era5_search",
        "s2": "/v1/s2_search", 
        "sentinel2": "/v1/s2_search",  # Alias
        "landsat8_l1": "/v1/landsat8_l1_search",
        "landsat8_l2": "/v1/landsat8_l2_search", 
        "landsat9_l1": "/v1/landsat9_l1_search",
        "landsat9_l2": "/v1/landsat9_l2_search",
        "aster": "/v1/aster_search"
    }
    
    # Download endpoints
    DOWNLOAD_ENDPOINTS = {
        "era5": "/v1/era5_download",
        "s2": "/v1/s2_download",
        "sentinel2": "/v1/s2_download",  # Alias
        "landsat8_l1": "/v1/landsat8_l1_download",
        "landsat8_l2": "/v1/landsat8_l2_download",
        "landsat9_l1": "/v1/landsat9_l1_download", 
        "landsat9_l2": "/v1/landsat9_l2_download",
        "aster": "/v1/aster_download"
    }
    
    # ============================
    # Dataset Configuration
    # ============================
    
    # ERA5 configuration
    ERA5_CONFIG = {
        "default_variables": ["skin_temperature"],
        "default_utc_hours": [7],  # Default to 10 UTC (typical LST acquisition time)
        "default_limit": 50
    }
    
    # Sentinel-2 configuration  
    S2_CONFIG = {
        "default_bands": ["B2", "B3", "B4", "B8"],  # Blue, Green, Red, NIR
        "default_cloud_cover": 85.0,
        "composite_window_days": 4,  # ±4 days for 8-day composites
        "default_limit": 50
    }
    
    # Landsat configuration
    LANDSAT_CONFIG = {
        "L8": {
            "l1_endpoint_key": "landsat8_l1",
            "l2_endpoint_key": "landsat8_l2", 
            "tir_bands_l1": ["B10", "B11"],
            "tir_bands_l2": ["ST_B10"],
            "optical_bands_l2": ["SR_B4", "SR_B5"],  # Red, NIR for NDVI
            "red_band": "B4",
            "nir_band": "B5",
            "tir_band": "B10"
        },
        "L9": {
            "l1_endpoint_key": "landsat9_l1",
            "l2_endpoint_key": "landsat9_l2",
            "tir_bands_l1": ["B10", "B11"], 
            "tir_bands_l2": ["ST_B10"],
            "optical_bands_l2": ["SR_B4", "SR_B5"],
            "red_band": "B4",
            "nir_band": "B5", 
            "tir_band": "B10"
        },
        "default_cloud_cover": 80.0,
        "default_limit": 5,
        "cadence_days": 16  # Landsat repeat cycle
    }
    
    # ASTER configuration
    ASTER_CONFIG = {
        "default_bands": ["emissivity_band10", "emissivity_band11", "ndvi"],
        "emissivity_scaling": 0.001,
        "ndvi_scaling": 0.01
    }
    
    # ============================
    # Output Configuration  
    # ============================
    
    # Output file naming patterns
    OUTPUT_PATTERNS = {
        "era5": "ERA5_data_{date}.tif",
        "s2": "S2_8days_{date}.tif",
        "lst_l8": "L8_lst16days_{date}.tif",
        "lst_l9": "L9_lst16days_{date}.tif"
    }
    
    # Output directory structure
    OUTPUT_DIRS = {
        "era5": "era5",
        "s2": "s2_images", 
        "lst": "lst",
        "aster": "aster_ged"
    }
    
    # Nodata values for different datasets
    NODATA_VALUES = {
        "era5": "nan",  # Use NaN for ERA5
        "s2": -100,     # Use -100 for S2 to match preprocessing
        "lst": "nan",   # Use NaN for LST
        "aster": "nan"  # Use NaN for ASTER GED
    }
    
    # ============================
    # Processing Configuration
    # ============================
    
    # LST processing settings
    LST_PROCESSING = {
        "use_ndvi": True,  # Use dynamic emissivity calculation
        "ndvi_thresholds": {
            "ndvi_bg": 0.2,   # Bare ground NDVI
            "ndvi_vg": 0.86   # Full vegetation NDVI
        },
        "default_emissivity": 0.95,  # Fallback emissivity
        "default_tpw": 20.0,  # Default total precipitable water (mm)
        # Simplified SMW coefficients (would be lookup tables in full implementation)
        "smw_coefficients": {
            "L8": {"A": 0.04, "B": 0.95, "C": 1.85},
            "L9": {"A": 0.04, "B": 0.95, "C": 1.85}, 
            "L5": {"A": 0.06, "B": 0.90, "C": 2.10},
            "L7": {"A": 0.06, "B": 0.90, "C": 2.10}
        }
    }
    
    # ============================
    # Helper Methods
    # ============================
    
    @classmethod
    def get_search_endpoint(cls, dataset: str) -> Optional[str]:
        """Get search endpoint for a dataset"""
        return cls.SEARCH_ENDPOINTS.get(dataset.lower())
    
    @classmethod  
    def get_download_endpoint(cls, dataset: str) -> Optional[str]:
        """Get download endpoint for a dataset"""
        return cls.DOWNLOAD_ENDPOINTS.get(dataset.lower())
    
    @classmethod
    def get_landsat_config(cls, satellite: str) -> Optional[Dict]:
        """Get configuration for a specific Landsat satellite"""
        return cls.LANDSAT_CONFIG.get(satellite.upper())
    
    @classmethod
    def get_output_pattern(cls, dataset: str) -> str:
        """Get output filename pattern for a dataset"""
        return cls.OUTPUT_PATTERNS.get(dataset, "{dataset}_{date}.tif")
    
    @classmethod
    def get_output_dir(cls, dataset: str) -> str:
        """Get output directory name for a dataset"""
        return cls.OUTPUT_DIRS.get(dataset, dataset)
    
    @classmethod
    def get_nodata_value(cls, dataset: str):
        """Get nodata value for a dataset"""
        nodata = cls.NODATA_VALUES.get(dataset, "nan")
        if nodata == "nan":
            import numpy as np
            return np.nan
        return nodata


# Global config instance
config = ETLConfig()
