#!/usr/bin/env python3
"""
Query and download Sentinel-3 SL_2_LST___ products for a random ROI from a GeoJSON using
Copernicus Data Space Ecosystem (CDSE) OData API.

- Automatically unzips downloaded archives
- Stores extracted products under a structured schema:
  data/cdse_store/<Collection>/<ProductType>/<Platform>/<YYYY>/<MM>/<ProductName>/
  and writes a metadata.json per product
- Clips any GeoTIFF assets inside each product directory to the ROI into a 'clips' subfolder

Environment variables:
- SENTINEL_USER: CDSE username (same as Copernicus Browser)
- SENTINEL_PASSWORD: CDSE password
- SENTINEL_API_URL: Optional base URL (default: https://catalogue.dataspace.copernicus.eu)
- CDSE_KEEP_ZIP: If set to '1' or 'true', keep the downloaded ZIP alongside extracted data (default: delete ZIP)

Dependencies for clipping:
- rasterio, shapely, pyproj

Usage:
    python data_retrival_module/test_s3.py
"""

import os
import json
import random
from pathlib import Path
from typing import Tuple, Dict, Any, Iterable, Optional
from datetime import datetime
import zipfile

import requests
from requests import Response
from sentinelsat import geojson_to_wkt

# Optional clipping deps
try:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import shape as shp_shape, mapping as shp_mapping
    from shapely.ops import transform as shp_transform
    import pyproj
    _CLIP_DEPS_OK = True
except Exception:  # noqa: BLE001
    _CLIP_DEPS_OK = False

GEOJSON_PATH = "data/Grid_50K_MatchedDates.geojson"
DATE_RANGE = ("2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z")
COLLECTION = "SENTINEL-3"
PRODUCT_TYPE = "SL_2_LST___"
DOWNLOAD_DIR = Path("downloads/sentinel3_random_roi")
STORE_ROOT_DIR = Path("data/cdse_store")
KEEP_ZIP = os.environ.get("CDSE_KEEP_ZIP", "0").lower() in {"1", "true", "yes"}

# CDSE endpoints
DEFAULT_BASE_URL = "https://catalogue.dataspace.copernicus.eu"
DOWNLOAD_BASE_URL = "https://download.dataspace.copernicus.eu"
TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)


def select_random_footprint(geojson_path: str) -> Tuple[str, Dict[str, Any]]:
    """Load a GeoJSON, select a random feature, and return (WKT_footprint, feature)."""
    with open(geojson_path, "r") as f:
        gj = json.load(f)
    features = gj.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_path}")
    feature = random.choice(features)
    geometry = feature.get("geometry")
    if geometry is None:
        raise ValueError("Selected feature has no geometry")
    footprint_wkt = geojson_to_wkt(geometry)
    return footprint_wkt, feature


def get_access_token(username: str, password: str, timeout_sec: float = 30.0) -> str:
    """Obtain OAuth2 access token for CDSE (Resource Owner Password Credentials)."""
    data = {
        "grant_type": "password",
        "client_id": "cdse-public",
        "username": username,
        "password": password,
    }
    r: Response = requests.post(TOKEN_URL, data=data, timeout=timeout_sec)
    r.raise_for_status()
    token = r.json().get("access_token")
    if not token:
        raise RuntimeError("Failed to obtain access token from CDSE")
    return token


def build_odata_filter(footprint_wkt: str, start_iso: str, end_iso: str) -> str:
    """Build OData $filter string for collection, productType, time, and spatial intersects."""
    # Spatial operator expects geography SRID=4326 with WKT geometry
    spatial = f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint_wkt}')"
    # Product type is stored in Attributes as string attributes
    prod_type = (
        "Attributes/OData.CSC.StringAttribute/any(a: a/Name eq 'productType' and a/Value eq '"
        + PRODUCT_TYPE
        + "')"
    )
    collection = f"Collection/Name eq '{COLLECTION}'"
    # Time filter uses product content start/end datetimes
    time_filter = f"ContentDate/Start ge {start_iso} and ContentDate/End le {end_iso}"
    return f"{collection} and {prod_type} and {spatial} and {time_filter}"


def query_products(
    token: str,
    base_url: str,
    odata_filter: str,
    top: int = 100,
    orderby: str = "ContentDate/Start desc",
    timeout_sec: float = 60.0,
) -> Dict[str, Any]:
    """Call OData Products endpoint with filter; return parsed JSON."""
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "$filter": odata_filter,
        "$top": str(top),
        "$orderby": orderby,
    }
    url = f"{base_url.rstrip('/')}/odata/v1/Products"
    r: Response = requests.get(url, headers=headers, params=params, timeout=timeout_sec)
    r.raise_for_status()
    return r.json()


def download_product(
    token: str,
    product_id: str,
    out_dir: Path,
    filename_hint: Optional[str] = None,
    timeout_sec: float = 600.0,
) -> Path:
    """Download single product by Id via OData $value endpoint from CDSE download host."""
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{DOWNLOAD_BASE_URL}/odata/v1/Products({product_id})/$value"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Choose filename
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = filename_hint or f"product_{product_id}.zip"
    # Ensure .zip suffix
    if not fname.lower().endswith(".zip"):
        fname = f"{fname}_{ts}.zip"
    out_path = out_dir / fname
    with requests.get(url, headers=headers, stream=True, timeout=timeout_sec, allow_redirects=False) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out_path


def _get_attr_value(attributes: Iterable[Dict[str, Any]], name: str) -> Optional[str]:
    """Return attribute Value by Name from OData Attributes list."""
    if not attributes:
        return None
    for a in attributes:
        if isinstance(a, dict) and a.get("Name") == name:
            return a.get("Value")
    return None


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Safely extract ZIP to destination, preventing path traversal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            member_path = Path(member.filename)
            # Skip absolute paths or parent refs
            resolved = (dest_dir / member_path).resolve()
            if not str(resolved).startswith(str(dest_dir.resolve()) + os.sep):
                continue
            if member.is_dir():
                resolved.mkdir(parents=True, exist_ok=True)
            else:
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member, 'r') as src, open(resolved, 'wb') as dst:
                    dst.write(src.read())


def determine_storage_dir(store_root: Path, product: Dict[str, Any]) -> Path:
    """Compute storage directory based on collection, productType, platform, and date."""
    collection_name = (
        (product.get("Collection") or {}).get("Name")
        or COLLECTION
    )
    attrs = product.get("Attributes") or []
    prod_type = _get_attr_value(attrs, "productType") or PRODUCT_TYPE
    platform = _get_attr_value(attrs, "platformShortName") or "UNKNOWN"
    name = product.get("Name") or str(product.get("Id"))
    # Dates
    start_iso = ((product.get("ContentDate") or {}).get("Start")) or DATE_RANGE[0]
    try:
        start_dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    except Exception:
        start_dt = datetime.utcnow()
    year = f"{start_dt.year:04d}"
    month = f"{start_dt.month:02d}"
    target = store_root / collection_name / prod_type / platform / year / month / name
    return target


def write_metadata(metadata_dir: Path, product: Dict[str, Any]) -> None:
    metadata_dir.mkdir(parents=True, exist_ok=True)
    out = metadata_dir / "metadata.json"
    with open(out, "w") as f:
        json.dump(product, f, indent=2)


def _reproject_geometry_to_crs(geom_geojson: Dict[str, Any], dst_crs: str) -> Optional[Dict[str, Any]]:
    """Reproject GeoJSON geometry (assumed EPSG:4326) to dst_crs and return GeoJSON mapping."""
    if not _CLIP_DEPS_OK:
        return None
    try:
        geom = shp_shape(geom_geojson)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
        geom_proj = shp_transform(transformer.transform, geom)
        return shp_mapping(geom_proj)
    except Exception:
        return None


def clip_rasters_in_dir(product_dir: Path, roi_geom_geojson: Dict[str, Any]) -> int:
    """Clip all GeoTIFFs under product_dir to ROI; save into product_dir/'clips'. Returns count."""
    if not _CLIP_DEPS_OK:
        print("Clipping skipped: rasterio/shapely/pyproj not available.")
        return 0
    clips_dir = product_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for tif_path in list(product_dir.rglob("*.tif")) + list(product_dir.rglob("*.tiff")):
        try:
            with rasterio.open(tif_path) as src:
                if src.crs is None:
                    print(f"Skipping (no CRS): {tif_path}")
                    continue
                roi_in_raster_crs = _reproject_geometry_to_crs(roi_geom_geojson, src.crs.to_string())
                if roi_in_raster_crs is None:
                    print(f"Skipping (cannot reproject ROI): {tif_path}")
                    continue
                out_image, out_transform = rio_mask(src, [roi_in_raster_crs], crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "lzw",
                })
                out_name = tif_path.stem + "_clip.tif"
                out_path = clips_dir / out_name
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image)
                count += 1
        except Exception as e:  # noqa: BLE001
            print(f"Failed to clip {tif_path}: {e}")
            continue
    return count


def main() -> None:
    user = os.environ.get("SENTINEL_USER")
    password = os.environ.get("SENTINEL_PASSWORD")
    base_url = os.environ.get("SENTINEL_API_URL", DEFAULT_BASE_URL)

    if not user or not password:
        raise RuntimeError(
            "Please set SENTINEL_USER and SENTINEL_PASSWORD environment variables"
        )

    # Guard against deprecated SciHub endpoints
    if base_url and "apihub.copernicus.eu" in base_url:
        print(
            "Deprecated SciHub base URL detected in SENTINEL_API_URL. Overriding to CDSE catalogue."
        )
        base_url = DEFAULT_BASE_URL

    # Auth: get access token
    print("Authenticating to CDSE...")
    token = get_access_token(user, password)

    # Pick ROI
    footprint, feature = select_random_footprint(GEOJSON_PATH)
    feature_id = (
        feature.get("properties", {}).get("id")
        or feature.get("id")
        or feature.get("properties", {}).get("name")
        or "unknown_id"
    )
    print(f"Selected ROI: {feature_id}")

    # Build and execute query
    print(f"Using catalogue base URL: {base_url}")
    print("Querying Sentinel-3 products via OData...")
    odata_filter = build_odata_filter(footprint, DATE_RANGE[0], DATE_RANGE[1])
    result = query_products(token, base_url, odata_filter)
    products: Iterable[Dict[str, Any]] = result.get("value", [])

    count = len(products)
    if count == 0:
        print("No products found for the selected ROI and date range.")
        return

    print(f"Found {count} products. Downloading to: {DOWNLOAD_DIR.resolve()}")
    downloaded: int = 0
    extracted: int = 0
    clipped_total: int = 0

    for p in products:
        pid = p.get("Id") or p.get("id")
        name = p.get("Name") or p.get("name")
        if not pid:
            continue
        try:
            zip_path = download_product(token, str(pid), DOWNLOAD_DIR, filename_hint=name)
            downloaded += 1
            # Determine storage dir and extract
            target_dir = determine_storage_dir(STORE_ROOT_DIR, p)
            _safe_extract_zip(zip_path, target_dir)
            write_metadata(target_dir, p)
            extracted += 1
            # Clip rasters to ROI
            clips_made = clip_rasters_in_dir(target_dir, feature.get("geometry"))
            clipped_total += clips_made
            if clips_made:
                print(f"Clipped {clips_made} raster(s) for: {target_dir}")
            else:
                print(f"No raster clips created (none found or clipping skipped): {target_dir}")
            if not KEEP_ZIP:
                try:
                    zip_path.unlink(missing_ok=True)
                except Exception:
                    pass
            else:
                # Move zip alongside extracted data
                try:
                    (target_dir / Path(zip_path.name)).write_bytes(zip_path.read_bytes())
                    zip_path.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception as e:  # noqa: BLE001
            print(f"Failed to download/extract product {pid}: {e}")

    if downloaded == 0:
        print("Query returned products but none were downloaded.")
    else:
        print(
            f"Completed. Downloaded: {downloaded}, Extracted: {extracted}, Clipped rasters: {clipped_total}.\n"
            f"Store root: {STORE_ROOT_DIR.resolve()}"
        )


if __name__ == "__main__":
    main()