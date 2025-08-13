import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openeo

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("s3_slstr_l2_lst_openeo")


# -----------------------------------------------------------------------------
# GeoJSON grid utilities (modeled after data_retrival_module/main.py behavior)
# -----------------------------------------------------------------------------

def find_grid_feature(phien_hieu: str, grid_file_path: str) -> Optional[dict]:
    """Find a feature by 'PhienHieu' property in a GeoJSON file.

    Args:
        phien_hieu: Identifier to match in feature.properties.PhienHieu
        grid_file_path: Path to the GeoJSON file

    Returns:
        The matching GeoJSON feature dict or None if not found.
    """
    try:
        with open(grid_file_path, "r") as f:
            grid_data = json.load(f)
        for feature in grid_data.get("features", []):
            if feature.get("properties", {}).get("PhienHieu") == phien_hieu:
                return feature
        logger.error("Failed to find feature with PhienHieu '%s' in %s", phien_hieu, grid_file_path)
        return None
    except FileNotFoundError:
        logger.critical("Grid file not found at: %s", grid_file_path)
        return None
    except json.JSONDecodeError:
        logger.critical("Failed to read/parse grid file: %s", grid_file_path)
        return None


# -----------------------------------------------------------------------------
# openEO helpers
# -----------------------------------------------------------------------------

_DEFAULT_BACKEND = "https://openeo.dataspace.copernicus.eu"
# We try a few likely collection IDs in case of minor naming differences between deployments.
_POSSIBLE_COLLECTION_IDS = [
    "SENTINEL3_SLSTR_L2_LST",
    "SENTINEL-3_SLSTR_L2_LST",
    "S3_SLSTR_L2_LST",
    "SENTINEL3_SL_2_LST",
]


def connect_and_authenticate(
    backend_url: str,
    oidc_client_id: Optional[str],
    oidc_refresh_token: Optional[str],
    basic_username: Optional[str],
    basic_password: Optional[str],
    interactive_login: bool = False,
    oidc_provider_id: Optional[str] = None,
    oidc_redirect_uri: Optional[str] = None,
) -> openeo.Connection:
    """Connect to openEO backend and authenticate.

    Priority: interactive (if requested) -> OIDC refresh token -> Basic -> unauthenticated.
    """
    logger.info("Connecting to openEO backend: %s", backend_url)
    conn = openeo.connect(backend_url)

    # Interactive OIDC login if requested: prefer browser auth-code flow
    if interactive_login:
        try:
            logger.info(
                "Starting interactive OIDC login (auth-code) client_id=%s provider=%s",
                oidc_client_id or "cdse-public",
                oidc_provider_id or "CDSE",
            )
            # Try with provider_id & redirect_uri if supported by client version
            conn.authenticate_oidc(
                client_id=oidc_client_id or "cdse-public",
                provider_id=oidc_provider_id or "CDSE",
                redirect_uri=oidc_redirect_uri or "http://localhost:8080/",
            )
            logger.info("Interactive OIDC (auth-code) login successful.")
            return conn
        except TypeError as exc:
            # Fallback for older clients that don't support some kwargs
            logger.warning("Interactive OIDC kwargs not supported (%s). Retrying with minimal args.", exc)
            try:
                conn.authenticate_oidc(client_id=oidc_client_id or "cdse-public")
                logger.info("Interactive OIDC login successful (minimal args).")
                return conn
            except Exception as exc2:  # noqa: BLE001
                logger.warning("Interactive OIDC (minimal) failed: %s", exc2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Interactive OIDC login failed: %s", exc)

    # Try OIDC refresh token if provided
    if oidc_refresh_token:
        try:
            logger.info("Authenticating via OIDC refresh token (client_id=%s)", oidc_client_id or "cdse-public")
            conn.authenticate_oidc(
                client_id=oidc_client_id or "cdse-public",
                refresh_token=oidc_refresh_token,
                provider_id=oidc_provider_id or None,
            )
            logger.info("OIDC authentication successful.")
            return conn
        except Exception as exc:  # noqa: BLE001
            logger.warning("OIDC refresh-token authentication failed: %s", exc)

    # Fallback: basic auth
    if basic_username and basic_password:
        try:
            logger.info("Authenticating via basic auth (username only logged)")
            conn.authenticate_basic(basic_username, basic_password)
            logger.info("Basic auth successful.")
            return conn
        except Exception as exc:  # noqa: BLE001
            logger.warning("Basic authentication failed: %s", exc)

    logger.info("Proceeding without authentication. If the backend requires auth, requests may fail.")
    return conn


def resolve_collection_id(conn: openeo.Connection, preferred: Optional[str]) -> str:
    """Pick an available collection ID for SLSTR L2 LST.

    Checks the `preferred` collection first, then tries a built-in candidate list.
    Raises RuntimeError if none is available.
    """
    collections = conn.list_collections()
    available_ids = {c["id"] for c in collections} if isinstance(collections, list) else set()

    if preferred:
        if preferred in available_ids:
            logger.info("Using provided collection id: %s", preferred)
            return preferred
        logger.warning("Provided collection id '%s' not found on backend. Trying fallbacks.", preferred)

    for cid in _POSSIBLE_COLLECTION_IDS:
        if cid in available_ids:
            logger.info("Resolved collection id to: %s", cid)
            return cid

    # As a last resort: try a substring match
    for cid in available_ids:
        if "SLSTR" in cid and "L2" in cid and ("LST" in cid or "_LST" in cid):
            logger.info("Heuristically matched collection id: %s", cid)
            return cid

    raise RuntimeError(
        "Could not resolve a Sentinel-3 SLSTR L2 LST collection id on the backend. "
        "Pass --collection_id explicitly and/or verify backend supports SLSTR L2 LST."
    )


def detect_lst_band(conn: openeo.Connection, collection_id: str) -> Optional[str]:
    """Try to detect the LST band name from collection metadata.

    Returns a band name (e.g., 'LST') or None if not found.
    """
    try:
        meta = conn.describe_collection(collection_id)
        bands = meta.get("summaries", {}).get("bands") or meta.get("bands")
        if isinstance(bands, list):
            # bands entries may be dicts with 'name' and 'description'
            candidates = []
            for b in bands:
                if isinstance(b, dict):
                    name = b.get("name") or b.get("id")
                    desc = (b.get("description") or "").lower()
                else:
                    name = str(b)
                    desc = ""
                lname = (name or "").lower()
                if "lst" in lname or "land surface temperature" in desc:
                    candidates.append(name)
            if candidates:
                # Prefer exact 'LST' name if present
                for c in candidates:
                    if c.upper() == "LST":
                        return c
                return candidates[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not inspect collection bands: %s", exc)
    return None


# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------

def daterange(start_date: datetime, end_date: datetime) -> List[datetime]:
    days = []
    d = start_date
    while d <= end_date:
        days.append(d)
        d = d + timedelta(days=1)
    return days


def ensure_output_schema(base: Path, roi: str) -> Tuple[Path, Path]:
    """Return (root_dir, daily_dir), creating them if needed.

    root_dir: {base}/{roi}/s3_slstr_lst
    daily_dir: created per-year under root_dir when saving
    """
    root_dir = base / roi / "s3_slstr_lst"
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir, root_dir


def build_spatial_extent(geometry: dict) -> Dict:
    """Convert a Polygon geometry to openEO spatial_extent dict (WGS84 CRS)."""
    # Assume coordinates are in lon/lat (WGS84) as is typical for GeoJSON
    # openEO supports GeoJSON-like dict for spatial_extent
    return {
        "type": geometry.get("type", "Polygon"),
        "coordinates": geometry.get("coordinates"),
        "crs": "EPSG:4326",
    }


def fetch_daily_lst(
    conn: openeo.Connection,
    collection_id: str,
    lst_band: Optional[str],
    roi_geom: dict,
    day_start: datetime,
    out_tif: Path,
    to_celsius: bool,
    job_options: Optional[Dict],
) -> bool:
    """Fetch daily LST for a given day (UTC) and save as a single GeoTIFF.

    Returns True if a new file was written, False if skipped.
    """
    if out_tif.exists():
        logger.info("Exists, skipping: %s", out_tif)
        return False

    day_end = day_start + timedelta(days=1)

    spatial_extent = build_spatial_extent(roi_geom)

    # Load cube
    try:
        cube = conn.load_collection(
            collection_id=collection_id,
            spatial_extent=spatial_extent,
            temporal_extent=[day_start.strftime("%Y-%m-%d"), day_end.strftime("%Y-%m-%d")],
            bands=[lst_band] if lst_band else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("load_collection failed for %s: %s", day_start.strftime("%Y-%m-%d"), exc)
        return False

    # If multiple granules within the day: aggregate to daily mean
    cube = cube.aggregate_temporal_period("day", reducer="mean")

    # Optionally convert K -> C
    if to_celsius:
        cube = cube - 273.15

    # Save and execute as a small batch job to a temporary folder
    result = cube.save_result(format="GTiff", options={"tiled": True, "compress": "deflate"})

    try:
        job = conn.create_job(result)
        if job_options:
            job.start_and_wait(**job_options)
        else:
            job.start_and_wait()
        assets = job.get_results().get_assets()
        # Download first GeoTIFF asset to out_tif
        geotiff_assets = [a for a in assets if a.get("type", "").lower().endswith("geotiff") or a.get("href", "").lower().endswith(".tif")]
        if not geotiff_assets:
            # Fallback: download all; pick first .tif file
            tmp_dir = out_tif.parent / f"tmp_{day_start.strftime('%Y%m%d') }"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            job.get_results().download_files(str(tmp_dir))
            tifs = sorted(tmp_dir.glob("*.tif"))
            if not tifs:
                logger.error("No GeoTIFFs found in job results for %s", day_start.strftime("%Y-%m-%d"))
                return False
            tifs[0].rename(out_tif)
            # Clean up temp dir
            for p in tmp_dir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            try:
                tmp_dir.rmdir()
            except Exception:
                pass
        else:
            # Directly download selected asset
            href = geotiff_assets[0]["href"]
            job.get_results().download_file(href, str(out_tif))
        logger.info("Saved: %s", out_tif)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Batch execution failed for %s: %s", day_start.strftime("%Y-%m-%d"), exc)
        return False
    finally:
        try:
            job.delete()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-3 SLSTR L2 LST via openEO for a ROI identified by PhienHieu and save daily GeoTIFFs.",
    )
    parser.add_argument("--roi_name", type=str, required=True, help="The 'PhienHieu' identifier of the grid to process.")
    parser.add_argument("--start_date", type=str, required=True, help="Start date (YYYY-MM-DD) inclusive.")
    parser.add_argument("--end_date", type=str, required=True, help="End date (YYYY-MM-DD) inclusive.")
    parser.add_argument(
        "--grid_file",
        type=str,
        default="data/Grid_50K_MatchedDates.geojson",
        help="Path to the GeoJSON grid file.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="/mnt/hdd12tb/code/nhatvm/DELAG_main/data/retrieved_data",
        help="Base output folder.",
    )
    # openEO backend/auth
    parser.add_argument("--backend_url", type=str, default=_DEFAULT_BACKEND, help="openEO backend URL.")
    parser.add_argument("--collection_id", type=str, default=None, help="Explicit collection id for SLSTR L2 LST.")
    parser.add_argument("--oidc_client_id", type=str, default="cdse-public", help="OIDC client id.")
    parser.add_argument("--oidc_refresh_token", type=str, default=None, help="OIDC refresh token.")
    parser.add_argument("--basic_username", type=str, default=None, help="Basic auth username (fallback).")
    parser.add_argument("--basic_password", type=str, default=None, help="Basic auth password (fallback).")
    parser.add_argument("--interactive_login",default=True, help="Start interactive OIDC login flow (browser auth code).")
    parser.add_argument("--oidc_provider_id", type=str, default="CDSE", help="OIDC provider id (e.g., CDSE).")
    parser.add_argument(
        "--oidc_redirect_uri",
        type=str,
        default="http://localhost:8080/",
        help="Redirect URI for interactive login (must match client config if customized).",
    )
    # Behavior
    parser.add_argument("--to_celsius", action="store_true", help="Convert output from Kelvin to Celsius (best-effort).")
    parser.add_argument(
        "--output_units",
        type=str,
        default="kelvin",
        choices=["kelvin", "celsius"],
        help="Units to write to disk. If 'celsius', convert K→°C. Defaults to 'kelvin'.",
    )
    parser.add_argument("--max_concurrent_jobs", type=int, default=1, help="Max parallel days to process (1 = serial).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Parse dates
    try:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"Invalid date format: {exc}") from exc
    if end_dt < start_dt:
        raise SystemExit("end_date must be >= start_date")

    # Read grid and select ROI
    feature = find_grid_feature(args.roi_name, args.grid_file)
    if not feature:
        raise SystemExit(1)
    roi_name = feature["properties"]["PhienHieu"]
    roi_geom = feature["geometry"]

    # Connect/authenticate to openEO
    conn = connect_and_authenticate(
        backend_url=args.backend_url,
        oidc_client_id=args.oidc_client_id,
        oidc_refresh_token=args.oidc_refresh_token or os.environ.get("OPENEO_OIDC_REFRESH_TOKEN"),
        basic_username=args.basic_username or os.environ.get("OPENEO_BASIC_USERNAME"),
        basic_password=args.basic_password or os.environ.get("OPENEO_BASIC_PASSWORD"),
        interactive_login=args.interactive_login,
        oidc_provider_id=args.oidc_provider_id,
        oidc_redirect_uri=args.oidc_redirect_uri,
    )

    # Resolve collection and band
    collection_id = resolve_collection_id(conn, args.collection_id)
    lst_band = detect_lst_band(conn, collection_id)
    if lst_band:
        logger.info("Detected LST band: %s", lst_band)
    else:
        logger.info("Proceeding without explicit band selection (using collection default).")

    # Prepare output schema
    base_out = Path(args.output_folder)
    root_dir, _ = ensure_output_schema(base_out, roi_name)

    # Save run metadata
    run_meta: Dict[str, object] = {
        "roi": roi_name,
        "source": "Sentinel-3 SLSTR L2 LST via openEO",
        "backend_url": args.backend_url,
        "collection_id": collection_id,
        "date_range": {"start": args.start_date, "end": args.end_date},
        "units": "Celsius" if args.output_units == "celsius" or args.to_celsius else "Kelvin",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "grid_file": args.grid_file,
        "auth_mode": (
            "interactive_oidc" if args.interactive_login else (
                "oidc_refresh_token" if (args.oidc_refresh_token or os.environ.get("OPENEO_OIDC_REFRESH_TOKEN")) else (
                    "basic" if (args.basic_username or os.environ.get("OPENEO_BASIC_USERNAME")) else "none"
                )
            )
        ),
        "oidc_provider_id": args.oidc_provider_id,
        "oidc_redirect_uri": args.oidc_redirect_uri,
    }
    with open(root_dir / "metadata.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    # Job options (tune if needed)
    job_options = {"connection_retry": 3}

    # Process per-day (optionally parallel). Keep serial by default for safety.
    days = daterange(start_dt, end_dt)

    # Simple serial execution
    for d in days:
        year_dir = root_dir / f"{d.year:04d}"
        year_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"S3_SLSTR_L2_LST_{roi_name}_{d.strftime('%Y-%m-%d')}.tif"
        out_path = year_dir / out_name
        to_c = bool(args.to_celsius or args.output_units == "celsius")
        fetch_daily_lst(
            conn=conn,
            collection_id=collection_id,
            lst_band=lst_band,
            roi_geom=roi_geom,
            day_start=d,
            out_tif=out_path,
            to_celsius=to_c,
            job_options=job_options,
        )

    logger.info("Done. Output at: %s", root_dir)


if __name__ == "__main__":
    main() 