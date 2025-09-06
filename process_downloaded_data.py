#!/usr/bin/env python3
"""
Process downloaded GEE data (offline) for a given ROI folder.

- Reads task_mapping.json inside the ROI folder (e.g., downloads/D-49-49-A/task_mapping.json)
- Unzips the dataset archives under downloaded_zips/
- Merges band-separated products into convenient multi-band images per dataset
- Writes standardized outputs to a temp processing folder for ETL_data_retrieval_module
- Generates processed_mapping.json summarizing outputs

Usage:
  python process_downloaded_data.py \
      --roi_dir downloads/D-49-49-A \
      --output_root temp_processed_data
"""

import os
import re
import json
import argparse
import zipfile
import shutil
from typing import Dict, List, Tuple, Optional

import rasterio

# ----------------------------
# Helpers and configuration
# ----------------------------

S2_BAND_ORDER = ["B4", "B3", "B2", "B8"]
LANDSAT_L1_BAND_ORDER = ["B10", "B11"]
LANDSAT_L2_BAND_ORDER = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "QA_PIXEL"]
ERA5_VAR_ORDER = ["skin_temperature", "temperature_2m"]
ASTER_BAND_ORDER = ["emissivity_band10", "emissivity_band11", "emissivity_band12", "emissivity_band13", "emissivity_band14", "ndvi"]

# Regex patterns for filename parsing
RE_S2_BAND = re.compile(r"^S2_SR_(?P<band>B[2348])_(?P<date>\d{8})_(?P<tile>[A-Z0-9]+)\.tif$")
RE_S2_MERGED = re.compile(r"^S2_SR_(?:merged_)?(?P<date>\d{8})_(?P<tile>[A-Z0-9]+)\.tif$")
RE_LANDSAT = re.compile(r"^L(?P<sat>[89])_(?P<level>L[12])_(?P<band>(?:B10|B11|SR_B[1-7]|QA_PIXEL))_(?P<date>\d{8})_(?P<scene>[^.]+)\.tif$")
RE_ERA5 = re.compile(r"^(?P<var>skin_temperature|temperature_2m)_era5_(?P<dt>\d{8}_\d{6})Z\.tif$")
RE_ASTER = re.compile(r"^ASTER_(?P<var>emissivity_band1[0-4]|ndvi)\.tif$")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def unzip_datasets(task_mapping_path: str, zips_dir: str, extract_root: str) -> Dict[str, str]:
    """
    Unzip all dataset zips referenced in task_mapping.json into per-dataset folders.

    Returns a mapping {dataset_type: extracted_dir}.
    """
    with open(task_mapping_path, 'r') as f:
        mapping = json.load(f)

    # Map each dataset type to its zip path(s); in current design 1 zip per dataset
    dataset_to_zip_paths: Dict[str, List[str]] = {}
    for task_id, info in mapping.get('task_mapping', {}).items():
        dataset = info.get('dataset_type')
        zip_name = info.get('zip_path')
        if not dataset or not zip_name:
            continue
        dataset_to_zip_paths.setdefault(dataset, []).append(os.path.join(zips_dir, zip_name))

    dataset_to_extract_dir: Dict[str, str] = {}

    for dataset, zip_paths in dataset_to_zip_paths.items():
        target_dir = os.path.join(extract_root, dataset)
        ensure_dir(target_dir)
        for zp in zip_paths:
            if not os.path.exists(zp):
                print(f"  > Warning: Zip not found for dataset {dataset}: {zp}")
                continue
            try:
                with zipfile.ZipFile(zp, 'r') as zf:
                    zf.extractall(target_dir)
                print(f"  > Extracted {os.path.basename(zp)} -> {target_dir}")
            except Exception as e:
                print(f"  > Error extracting {zp}: {e}")
        dataset_to_extract_dir[dataset] = target_dir

    return dataset_to_extract_dir


def merge_to_multiband(output_path: str, band_files_in_order: List[str]) -> bool:
    """
    Merge given single-band tif files (already ordered) into a single multi-band tif.
    Returns True on success.
    """
    try:
        if not band_files_in_order:
            return False
        # Read profile from first file
        with rasterio.open(band_files_in_order[0]) as src0:
            profile = src0.profile.copy()
        profile.update(count=len(band_files_in_order), driver='GTiff', compress='lzw')
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i, band_file in enumerate(band_files_in_order, start=1):
                with rasterio.open(band_file) as src:
                    dst.write(src.read(1), i)
        return True
    except Exception as e:
        print(f"  > Merge failed ({os.path.basename(output_path)}): {e}")
        return False


def list_all_tifs(root_dir: str) -> List[str]:
    tifs = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith('.tif'):
                tifs.append(os.path.join(base, fn))
    return tifs


def process_sentinel2(extracted_dir: str, out_dir: str, processed: List[Dict]) -> None:
    ensure_dir(out_dir)
    all_tifs = list_all_tifs(extracted_dir)
    # First handle already merged products
    seen_keys = set()

    for tif in all_tifs:
        m = RE_S2_MERGED.match(os.path.basename(tif))
        if m:
            date = m.group('date')
            tile = m.group('tile')
            key = (date, tile)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            out_name = f"S2_SR_{date}_{tile}.tif"
            out_path = os.path.join(out_dir, out_name)
            # Copy/rename
            try:
                if os.path.abspath(tif) != os.path.abspath(out_path):
                    shutil.copy2(tif, out_path)
                processed.append({
                    'dataset': 'sentinel2',
                    'date': date,
                    'tile': tile,
                    'output': out_name,
                    'bands': 'merged'
                })
                print(f"  S2 merged: {out_name}")
            except Exception as e:
                print(f"  > Copy failed for {tif} -> {out_path}: {e}")

    # Then group band-separated scenes
    groups: Dict[Tuple[str, str], Dict[str, str]] = {}
    for tif in all_tifs:
        mb = RE_S2_BAND.match(os.path.basename(tif))
        if not mb:
            continue
        band = mb.group('band')
        date = mb.group('date')
        tile = mb.group('tile')
        groups.setdefault((date, tile), {})[band] = tif

    for (date, tile), band_map in groups.items():
        out_name = f"S2_SR_{date}_{tile}.tif"
        out_path = os.path.join(out_dir, out_name)
        if os.path.exists(out_path):
            # Already written by merged case above
            continue
        ordered_files = [band_map[b] for b in S2_BAND_ORDER if b in band_map]
        if not ordered_files:
            continue
        if merge_to_multiband(out_path, ordered_files):
            processed.append({
                'dataset': 'sentinel2',
                'date': date,
                'tile': tile,
                'output': out_name,
                'bands': [b for b in S2_BAND_ORDER if b in band_map]
            })
            print(f"  S2 merged bands: {out_name}")


def process_landsat(extracted_dir: str, out_dir: str, level: str, processed: List[Dict]) -> None:
    ensure_dir(out_dir)
    all_tifs = list_all_tifs(extracted_dir)
    groups: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for tif in all_tifs:
        m = RE_LANDSAT.match(os.path.basename(tif))
        if not m:
            continue
        if m.group('level') != level:
            continue
        sat = m.group('sat')
        band = m.group('band')
        date = m.group('date')
        scene = m.group('scene')
        key = (sat, level, date, scene)
        groups.setdefault(key, {})[band] = tif

    order = LANDSAT_L1_BAND_ORDER if level == 'L1' else LANDSAT_L2_BAND_ORDER

    for (sat, lv, date, scene), band_map in groups.items():
        out_name = f"L{sat}_{lv}_{date}_{scene}.tif"
        out_path = os.path.join(out_dir, out_name)
        ordered_files = [band_map[b] for b in order if b in band_map]
        if not ordered_files:
            continue
        if merge_to_multiband(out_path, ordered_files):
            processed.append({
                'dataset': f'landsat{sat.lower()}_{lv.lower()}',
                'satellite': sat,
                'level': lv,
                'date': date,
                'scene_id': scene,
                'output': out_name,
                'bands': [b for b in order if b in band_map]
            })
            print(f"  Landsat {sat} {lv} merged: {out_name}")


def process_era5(extracted_dir: str, out_dir: str, processed: List[Dict]) -> None:
    ensure_dir(out_dir)
    all_tifs = list_all_tifs(extracted_dir)
    groups: Dict[str, Dict[str, str]] = {}
    for tif in all_tifs:
        m = RE_ERA5.match(os.path.basename(tif))
        if not m:
            continue
        var = m.group('var')
        dt = m.group('dt')
        groups.setdefault(dt, {})[var] = tif

    for dt, var_map in groups.items():
        out_name = f"ERA5_{dt}Z.tif"
        out_path = os.path.join(out_dir, out_name)
        ordered_files = [var_map[v] for v in ERA5_VAR_ORDER if v in var_map]
        if not ordered_files:
            continue
        if merge_to_multiband(out_path, ordered_files):
            processed.append({
                'dataset': 'era5',
                'datetime': dt + 'Z',
                'output': out_name,
                'variables': [v for v in ERA5_VAR_ORDER if v in var_map]
            })
            print(f"  ERA5 merged: {out_name}")


def process_aster(extracted_dir: str, out_dir: str, processed: List[Dict]) -> None:
    ensure_dir(out_dir)
    all_tifs = list_all_tifs(extracted_dir)
    band_map: Dict[str, str] = {}
    for tif in all_tifs:
        m = RE_ASTER.match(os.path.basename(tif))
        if not m:
            continue
        var = m.group('var')
        band_map[var] = tif

    ordered_files = [band_map[b] for b in ASTER_BAND_ORDER if b in band_map]
    if not ordered_files:
        return
    out_name = "ASTER_emissivity_ndvi.tif" if "ndvi" in band_map else "ASTER_emissivity.tif"
    out_path = os.path.join(out_dir, out_name)
    if merge_to_multiband(out_path, ordered_files):
        processed.append({
            'dataset': 'aster',
            'output': out_name,
            'bands': [b for b in ASTER_BAND_ORDER if b in band_map]
        })
        print(f"  ASTER merged: {out_name}")


def process_roi(roi_dir: str, output_root: str) -> str:
    """
    Main processing pipeline for a given ROI downloads directory.

    Returns path to processed_mapping.json
    """
    task_mapping_path = os.path.join(roi_dir, 'task_mapping.json')
    zips_dir = os.path.join(roi_dir, 'downloaded_zips')
    if not os.path.exists(task_mapping_path):
        raise FileNotFoundError(f"task_mapping.json not found: {task_mapping_path}")
    if not os.path.exists(zips_dir):
        raise FileNotFoundError(f"downloaded_zips dir not found: {zips_dir}")

    roi_name = os.path.basename(os.path.normpath(roi_dir))
    tmp_extract_root = os.path.join(roi_dir, '_tmp_extracted')
    if os.path.exists(tmp_extract_root):
        shutil.rmtree(tmp_extract_root)
    ensure_dir(tmp_extract_root)

    # Unzip
    print("Unzipping datasets...")
    dataset_to_dir = unzip_datasets(task_mapping_path, zips_dir, tmp_extract_root)

    # Prepare outputs
    roi_out_root = os.path.join(output_root, roi_name)
    ensure_dir(roi_out_root)

    processed_entries: List[Dict] = []

    # Process datasets if available
    if 'sentinel2' in dataset_to_dir:
        s2_out = os.path.join(roi_out_root, 'sentinel2')
        process_sentinel2(dataset_to_dir['sentinel2'], s2_out, processed_entries)

    # Landsat datasets may be split by level and sat but extracted into one dir
    if any(ds.startswith('landsat') for ds in dataset_to_dir.keys()):
        landsat_dir = None
        # Merge all landsat extractions into a common temp dir for grouping
        for k, v in dataset_to_dir.items():
            if k.startswith('landsat'):
                landsat_dir = v if landsat_dir is None else landsat_dir
                if v != landsat_dir:
                    # Copy files into common dir
                    for tif in list_all_tifs(v):
                        try:
                            shutil.copy2(tif, os.path.join(landsat_dir, os.path.basename(tif)))
                        except Exception:
                            pass
        if landsat_dir:
            l1_out = os.path.join(roi_out_root, 'landsat_l1')
            l2_out = os.path.join(roi_out_root, 'landsat_l2')
            process_landsat(landsat_dir, l1_out, 'L1', processed_entries)
            process_landsat(landsat_dir, l2_out, 'L2', processed_entries)

    if 'era5' in dataset_to_dir:
        era5_out = os.path.join(roi_out_root, 'era5')
        process_era5(dataset_to_dir['era5'], era5_out, processed_entries)

    if 'aster' in dataset_to_dir:
        aster_out = os.path.join(roi_out_root, 'aster')
        process_aster(dataset_to_dir['aster'], aster_out, processed_entries)

    # Save processed mapping
    processed_mapping = {
        'roi': roi_name,
        'output_root': roi_out_root,
        'total_outputs': len(processed_entries),
        'entries': processed_entries
    }
    processed_mapping_path = os.path.join(roi_out_root, 'processed_mapping.json')
    with open(processed_mapping_path, 'w') as f:
        json.dump(processed_mapping, f, indent=2)
    print(f"Processed mapping saved: {processed_mapping_path}")

    # Cleanup
    try:
        shutil.rmtree(tmp_extract_root)
    except Exception:
        print(f"  > Warning: Failed to remove temp dir: {tmp_extract_root}")

    return processed_mapping_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process offline downloaded GEE datasets for an ROI.")
    parser.add_argument('--roi_dir', required=True, help='Path to ROI downloads directory, e.g., downloads/D-49-49-A')
    parser.add_argument('--output_root', default='temp_processed_data', help='Root temp output directory (default: temp_processed_data)')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        ensure_dir(args.output_root)
        process_roi(args.roi_dir, args.output_root)
        print("\nAll done.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
