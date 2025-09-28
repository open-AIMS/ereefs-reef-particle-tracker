"""Config-driven GBR1 daily subcube extractor (refactored to spec).

Generates one NetCDF per reef per day at target depth (nearest vertical level).

Directory structure:
    <data_root>/<LABEL_ID>/<YEAR>/<file>.nc
Filename pattern (spec):
    {model_name}{label_id}UV{depthToken}{YYYYMMDD}.nc
  depthToken keeps sign, up to 2 decimals, suffixed with 'm' (e.g. -2.35m, -2m)

Config keys (TOML):
    REQUIRED: ids, years, model_name, depth_m, k_index, opendap_url, data_root
    OPTIONAL: date_periods = ["Jan01-Apr30", "Nov15-Dec31"] (if omitted or empty => full year)
                        buffer_km = <float> (optional horizontal buffer applied around reef polygon when computing ij window; default 0)

This script is now config-only (CLI provides only --config). Week logic removed.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform
import pyproj
import numpy as np
import xarray as xr

from utils import (
    configure_logging,
    sanitize_name,
    load_config,
    expand_date_periods_for_year,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_REEF_SHP = Path(
    "data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp"
)


@dataclass(frozen=True)
class GridArrays:
    lon: np.ndarray
    lat: np.ndarray
    zc: np.ndarray


# ---------------- Grid & selection helpers -----------------

def load_grid(opendap_url: str) -> GridArrays:
    logging.info("Loading grid: %s", opendap_url)
    ds = xr.open_dataset(opendap_url, decode_times=True)
    try:
        lon = ds["longitude"].astype("float64").values
        lat = ds["latitude"].astype("float64").values
        zc = ds["zc"].astype("float64").values if "zc" in ds.variables else np.array([])
    finally:
        ds.close()
    return GridArrays(lon=lon, lat=lat, zc=zc)


def preload_label_polygons(shp_path: Path, requested_ids: List[str]):
    """Prepare reef polygon metadata for all requested LABEL_IDs.

    Returns
    -------
    dict
        { LABEL_ID: { 'reef_name': str, 'polygon': shapely geometry, 'bbox': (minx,miny,maxx,maxy) } }

    This function is intentionally structured in clear phases so failures (e.g.,
    missing IDs, empty geometries) occur *before* any remote OPeNDAP access.
    """
    # --- Phase 1: Load & basic validation ---------------------------------
    if not shp_path.exists():
        raise SystemExit(f"Reef shapefile not found: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if 'LABEL_ID' not in gdf.columns or 'GBR_NAME' not in gdf.columns:
        raise SystemExit('Shapefile missing LABEL_ID or GBR_NAME columns')

    # --- Phase 2: Normalise & deduplicate LABEL_IDs ------------------------
    gdf = gdf.drop_duplicates(subset=['LABEL_ID']).copy()
    gdf['LABEL_ID'] = gdf['LABEL_ID'].astype(str)
    requested_ids = [str(i) for i in requested_ids]
    id_set = set(requested_ids)

    # --- Phase 3: Presence check (fail fast if any missing) ----------------
    present_ids = set(gdf['LABEL_ID'])
    missing = sorted(id_set - present_ids)
    if missing:
        raise SystemExit(f"LABEL_ID(s) not found in reef layer: {', '.join(missing)}")

    # --- Phase 4: Slice to only requested IDs ------------------------------
    subset = gdf[gdf['LABEL_ID'].isin(id_set)].copy()

    # --- Phase 5: Build mapping with geometry & bounding box ---------------
    mapping: Dict[str, Dict[str, object]] = {}
    for _, row in subset.iterrows():  # type: ignore[assignment]
        lid = row['LABEL_ID']
        geom = row.geometry
        if geom is None or geom.is_empty:
            raise SystemExit(f"Geometry empty for LABEL_ID {lid}")
        minx, miny, maxx, maxy = geom.bounds
        mapping[lid] = {
            'reef_name': str(row['GBR_NAME']),
            'polygon': geom,
            'bbox': (float(minx), float(miny), float(maxx), float(maxy)),
        }

    # --- Phase 6: Final integrity check ------------------------------------
    if len(mapping) != len(id_set):
        still_missing = sorted(id_set - set(mapping))
        raise SystemExit(f"Unexpected missing IDs after load: {', '.join(still_missing)}")

    return mapping


# ---- Curvilinear window helpers -------------------------------------------------

def _buffer_polygon_epsg3112(poly: Polygon, buffer_km: float) -> Polygon:
    if buffer_km <= 0:
        return poly
    # Project to EPSG:3112 (Australian Albers) for metre-based buffer
    proj_src = pyproj.CRS("EPSG:4326")
    proj_tgt = pyproj.CRS("EPSG:3112")
    transformer_fwd = pyproj.Transformer.from_crs(proj_src, proj_tgt, always_xy=True).transform
    transformer_inv = pyproj.Transformer.from_crs(proj_tgt, proj_src, always_xy=True).transform
    poly_proj = shp_transform(transformer_fwd, poly)
    buffered = poly_proj.buffer(buffer_km * 1000.0)
    return shp_transform(transformer_inv, buffered)

def ij_window_for_buffered_bbox(grid: GridArrays, poly: Polygon, buffer_km: float) -> tuple[int,int,int,int]:
    buffered = _buffer_polygon_epsg3112(poly, buffer_km)
    minx, miny, maxx, maxy = buffered.bounds
    lon = grid.lon; lat = grid.lat
    mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    finite = np.isfinite(lon) & np.isfinite(lat)
    mask &= finite
    if not mask.any():
        raise ValueError("Buffered bbox does not intersect grid; check inputs or increase buffer")
    js, is_ = np.where(mask)
    return int(js.min()), int(js.max()), int(is_.min()), int(is_.max())


def determine_k_index(target_depth_m: float, available_zc: np.ndarray, fallback_k: int) -> tuple[int, float]:
    if available_zc is None or available_zc.size == 0:
        return fallback_k, float("nan")
    wanted = target_depth_m if target_depth_m <= 0 else -abs(target_depth_m)
    diffs = np.abs(available_zc - wanted)
    k = int(diffs.argmin())
    return k, float(available_zc[k])


def build_dates(year: int, period_specs: List[str] | None) -> List[date]:
    if period_specs:
        return expand_date_periods_for_year(year, period_specs)
    start = date(year, 1, 1); end = date(year, 12, 31)
    out = []
    cur = start
    while cur <= end:
        out.append(cur); cur += timedelta(days=1)
    return out


def depth_token(depth_m: float) -> str:
    s = f"{depth_m:.2f}".rstrip('0').rstrip('.')
    return f"{s}m"


def make_daily_filename(data_root: Path, label_id: str, year: int, model_name: str, depth_tok: str, day: date) -> Path:
    fname = f"{model_name}_{label_id}_UV_{depth_tok}_{day:%Y%m%d}.nc"
    return data_root / label_id / str(year) / fname


def discover_uv_vars(ds: xr.Dataset) -> tuple[str, str]:
    cand_u = [v for v in ds.data_vars if v.lower().startswith("u")]
    cand_v = [v for v in ds.data_vars if v.lower().startswith("v")]
    if not cand_u or not cand_v:
        raise ValueError("Cannot identify U/V variables in dataset")
    return cand_u[0], cand_v[0]


def find_vertical_dim(ds: xr.Dataset, zc_len: int | None) -> str:
    for cand in ["k", "layer", "depth", "zc", "z"]:
        if cand in ds.dims and (zc_len is None or ds.dims[cand] == zc_len):
            return cand
    if zc_len is not None:
        for name, size in ds.dims.items():
            if size == zc_len:
                return name
    raise ValueError("Could not determine vertical dimension name")


# ---------------- Core extraction -----------------

def extract(cfg):  # noqa: C901
    log = logging.getLogger(__name__)
    log.info("Starting extraction: reefs=%d years=%s", len(cfg.ids), cfg.years)


    # Preload and validate all requested reef polygons BEFORE any OPeNDAP access (fail fast)
    reef_meta = preload_label_polygons(DEFAULT_REEF_SHP, [str(i) for i in cfg.ids])
    grid = load_grid(cfg.opendap_url)

    # Precompute grid window (j/i min/max) for each reef using its bbox once.
    for lid, meta in reef_meta.items():
        poly = meta['polygon']  # type: ignore[index]
        try:
            j0,j1,i0,i1 = ij_window_for_buffered_bbox(grid, poly, cfg.buffer_km)
        except ValueError as e:
            raise SystemExit(f"Reef {lid}: {e}") from e
        meta['j_window'] = (j0, j1)  # type: ignore[index]
        meta['i_window'] = (i0, i1)  # type: ignore[index]
        log.info("Reef %s window j:[%d,%d] i:[%d,%d] buffer_km=%.2f km", lid, j0, j1, i0, i1, cfg.buffer_km)
    k_idx, actual_depth = determine_k_index(cfg.depth_m, grid.zc, cfg.k_index)
    log.info("Depth target %.2fm mapped to k_index=%d (model depth %.2fm)", cfg.depth_m, k_idx, actual_depth)

    for label_id in cfg.ids:
        label_id = str(label_id)
        meta = reef_meta[label_id]
        reef_name = meta['reef_name']  # type: ignore[index]
        safe_label = sanitize_name(str(reef_name))  # kept only for metadata
        for year in cfg.years:
            days = build_dates(year, cfg.date_periods)
            year_dir = cfg.data_root / label_id / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = year_dir / "manifest.txt"
            done = set()
            if manifest_path.exists():
                try:
                    done = set(x.strip() for x in manifest_path.read_text().splitlines() if x.strip())
                except Exception:  # noqa: BLE001
                    pass
            depth_tok = depth_token(cfg.depth_m)
            ds = xr.open_dataset(cfg.opendap_url)
            try:
                u_var, v_var = discover_uv_vars(ds)
                vdim = find_vertical_dim(ds, len(grid.zc) if grid.zc is not None else None)
                # Use precomputed bounding box for this reef (fast; already validated)
                (jmin, jmax) = meta['j_window']  # type: ignore[index]
                (imin, imax) = meta['i_window']  # type: ignore[index]
                log.info("Reef %s year %d days=%d window j:%d-%d i:%d-%d", label_id, year, len(days), jmin, jmax, imin, imax)
                updates: List[str] = []
                for day in days:
                    tag = day.strftime('%Y%m%d')
                    out_path = make_daily_filename(cfg.data_root, label_id, year, cfg.model_name, depth_tok, day)
                    tmp_path = out_path.with_suffix('.tmp.nc')
                    if out_path.exists() and not cfg.overwrite:
                        done.add(tag); continue
                    if tag in done and not cfg.overwrite:
                        continue
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    next_day = day + timedelta(days=1)
                    # Use half-open interval [day, next_day) to avoid including the next day's 00:00 which would create a 25th sample
                    try:
                        sel = ds.sel(time=slice(np.datetime64(day), np.datetime64(next_day)))
                        # If the last timestep equals next_day exactly, drop it (floating point tolerant)
                        if 'time' in sel.coords:
                            times = sel['time'].values
                            if times.size > 0:
                                last = np.datetime64(times[-1])
                                if last >= np.datetime64(next_day):
                                    sel = sel.isel(time=slice(0, -1))
                        # If still more than 24 hourly steps, attempt truncation to first 24 (defensive)
                        #if 'time' in sel.coords and sel.dims.get('time', 0) > 24:
                        #    sel = sel.isel(time=slice(0,24))
                    except Exception as e:  # noqa: BLE001
                        log.warning("Time slice failed %s: %s", day, e); continue
                    try:
                        sub = sel[{vdim: k_idx, 'j': slice(jmin, jmax+1), 'i': slice(imin, imax+1)}]
                    except Exception:
                        try:
                            sub = sel.isel({vdim: k_idx, 'j': slice(jmin, jmax+1), 'i': slice(imin, imax+1)})
                        except Exception:
                            log.error("Horizontal slice failed %s %s", label_id, tag); continue
                    try:
                        sub_uv = sub[[u_var, v_var]].squeeze(drop=True)
                    except Exception:
                        sub_uv = sub
                    enc = {v: {"zlib": True, "complevel": 4} for v in sub_uv.data_vars}
                    sub_uv.attrs.update({
                        'model_name': cfg.model_name,
                        'label_id': label_id,
                        'reef_name': reef_name,
                        'safe_label': safe_label,
                        'depth_m': cfg.depth_m,
                        'k_index': k_idx,
                        'actual_model_depth_m': actual_depth,
                        'j_start': jmin,
                        'j_stop': jmax,
                        'i_start': imin,
                        'i_stop': imax,
                        'source_opendap': cfg.opendap_url,
                        'buffer_km': cfg.buffer_km,
                        'hours_in_file': int(sub_uv.dims.get('time', -1)),
                        'extraction_date': date.today().isoformat(),
                        'history': f"created by 03-extract-gbr1-subcubes.py on {date.today().isoformat()}",
                        'note_depth_resolution': 'depth_m mapped to nearest model zc at extraction time',
                    })
                    try:
                        # Write to temporary file first to avoid leaving corrupt partial outputs
                        sub_uv.to_netcdf(tmp_path, format='NETCDF4', encoding=enc)
                        # Atomic-ish rename (on same filesystem) to final path
                        tmp_path.replace(out_path)
                        updates.append(tag)
                        log.info("Wrote %s (via temp)", out_path)
                    except Exception as e:  # noqa: BLE001
                        log.error("Write failed %s: %s", out_path, e)
                        # Clean up temp file if present
                        try:
                            if tmp_path.exists():
                                tmp_path.unlink()
                        except Exception:  # noqa: BLE001
                            pass
                if updates:
                    with manifest_path.open('a', encoding='utf-8') as mf:
                        for t in updates:
                            mf.write(f"{t}\n")
            finally:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try: ds.close()  # type: ignore[has-type]
                    except Exception:  # noqa: BLE001
                        pass
    log.info("Extraction complete")


# ---------------- CLI -----------------

def parse_args():
    p = argparse.ArgumentParser(description="GBR1 daily subcube extractor (config-only)")
    p.add_argument('--config', required=True, help='Path to TOML config')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(Path(args.config))
    configure_logging(cfg.log_file)
    extract(cfg)


if __name__ == '__main__':
    raise SystemExit(main())
