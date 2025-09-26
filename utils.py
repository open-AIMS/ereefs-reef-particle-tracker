"""
utils.py - Shared helper functions for the eReefs particle / extraction workflow.

LLM PRIMER / QUICK ORIENTATION
================================
Purpose: This single file centralizes every cross-cutting helper so large language models (and humans) can reason about the codebase without chasing many tiny modules. Import from here instead of redefining helpers in scripts.

WHEN TO IMPORT WHAT (Cheat Sheet)
---------------------------------
Logging:
    configure_logging(log_file=None, level=INFO) -> None
            Call once at the start of a CLI script to standardize log formatting and optional file logging.

Naming & Paths:
    sanitize_name(name) -> str
            Convert a reef / human name to a safe token used in filenames.
    make_output_stem(safe_name, label_id, k_index, year, week_tag) -> str
            Core stem used for ALL particle-related derivative files; keep naming consistent.
    particles_zarr_path(root, label_id, year, safe_name, k_index, week_tag) -> Path
            Construct canonical Zarr output directory path written by simulation script (04).
    shapefile_path(root, stem, suffix='track') -> Path
            Build shapefile export path (used by converter script 04b).
    png_path_from_stem(root, stem, suffix='track') -> Path
            Derive PNG path for plots (script 05) based on the common stem.

Year / Week Utilities:
    normalize_years(list[int]) -> list[int]
            Expand a 1- or 2-length list (range endpoints) or deduplicate many years into a sorted list.
    normalize_weeks(list[int]|None) -> list[int]|None
            Same expansion semantics for ordinal weeks (1 = Jan1-Jan7). Returns None if input empty/None.
    make_week_tag(weeks|None) -> str
            Turn a list of weeks into naming token: 'all', 'wNN', or 'wNN-wMM'. Use for file names not for logic.
    ordinal_week_for_date(date) -> int
            Convert a Python date into the ordinal week number used by the pipeline (7-day fixed blocks).
    expand_weeks_to_dates(year, weeks) -> list[date]
            Produce actual calendar dates (limited to same year) for the given ordinal week numbers.

Reef Metadata:
    load_reef_layer(path) -> GeoDataFrame
            Lazy-load heavy geopandas dependency; validates required columns.
    build_label_name_map(gdf) -> dict[label_id -> GBR_NAME]
            Helper to quickly map LABEL_ID to reef display name.

Trajectory Loading & Time Helpers:
    Trajectory dataclass(lon, lat, time)
            Simple container used across scripts for uniform attribute access.
    load_parcels_zarr(store, assume_dt_hours=1.0) -> Trajectory
            Robust reader for Parcels output stored as Zarr (single or (traj, obs) forms). Reconstructs or extends time sequence if missing or underspecified and always returns UTC tz-aware timestamps.
    ensure_utc(pd.Timestamp) -> pd.Timestamp
            Normalize any timestamp to UTC (tz-localize or convert).
    reconstruct_times(base_ts, count, dt_hours) -> list[pd.Timestamp]
            Generate evenly spaced UTC timestamps, mainly used internally when time reconstruction needed.

Discovery:
    discover_week_tags(dir, safe_name, label_id, year, k_index) -> list[str]
            Scan a year directory for week-tagged particle Zarr outputs, returning tokens like 'w03' or 'all'.
    auto_detect_single_week(requested_tag, candidates) -> str
            If multiple candidates exist and requested_tag == 'all', choose best available (first or 'all').

Design Notes:
    - Single file for simplicity & LLM discoverability.
    - Heavy / optional deps (geopandas, xarray) are imported lazily inside functions.
    - All times returned from trajectory loader are UTC tz-aware.
    - Functions have minimal side effects (only logging).
    - Scripts 03/04/04b/05 should import these to avoid drift / duplication.

Patterns / Naming Contract:
    Filename stem = {safe}_{LABEL_ID}_k{k}_{YEAR}{week_tag}
    Zarr directory = stem + '_particles.zarr'
    Derived exports (PNG/shape) append '_track.(png|shp)' using the same stem.

Extensibility Guidance (for future contributors / LLMs):
    - When adding a new artifact type (e.g., CSV summary), derive its path from make_output_stem to remain discoverable.
    - Prefer adding new small helpers here rather than re-implementing ad-hoc logic in scripts.
    - Keep parameter names consistent (label_id, safe_name, week_tag) to reduce confusion.

Error Philosophy:
    - Path builders never hit the filesystem (pure functions); callers decide to mkdir.
    - Loaders raise standard exceptions (FileNotFoundError, ValueError). CLI layers catch & log.
    - Discovery returns an empty list instead of raising; caller decides fallback.

Examples (illustrative, not executed here):
    from utils import particles_zarr_path, sanitize_name, make_week_tag
    safe = sanitize_name('Davies Reef')
    tag = make_week_tag([3,4,5])  # -> 'w03-w05'
    zpath = particles_zarr_path(Path('working/02'), '18-096', 2016, safe, 40, tag)
    # working/02/18-096/2016/Davies_Reef_18-096_k40_2016w03-w05_particles.zarr

This primer is intentionally verbose to aid LLM reasoning: do not shorten without retaining the above semantic map.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
import logging
import sys
import numpy as np
import pandas as pd
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------------------------

def configure_logging(log_file: Path | None = None, level: int = logging.INFO, force: bool = False) -> None:
    """Configure root logger once (optionally force reset)."""
    root = logging.getLogger()
    if root.handlers and not force:
        return
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    root.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(fmt)
        root.addHandler(fh)

# ---------------------------------------------------------------------------
# 2. Naming & Paths
# ---------------------------------------------------------------------------

def sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name.replace(" ", "_"))

def make_output_stem(safe_name: str, label_id: str, k_index: int, year: int, week_tag: str) -> str:
    return f"{safe_name}_{label_id}_k{k_index}_{year}{week_tag}"

def particles_zarr_path(root: Path, label_id: str, year: int, safe_name: str, k_index: int, week_tag: str) -> Path:
    return root / label_id / str(year) / f"{make_output_stem(safe_name, label_id, k_index, year, week_tag)}_particles.zarr"

def shapefile_path(root: Path, stem: str, suffix: str = 'track') -> Path:
    return root / f"{stem}_{suffix}.shp"

def png_path_from_stem(root: Path, stem: str, suffix: str = 'track') -> Path:
    return root / f"{stem}_{suffix}.png"

# ---------------------------------------------------------------------------
# 3. Year / Week Utilities
# ---------------------------------------------------------------------------

def normalize_years(vals: List[int]) -> List[int]:
    if len(vals) == 1:
        return vals
    if len(vals) == 2:
        a, b = vals
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return sorted(set(vals))

def normalize_weeks(vals: List[int] | None) -> List[int] | None:
    if not vals:
        return None
    if len(vals) == 1:
        weeks = vals
    elif len(vals) == 2:
        a, b = vals
        if a > b:
            a, b = b, a
        weeks = list(range(a, b + 1))
    else:
        weeks = sorted(set(vals))
    for w in weeks:
        if w < 1:
            raise ValueError(f"Invalid week number {w}; must be >=1")
    return weeks

def make_week_tag(weeks: List[int] | None) -> str:
    if not weeks:
        return "all"
    uniq = sorted(set(weeks))
    return f"w{uniq[0]:02d}" if len(uniq) == 1 else f"w{uniq[0]:02d}-{uniq[-1]:02d}"

def ordinal_week_for_date(d: date) -> int:
    return ((d.timetuple().tm_yday - 1) // 7) + 1

def expand_weeks_to_dates(year: int, weeks: List[int]) -> List[date]:
    days = []
    for w in sorted(set(weeks)):
        start = date(year, 1, 1) + timedelta(days=7 * (w - 1))
        for off in range(7):
            cur = start + timedelta(days=off)
            if cur.year != year:
                break
            days.append(cur)
    return sorted(set(days))

# ---------------------------------------------------------------------------
# 4. Reef Metadata
# ---------------------------------------------------------------------------

def load_reef_layer(path: Path):  # return type left generic to avoid hard import at top
    import geopandas as gpd  # local import (heavy dependency)
    if not path.exists():
        raise FileNotFoundError(f"Reef layer not found: {path}")
    gdf = gpd.read_file(path)
    if 'LABEL_ID' not in gdf.columns or 'GBR_NAME' not in gdf.columns:
        raise ValueError('Reef layer missing LABEL_ID or GBR_NAME columns')
    return gdf

def build_label_name_map(gdf) -> dict[str, str]:  # noqa: ANN001
    return gdf.drop_duplicates(subset=['LABEL_ID']).set_index('LABEL_ID')['GBR_NAME'].to_dict()

# ---------------------------------------------------------------------------
# 5. Trajectory Loading
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    lon: np.ndarray
    lat: np.ndarray
    time: pd.DatetimeIndex

def ensure_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize('UTC')
    return ts.tz_convert('UTC')

def reconstruct_times(base: pd.Timestamp, count: int, dt_hours: float) -> list[pd.Timestamp]:
    base = ensure_utc(base)
    return [base + pd.Timedelta(hours=dt_hours * i) for i in range(count)]

def load_parcels_zarr(store: Path, assume_dt_hours: float = 1.0) -> Trajectory:
    import xarray as xr  # local import
    if not store.exists():
        raise FileNotFoundError(f"Zarr store not found: {store}")
    ds = xr.open_zarr(str(store))
    try:
        if 'lon' not in ds or 'lat' not in ds:
            raise ValueError('Missing lon/lat in trajectory dataset')
        lon = np.asarray(ds['lon'].values)
        lat = np.asarray(ds['lat'].values)
        # Flatten (traj, obs)
        if lon.ndim > 1:
            lon = lon.reshape(-1)
        if lat.ndim > 1:
            lat = lat.reshape(-1)
        if lon.size != lat.size:
            raise ValueError(f'Lon/lat size mismatch {lon.size} vs {lat.size}')
        if 'time' in ds.variables:
            raw_time = np.asarray(ds['time'].values)
            if raw_time.ndim > 1:
                raw_time = raw_time.reshape(-1)
            try:
                times = pd.to_datetime(raw_time).to_pydatetime().tolist()  # type: ignore[arg-type]
            except Exception:
                times = []
        else:
            times = []
    finally:
        ds.close()
    # Reconstruct when needed
    if not times:
        base = pd.Timestamp.utcnow().tz_localize('UTC')
        times = reconstruct_times(base, lon.size, assume_dt_hours)
    elif len(times) == 1 and lon.size > 1:
        times = reconstruct_times(pd.Timestamp(times[0]), lon.size, assume_dt_hours)
    # Normalize
    norm = [ensure_utc(pd.Timestamp(t)) for t in times]
    if len(norm) != lon.size:
        # Pad or truncate
        if len(norm) > lon.size:
            norm = norm[:lon.size]
        else:
            last = norm[-1]
            extra = reconstruct_times(last + pd.Timedelta(hours=assume_dt_hours), lon.size - len(norm), assume_dt_hours)
            norm.extend(extra)
    return Trajectory(lon=lon, lat=lat, time=pd.DatetimeIndex(norm))

# ---------------------------------------------------------------------------
# 7. Discovery
# ---------------------------------------------------------------------------

def discover_week_tags(dir_path: Path, safe_name: str, label_id: str, year: int, k_index: int) -> list[str]:
    if not dir_path.exists():
        return []
    prefix = f"{safe_name}_{label_id}_k{k_index}_{year}"
    tags = []
    for p in dir_path.glob(f"{prefix}w*_particles.zarr"):
        stem = p.name
        try:
            seg = stem.split(f"_{year}", 1)[1]
            wk = seg.split('_particles', 1)[0]
            tags.append(wk)
        except Exception:
            continue
    all_path = dir_path / f"{prefix}all_particles.zarr"
    if all_path.exists():
        tags.append('all')
    # Deduplicate preserving order
    seen = set()
    return [t for t in tags if not (t in seen or seen.add(t))]

def auto_detect_single_week(requested_tag: str, candidates: list[str]) -> str:
    if requested_tag != 'all':
        return requested_tag
    if 'all' in candidates:
        return 'all'
    return candidates[0] if len(candidates) == 1 else requested_tag

__all__ = [
    'configure_logging','sanitize_name','make_output_stem','particles_zarr_path','shapefile_path','png_path_from_stem',
    'normalize_years','normalize_weeks','make_week_tag','ordinal_week_for_date','expand_weeks_to_dates',
    'load_reef_layer','build_label_name_map','Trajectory','load_parcels_zarr','reconstruct_times','ensure_utc',
    'discover_week_tags','auto_detect_single_week'
]
