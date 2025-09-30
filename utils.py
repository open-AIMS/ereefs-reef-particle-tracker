"""utils.py - Shared helper functions for the eReefs extraction / particle workflow.

UPDATED ORIENTATION (Depth + Date Period Era)
=============================================
This module centralises all cross-cutting helpers (config loading, naming, trajectory I/O, legacy week
utilities) so scripts 03 / 04 / 05 remain thin orchestration layers. New depth-aware / date-period
conventions co‑exist with legacy week naming for backward compatibility.

PRIMARY GROUPS (import what you need)
-------------------------------------
Logging
    configure_logging(log_file=None, level=INFO) -> None
        Uniform console/file logging set‑up (call exactly once per CLI).

Modern Naming (depth + period based)
    depth_token(depth_m) -> str
        Format target depth with sign, <=2 decimals, trailing zeros trimmed, suffixed 'm' (e.g. -2.35m, -2m).
    simulation_filename(root, model_name, label_id, depth_m, start_date, length_days) -> Path
        New particle simulation Zarr path pattern:
            {model_name}_{LABEL_ID}_particles_{depthToken}_{YYYYMMDD}_{Nd}.zarr
        (Used by script 04; discovered by script 05.)
    png_path_from_stem / shapefile_path / make_output_stem
        Retained for legacy week-based stems (converter or archival workflows).

Legacy Week Naming (kept temporarily)
    make_output_stem(safe_name, label_id, k_index, year, week_tag) -> str
    particles_zarr_path(root, label_id, year, safe_name, k_index, week_tag) -> Path
    discover_week_tags(...), make_week_tag(...), normalize_weeks(...)
        These support earlier week-sliced simulations. Marked for deprecation once all downstream
        tooling migrates to date periods + explicit runtime / start-day semantics.

Date / Period Utilities
    normalize_years(list[int]) -> list[int]
        Expand 2-value inclusive range or deduplicate arbitrary list.
    normalize_date_periods(raw) -> list[str] | None
        Accept single string or list (e.g. "Jan01-Apr30"). Returns validated canonical strings.
    expand_date_periods_for_year(year, specs) -> list[date]
        Expand each specification to concrete dates inside the given year (wrap portions truncated).
    make_period_tag(specs|None) -> str
        Compact descriptor (e.g. Jan01-Apr30+Nov15-Dec31 or 'all').

Reef Metadata
    load_reef_layer(path) -> GeoDataFrame
        Validates presence of LABEL_ID & GBR_NAME. Heavy import done lazily.
    build_label_name_map(gdf) -> dict
        Quick LABEL_ID -> GBR_NAME mapping.

Trajectory / Time Handling
    Trajectory dataclass(lon, lat, time)
    load_parcels_zarr(store, assume_dt_hours=1.0) -> Trajectory
        Robust against flattened vs (traj, obs) layouts; reconstructs time axis if absent/underspecified.
    ensure_utc(ts), reconstruct_times(base, count, dt_hours)
        Guarantee UTC-normalised monotonically spaced DateTimeIndex.

Config & Parameters
    Config dataclass fields include: ids, years, weeks (legacy), date_periods, depth_m, k_index,
    data_root (daily subcubes), traces_root (particle simulations), plots_root (optional plot output root),
    opendap_url, reef_shp (reef metadata layer), dt_hours, runtime, overwrite,
    log_file, buffer_km.
    load_config(path) -> Config
        TOML reader supporting single-string or list date_periods.

Error & Design Philosophy
    * Pure path builders (no filesystem effects). Callers decide mkdir/exists semantics.
    * Fail early with clear ValueError / FileNotFoundError; CLI wrappers convert to user-facing messages.
    * Heavy deps (geopandas, xarray) loaded only when needed.
    * UTC everywhere for temporal data; attribute enrichment done at write points (scripts 03 & 04).

Current Naming Contracts
    DAILY SUBCUBES (script 03; under data_root):
        {model_name}_{LABEL_ID}_UV_{depthToken}_{YYYYMMDD}.nc
    PARTICLE ZARR (script 04; under traces_root):
        {model_name}_{LABEL_ID}_particles_{depthToken}_{YYYYMMDD}_{Nd}.zarr
    PLOTS (script 05; under plots_root if set else traces_root):
        *_track.png
    (Legacy week form retained internally: {safe}_{LABEL_ID}_k{k}_{YEAR}{week_tag}_particles.zarr)

Examples
    from utils import depth_token, simulation_filename
    tok = depth_token(-2.35)               # -> '-2.35m'
    zpath = simulation_filename(Path('working/02'), 'gbr1', '18-096', -2.35, date(2016,1,1), 3)
    # working/02/18-096/2016/gbr1_18-096_particles_-2.35m_20160101_3d.zarr

Deprecation Roadmap
    Week-based helpers will be pruned after remaining converters & plotters migrate fully to
    period + depth naming. Until then, they remain stable but clearly partitioned.

This docstring is intentionally verbose to aid both human and LLM comprehension; keep structural
sections intact if editing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional
import logging
import textwrap
import sys
import numpy as np
import pandas as pd
from datetime import date, timedelta
try:
    import tomllib  # Python 3.11+
except Exception:  # noqa: BLE001
    tomllib = None  # type: ignore

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

# New depth & simulation naming (daily extraction already embeds depth token pattern); depth token keeps sign
def depth_token(depth_m: float) -> str:
    s = f"{depth_m:.2f}".rstrip('0').rstrip('.')
    return f"{s}m"

def simulation_filename(root: Path, model_name: str, label_id: str, depth_m: float, start_date, length_days: int) -> Path:  # start_date: date
    tok = depth_token(depth_m)
    stem = f"{model_name}_{label_id}_particles_{tok}_{start_date:%Y%m%d}_{length_days}d.zarr"
    return root / label_id / f"{start_date.year}" / stem

def multi_particle_simulation_filename(root: Path, model_name: str, label_id: str, depth_m: float, start_date, length_days: int, num_particles: int) -> Path:
    """Return path for multi-particle simulation Zarr store.

    Pattern:
        {model}_{LABEL_ID}_mparticles_{depthTok}_{YYYYMMDD}_{Nd}_{Np}p.zarr

    Rationale: Distinguish from single-particle runs and encode particle count for quick discovery
    and downstream grouping. 'mparticles' token avoids accidental collision with legacy naming.
    """
    tok = depth_token(depth_m)
    stem = f"{model_name}_{label_id}_mparticles_{tok}_{start_date:%Y%m%d}_{length_days}d_{num_particles}p.zarr"
    return root / label_id / f"{start_date.year}" / stem

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
# 6. Config (new minimal implementation to support runtime & log file)

@dataclass
class Config:
    ids: List[str]
    years: List[int]
    # Mutually exclusive legacy weeks vs new date_periods (date_periods takes precedence when provided)
    weeks: Optional[List[int]]
    date_periods: Optional[List[str]]  # raw period specs like "Jan01-Apr30", "Nov15-Dec31"
    model_name: str
    depth_m: float
    k_index: int
    data_root: Path
    traces_root: Path  # directory for simulation (particle) outputs
    plots_root: Path | None  # optional separate root for plots (falls back to traces_root if None)
    opendap_url: str
    reef_shp: Path
    fnode_nc: Path
    dt_hours: float = 1.0
    runtime: Optional[float] = None
    overwrite: bool = False
    log_file: Optional[Path] = None
    buffer_km: float = 0.0
    # Multi-particle experiment parameters (script 06)
    num_particles: int = 0              # 0 -> feature disabled unless script provides override
    injection_interval: int = 7         # days between injections (start-date stepping)
    random_seed: Optional[int] = None   # base seed; combined with label_id/start_date for deterministic runs
    kh_m2s: float = 0.0                 # horizontal diffusion coefficient (m^2/s); 0 disables diffusion

MONTH_MAP = {m.lower(): i for i, m in enumerate(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'], start=1)}

def _parse_month_day(token: str) -> tuple[int,int]:
    if len(token) < 5:
        raise ValueError(f"Invalid month-day token: {token}")
    mon_part = token[:3].lower()
    day_part = token[3:]
    if mon_part not in MONTH_MAP:
        raise ValueError(f"Invalid month abbreviation in {token}")
    day = int(day_part)
    if not 1 <= day <= 31:
        raise ValueError(f"Invalid day in {token}")
    return MONTH_MAP[mon_part], day

def normalize_date_periods(raw_periods) -> List[str] | None:  # type: ignore[override]
    """Normalize date_periods from TOML input.

    Accepts:
      - None / empty -> None
      - Single string: "Jan01-Apr30"
      - List of strings
    Returns cleaned list preserving original order (deduplicated) or None.
    """
    if raw_periods is None:
        return None
    # If a single string was supplied in TOML, tomllib returns str not list.
    if isinstance(raw_periods, str):
        raw_list = [raw_periods]
    else:
        raw_list = list(raw_periods)
    cleaned: List[str] = []
    for spec in raw_list:
        if spec is None:
            continue
        spec = str(spec).strip()
        if not spec:
            continue
        if '-' not in spec:
            raise ValueError(f"Date period must contain '-' (start-end), got '{spec}'")
        start_tok, end_tok = spec.split('-',1)
        _parse_month_day(start_tok)
        _parse_month_day(end_tok)
        cleaned.append(f"{start_tok}-{end_tok}")
    seen = set(); out: List[str] = []
    for c in cleaned:
        if c not in seen:
            seen.add(c); out.append(c)
    return out or None

def expand_date_periods_for_year(year: int, period_specs: List[str]) -> List[date]:
    """Expand period specs into concrete date list for the given year.

    Handles wrap-around if end < start by spanning year boundary, but truncates outside 'year'.
    Example: Dec15-Jan10 yields Dec15..Dec31 of 'year' only (future year days ignored here). Multiple years should be processed separately.
    """
    days: set[date] = set()
    for spec in period_specs:
        start_tok, end_tok = spec.split('-',1)
        sm, sd = _parse_month_day(start_tok)
        em, ed = _parse_month_day(end_tok)
        import calendar
        # Helper to clamp invalid day at month end (defensive for 30/31 mismatches like Apr31)
        def _mk(y: int, m: int, d: int) -> date:
            last = calendar.monthrange(y,m)[1]
            if d>last:
                d=last
            return date(y,m,d)
        start = _mk(year, sm, sd)
        end = _mk(year, em, ed)
        if (em > sm) or (em==sm and ed >= sd):
            cur = start
            while cur <= end:
                days.add(cur)
                cur += timedelta(days=1)
        else:  # wrap-around (e.g., Nov15-Feb10) -> add start..Dec31
            cur = start
            while cur.year == year:
                days.add(cur)
                cur += timedelta(days=1)
            # Do not add next-year portion here; handled when calling for next year.
    return sorted(days)

def make_period_tag(period_specs: List[str] | None) -> str:
    if not period_specs:
        return 'all'
    # Canonical condensed form: Jan01-Apr30+Nov15-Dec31 (keep original order)
    return "+".join(period_specs)

def load_config(path: Path) -> Config:
    if tomllib is None:
        raise RuntimeError("tomllib not available; need Python 3.11+ for config loading")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open('rb') as f:
        raw = tomllib.load(f)
    def _req(key: str):
        if key not in raw:
            raise ValueError(f"Missing required config key: {key}")
        return raw[key]
    ids = list(map(str, _req('ids')))
    years = list(map(int, _req('years')))
    weeks = raw.get('weeks')
    if weeks is not None:
        weeks = [int(w) for w in weeks]
    date_periods = normalize_date_periods(raw.get('date_periods')) if 'date_periods' in raw else None
    model_name = str(raw.get('model_name', 'gbr1'))
    depth_m = float(raw.get('depth_m', -2.35))
    opendap_url = str(raw.get('opendap_url', 'https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml'))
    reef_shp = Path(raw.get('reef_shp', 'data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp'))
    fnode_nc = Path(raw.get('fnode_nc', 'working/02/gbr1_fnodes.nc'))
    k_index = int(raw.get('k_index', 40))
    data_root = Path(raw.get('data_root', 'working/03-data'))
    traces_root = Path(raw.get('traces_root', 'working/04-traces'))
    plots_root_raw = raw.get('plots_root')
    plots_root = Path(plots_root_raw) if plots_root_raw else None
    dt_hours = float(raw.get('dt_hours', 1.0))
    runtime = raw.get('runtime')
    runtime = float(runtime) if runtime is not None else None
    overwrite = bool(raw.get('overwrite', False))
    log_file_raw = raw.get('log_file')
    log_file = Path(log_file_raw) if log_file_raw else None
    buffer_km = float(raw.get('buffer_km', 0.0))
    num_particles = int(raw.get('num_particles', 0))
    injection_interval = int(raw.get('injection_interval', 7))
    random_seed = raw.get('random_seed')
    random_seed = int(random_seed) if random_seed is not None else None
    kh_m2s = float(raw.get('kh_m2s', 0.0))
    cfg = Config(
        ids=ids,
        years=years,
        weeks=weeks,
        date_periods=date_periods,
        model_name=model_name,
        depth_m=depth_m,
        k_index=k_index,
        data_root=data_root,
        traces_root=traces_root,
        plots_root=plots_root,
        opendap_url=opendap_url,
        reef_shp=reef_shp,
        fnode_nc=fnode_nc,
        dt_hours=dt_hours,
        runtime=runtime,
        overwrite=overwrite,
        log_file=log_file,
        buffer_km=buffer_km,
        num_particles=num_particles,
        injection_interval=injection_interval,
        random_seed=random_seed,
        kh_m2s=kh_m2s,
    )
    return cfg


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
    'discover_week_tags','auto_detect_single_week','Config','load_config',
    'normalize_date_periods','expand_date_periods_for_year','make_period_tag','depth_token','simulation_filename','multi_particle_simulation_filename'
]
