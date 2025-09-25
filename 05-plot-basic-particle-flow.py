#!/usr/bin/env python
"""
Plot particle trajectory outputs produced by 04-basic-particle-flow.py.

Features:
- Select reefs by LABEL_ID, years, ordinal weeks, and k-index to match naming.
- Automatically reconstruct the particle Zarr directory path(s) used in 04.
- Load lon/lat/time from the Zarr output and plot all trajectory points as small dots.
- Overlay the reef outline polygon(s) (full geometry subset to selected LABEL_IDs; if multiple IDs use all chosen).
- Optionally color points by elapsed time.

Assumptions:
- Output directory structure: working/02/<LABEL_ID>/<YEAR>/<SAFE_NAME>_<LABEL_ID>_k<kindex>_<YEAR><week_tag>_particles.zarr
  where week_tag is either 'wNN' or 'wNN-wMM' or 'all' when all weeks were run.
- Zarr contains variables lon, lat (1D or 2D). If 2D (time, particle) flatten is applied.
- Time may be 1D with same length as lon/lat or shorter; if missing, an index-based time is synthesized.

Week semantics:
- Weeks follow the ordinal definition from script 04: Jan1-Jan7 = week 1, next 7-day blocks sequential.
- If multiple week numbers supplied they are consolidated into a single tag matching 04's scheme.

Output:
- Saves PNG(s) to <outdir>/<LABEL_ID>/<YEAR>/<...>_track.png with same base pattern as trajectory.
- Can optionally show interactive window when --show is set.

Limitations:
- Designed for single-particle runs; multi-particle flattening will show all points without differentiation.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence, List, Dict
from datetime import date, timedelta, datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

# ---------------- CLI Helpers (mirroring 04) ----------------

def _sanitize_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name.replace(" ", "_"))

def _years_list(vals: List[int]) -> List[int]:
    if len(vals) == 1:
        return vals
    if len(vals) == 2:
        a, b = vals
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return sorted(set(vals))

def _weeks_list(vals: List[int] | None) -> List[int] | None:
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

# ---------------- Week Tag Logic (match 04) ----------------

def make_week_tag(weeks: List[int] | None) -> str:
    if not weeks:
        return "all"
    uniq = sorted(set(weeks))
    return f"w{uniq[0]:02d}" if len(uniq) == 1 else f"w{uniq[0]:02d}-{uniq[-1]:02d}"

# ---------------- Core Functionality ----------------

def discover_run_paths(base_out: Path, label_id: str, year: int, safe_name: str, k_index: int, week_tag: str) -> Path:
    zarr_dir = base_out / label_id / str(year) / f"{safe_name}_{label_id}_k{k_index}_{year}{week_tag}_particles.zarr"
    return zarr_dir


def load_reef_layer(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise SystemExit(f"Reef shapefile not found: {path}")
    gdf = gpd.read_file(path)
    if 'LABEL_ID' not in gdf.columns or 'GBR_NAME' not in gdf.columns:
        raise SystemExit("Reef file missing LABEL_ID or GBR_NAME columns")
    return gdf


def load_traj_points(base_path: Path, assume_dt_hours: float) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load lon/lat/time from a Parcels trajectory output (NetCDF or Zarr).

    Why this exists:
    The simplified assertion-based loader failed because Parcels writes lon/lat/time
    on a (traj, obs) grid (even for a single particle) so time appeared length 1
    while lon flattened to number-of-observations. This function tolerates the
    canonical Parcels layout and reconstructs sequential times when only a single
    timestamp is stored (observed with some backends).

    Rules:
    - Prefer Zarr if <base>_particles.zarr exists, else fall back to NetCDF <base>_particles.nc.
    - Accept lon/lat/time with shape (traj, obs) or (obs,) after squeeze.
    - If time has shape (traj, obs) mirror the flattening applied to lon/lat.
    - If time resolves to a single value but there are multiple observations,
      reconstruct sequential times assuming constant dt = assume_dt_hours.
    - Return UTC tz-aware DatetimeIndex.
    """
    zarr_dir = base_path
    nc_file = Path(str(base_path).replace('_particles.zarr', '_particles.nc')) if str(base_path).endswith('.zarr') else Path(str(base_path) + '.nc')

    ds: xr.Dataset | None = None
    opened_from = None
    if zarr_dir.exists():
        ds = xr.open_zarr(zarr_dir)
        opened_from = 'zarr'
    elif nc_file.exists():
        ds = xr.open_dataset(nc_file)
        opened_from = 'netcdf'
    else:
        raise SystemExit(f"No trajectory output found at {zarr_dir} or {nc_file}")
    try:
        for v in ('lon','lat'):
            if v not in ds.variables:
                raise SystemExit(f"Trajectory file missing variable '{v}' in {opened_from} dataset")
        lon = np.asarray(ds['lon'].values)
        lat = np.asarray(ds['lat'].values)
        time_vals = ds['time'].values if 'time' in ds.variables else None

        # Handle (traj, obs) layout
        def _flatten_xy(a: np.ndarray) -> np.ndarray:
            if a.ndim == 1:
                return a
            if a.ndim == 2:
                # Expect shape (traj, obs) or (obs, traj); choose the one with size==1 as traj axis
                if 1 in a.shape:
                    # Ensure obs axis is the one not equal to 1
                    if a.shape[0] == 1:
                        return a.reshape(-1)  # (1, obs)
                    if a.shape[1] == 1:
                        return a.reshape(-1)  # (obs,1)
                # Fallback: ravel in C order
                return a.reshape(-1)
            raise SystemExit(f"Unexpected lon/lat ndim {a.ndim}")

        lon_flat = _flatten_xy(lon)
        lat_flat = _flatten_xy(lat)
        if lon_flat.size != lat_flat.size:
            raise SystemExit(f"Lon/lat size mismatch after flatten: {lon_flat.size} vs {lat_flat.size}")

        # Time handling
        if time_vals is None:
            # Synthesize monotonic times from zero
            base = pd.Timestamp('1970-01-01T00:00:00Z')
            times_list = [base + pd.Timedelta(hours=assume_dt_hours*i) for i in range(lon_flat.size)]
        else:
            tarr = np.asarray(time_vals)
            # Flatten similar to lon if 2D
            if tarr.ndim == 2 and 1 in tarr.shape:
                tarr = tarr.reshape(-1)
            elif tarr.ndim == 2 and tarr.shape == lon.shape:
                # Same grid as lon; flatten aligning with ravel order
                tarr = tarr.reshape(-1)
            # Convert
            try:
                times_parsed = pd.to_datetime(tarr, errors='raise')
            except Exception as e:  # noqa: BLE001
                raise SystemExit(f"Failed to parse time values: {e}") from e
            # If single timestamp but multiple positions, reconstruct sequential times
            if getattr(times_parsed, 'size', len(times_parsed)) == 1 and lon_flat.size > 1:
                base_t = times_parsed[0] if isinstance(times_parsed, (pd.DatetimeIndex, list, tuple)) else times_parsed
                if not isinstance(base_t, pd.Timestamp):
                    base_t = pd.Timestamp(base_t)
                    if base_t.tzinfo is None:
                        base_t = base_t.tz_localize('UTC')
                if base_t.tzinfo is None:
                    base_t = base_t.tz_localize('UTC')
                times_list = [base_t + pd.Timedelta(hours=assume_dt_hours*i) for i in range(lon_flat.size)]
            else:
                # Ensure 1D alignment
                if isinstance(times_parsed, pd.DatetimeIndex):
                    times_list = list(times_parsed)
                else:
                    times_list = list(pd.to_datetime(times_parsed))
                # If still mismatch (e.g. provided per-obs per-traj but flattened) pad or truncate
                if len(times_list) != lon_flat.size:
                    if len(times_list) == 1:
                        base_t = times_list[0]
                        if base_t.tzinfo is None:
                            base_t = base_t.tz_localize('UTC')
                        times_list = [base_t + pd.Timedelta(hours=assume_dt_hours*i) for i in range(lon_flat.size)]
                    else:
                        # Truncate or pad with last timestamp
                        if len(times_list) > lon_flat.size:
                            times_list = times_list[:lon_flat.size]
                        else:
                            last_t = times_list[-1]
                            if last_t.tzinfo is None:
                                last_t = last_t.tz_localize('UTC')
                            extra = [last_t + pd.Timedelta(hours=assume_dt_hours*(i+1)) for i in range(lon_flat.size - len(times_list))]
                            times_list.extend(extra)

        # Normalize to UTC DatetimeIndex
        norm = []
        for t in times_list:
            ts = t if isinstance(t, pd.Timestamp) else pd.Timestamp(t)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
            norm.append(ts)
        times_idx = pd.DatetimeIndex(norm)
        if times_idx.size != lon_flat.size:
            raise SystemExit(f"Final times size {times_idx.size} != lon size {lon_flat.size}")
        return lon_flat, lat_flat, times_idx
    finally:
        if ds is not None:
            ds.close()


def plot_run(lon, lat, times, reef_gdf: gpd.GeoDataFrame, out_png: Path, label_id: str, gbr_name: str, show: bool, color_time: bool):
    fig, ax = plt.subplots(figsize=(6, 6))
    subset = reef_gdf[reef_gdf['LABEL_ID'] == label_id]
    if subset.empty:
        logging.warning("No reef geometry for LABEL_ID %s", label_id)
    else:
        subset.boundary.plot(ax=ax, color='black', linewidth=1.0, alpha=0.7)
    if color_time and len(times) == len(lon):
        # Normalize time progression 0..1
        t0 = times[0]
        frac = (times - t0) / (times[-1] - t0) if len(times) > 1 else np.zeros_like(times, dtype=float)
        sc = ax.scatter(lon, lat, c=frac, s=12, cmap='viridis', edgecolor='none')
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Relative Time Progression')
    else:
        ax.scatter(lon, lat, s=12, color='tab:red', edgecolor='none')
    ax.set_title(f"Trajectory: {gbr_name} ({label_id})")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    logging.info("Saved plot %s (%d points)", out_png, len(lon))
    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------------- Argument Parsing ----------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot particle trajectories produced by 04-basic-particle-flow.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--reef-shp', type=Path, default=Path('data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp'))
    p.add_argument('--ids', nargs='+', required=True, help='LABEL_ID values to plot.')
    p.add_argument('--years', nargs='+', type=int, required=True, help='Year or inclusive range (two numbers).')
    p.add_argument('--weeks', nargs='*', type=int, default=None, help='Ordinal week numbers (Jan1-Jan7=1 ...).')
    p.add_argument('--k-index', type=int, default=40, help='Vertical index to match simulation output naming.')
    p.add_argument('--outdir', type=Path, default=Path('working/02'), help='Root directory where particle outputs were written.')
    p.add_argument('--color-time', action='store_true', help='Color points by relative time progression.')
    p.add_argument('--assume-dt-hours', type=float, default=1.0, help='Assumed dt hours when reconstructing times (if only single timestamp present).')
    p.add_argument('--show', action='store_true', help='Display plot window instead of only saving PNG.')
    p.add_argument('--log-file', type=Path, default=None)
    return p.parse_args(argv)

# ---------------- Main ----------------

def configure_logging(log_file: Path | None):
    if logging.getLogger().handlers:
        return
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    if log_file:
        fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S'))
        logging.getLogger().addHandler(fh)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)
    logging.info("Starting trajectory plotting")

    try:
        years = _years_list([int(v) for v in args.years]) if isinstance(args.years[0], str) else _years_list(args.years)
    except Exception as e:  # noqa: BLE001
        logging.error("Invalid years specification: %s", e)
        return 2
    try:
        weeks = _weeks_list(args.weeks)
    except ValueError as e:  # noqa: BLE001
        logging.error(str(e))
        return 3

    reef_gdf = load_reef_layer(args.reef_shp)
    name_map: Dict[str, str] = reef_gdf.drop_duplicates(subset=['LABEL_ID']).set_index('LABEL_ID')['GBR_NAME'].to_dict()

    week_tag = make_week_tag(weeks)
    logging.info("Week tag: %s", week_tag)

    any_plotted = False
    for rid in args.ids:
        if rid not in name_map:
            logging.warning("LABEL_ID %s not found in reef file; skipping", rid)
            continue
        gbr_name = name_map[rid]
        safe_name = _sanitize_name(gbr_name)
        for year in years:
            zarr_dir = discover_run_paths(args.outdir, rid, year, safe_name, args.k_index, week_tag)
            if not zarr_dir.exists():
                logging.warning("Missing trajectory directory: %s", zarr_dir)
                continue
            try:
                lon, lat, times = load_traj_points(zarr_dir, assume_dt_hours=args.assume_dt_hours)
            except SystemExit as e:
                logging.error(str(e))
                continue
            png_path = zarr_dir.with_name(zarr_dir.name.replace('_particles.zarr', '_track.png'))
            plot_run(lon, lat, times, reef_gdf, png_path, rid, gbr_name, show=args.show, color_time=args.color_time)
            any_plotted = True

    if not any_plotted:
        logging.error("No plots produced (check inputs / outputs exist)")
        return 4
    logging.info("All plotting completed")
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
