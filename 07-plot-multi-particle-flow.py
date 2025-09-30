#!/usr/bin/env python
"""Plot multi-particle simulation outputs from `06-multi-particle-flow.py`.

Discovery Pattern
-----------------
Files produced by script 06 follow the naming convention (see `utils.multi_particle_simulation_filename`):
    {model}_{LABEL_ID}_mparticles_{depthTok}_{YYYYMMDD}_{Nd}_{Np}p.zarr
These live under:
    traces_root / LABEL_ID / YEAR /

Purpose
-------
For each multi-particle run, plot an overview PNG of particle trajectories:
    * Either all positions merged (default) or an optional subsample per particle id.
    * Colour points by elapsed hours since run start (optional flag) or uniform colour.
    * Overlay reef polygon boundary for spatial context.

Output PNG Naming
-----------------
Mirrors Zarr path, replacing `.zarr` with `_mtrack.png`:
    gbr1_18-096_mparticles_-2.35m_20160101_3d_100p_mtrack.png
If `plots_root` is set in the config, the relative directory structure is reproduced there; otherwise
PNG is written adjacent to the Zarr store.

Assumptions / Notes
-------------------
* Zarr structure produced by Parcels ParticleFile: expects lon/lat/time variables (potentially 2-D).
* Time variable may be 1-D (obs) or 2-D (trajectory, obs). We flatten to one array, repeating times
  per trajectory when necessary.
* If time length mismatches positions, we reconstruct a monotonic hourly index using config dt_hours.
* The script does not attempt per-particle path segmentation; it's a density / dispersion snapshot.
* Multi-run directories can be large; an optional `--max-points` and `--stride` allow thinning.

CLI Flags
---------
    --config        Path to TOML config
    --color-time    Colour points by hours since start
    --show          Display interactive window
    --max-points N  Hard cap on total plotted points after all filtering (random sample if exceeded)
    --stride S      Keep every S-th observation (applied before max-points sampling)
    --log-file      Override log file path

Future Enhancements (Not Implemented)
-------------------------------------
* Per-particle track fading / animation frames.
* Kernel density estimation contour overlay.
* Split panels by injection start date when overlapping.
* Residence statistics (separate analysis script).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import xarray as xr

from utils import (
    configure_logging,
    load_config,
    load_reef_layer,
    build_label_name_map,
    depth_token,
)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot multi-particle trajectory clouds (script 06 outputs)")
    p.add_argument('--config', required=True, type=Path, help='Path to TOML config file used for simulations')
    p.add_argument('--show', action='store_true', help='Display plot window')
    p.add_argument('--max-points', type=int, default=None, help='Maximum points to plot (after stride); random sample if exceeded')
    p.add_argument('--stride', type=int, default=1, help='Keep every S-th observation (>=1)')
    p.add_argument('--log-file', type=Path, default=None, help='Override log file path')
    return p.parse_args(argv)

# ---------------- Data Loading Helpers ----------------

def load_multi_run(store: Path, dt_hours: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp]:
    """Load a multi-particle Parcels Zarr store and return flattened lon/lat/time arrays.

    Returns (lon, lat, hours_since_start, t0_utc).
    hours_since_start is float array suitable for colour mapping when --color-time supplied.
    """
    if not store.exists():
        raise FileNotFoundError(store)
    ds = xr.open_zarr(str(store))
    try:
        if 'lon' not in ds or 'lat' not in ds:
            raise ValueError('Missing lon/lat in dataset')
        lon = np.asarray(ds['lon'].values)
        lat = np.asarray(ds['lat'].values)
        time = ds.get('time')
        # Normalise shapes -> flatten
        if lon.ndim == 2 and lat.ndim == 2:
            ntraj, nobs = lon.shape
            lon = lon.reshape(-1)
            lat = lat.reshape(-1)
            if time is not None:
                tvals = np.asarray(time.values)
                if tvals.ndim == 1 and tvals.size == nobs:
                    # Repeat for each trajectory
                    tvals = np.repeat(tvals, ntraj)
                elif tvals.ndim == 2:
                    tvals = tvals.reshape(-1)
                else:
                    # Fallback reconstruct
                    tvals = None
            else:
                tvals = None
        else:
            # Already flattened or 1-D
            if lon.ndim > 1:
                lon = lon.reshape(-1)
            if lat.ndim > 1:
                lat = lat.reshape(-1)
            if time is not None:
                tvals = np.asarray(time.values)
                if tvals.ndim > 1:
                    tvals = tvals.reshape(-1)
            else:
                tvals = None
        if lon.size != lat.size:
            raise ValueError('Lon/lat size mismatch after flatten')
        # Establish base time
        if tvals is not None and tvals.size == lon.size:
            try:
                t_index = pd.to_datetime(tvals, utc=True)
            except Exception:  # noqa: BLE001
                t_index = None
        else:
            t_index = None
        if t_index is None or t_index.size == 0:
            # Attempt reconstruction via attributes
            start_attr = ds.attrs.get('time_start') or ds.attrs.get('t_start')
            if start_attr:
                try:
                    t0 = pd.to_datetime(start_attr, utc=True)
                except Exception:  # noqa: BLE001
                    t0 = pd.Timestamp.utcnow().tz_localize('UTC')
            else:
                t0 = pd.Timestamp.utcnow().tz_localize('UTC')
            t_index = pd.DatetimeIndex([t0 + pd.Timedelta(hours=dt_hours * i) for i in range(lon.size)])
        else:
            t0 = t_index[0]
        # Compute hours since start
        t0 = t_index[0]
        hours_since_start = (t_index - t0) / np.timedelta64(1, 'h')
        return lon.astype(float), lat.astype(float), hours_since_start.astype(float), t0
    finally:
        ds.close()

# ---------------- Plotting ----------------

def plot_cloud(lon, lat, hours, reef_gdf: gpd.GeoDataFrame, label_id: str, gbr_name: str, t0: pd.Timestamp, out_png: Path, show: bool):
    fig, ax = plt.subplots(figsize=(6, 6))
    subset = reef_gdf[reef_gdf['LABEL_ID'] == label_id]
    if not subset.empty:
        subset.boundary.plot(ax=ax, color='black', linewidth=1.0, alpha=0.7)

    sc = ax.scatter(lon, lat, c=hours, s=6, cmap='viridis', edgecolor='none', alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Hours Since Start')

    # Format start time (ensure UTC) for inclusion in title
    try:
        t0_utc = t0.tz_convert('UTC') if t0.tzinfo else t0.tz_localize('UTC')  # type: ignore[attr-defined]
    except Exception:
        t0_utc = t0
    start_str = t0_utc.strftime('%Y-%m-%d %H:%MZ')
    ax.set_title(f"Multi-Particle Dispersion: {gbr_name} ({label_id})\nStart: {start_str}")
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_aspect('equal')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    logging.info("Saved plot %s (%d points)", out_png, lon.size)
    if show:
        plt.show()
    else:
        plt.close(fig)

# ---------------- Main Workflow ----------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    log_file = args.log_file if args.log_file else cfg.log_file
    configure_logging(log_file)
    logging.info("Starting multi-particle plotting")

    if args.stride < 1:
        logging.error('Stride must be >=1 (got %d)', args.stride)
        return 2

    reef_gdf = load_reef_layer(cfg.reef_shp)
    name_map = build_label_name_map(reef_gdf)
    depth_tok = depth_token(cfg.depth_m)  # align with simulation depth filter

    pattern_core = f"{cfg.model_name}_{{label}}_mparticles_{depth_tok}_*_*.zarr"

    any_plotted = False
    for label_id in cfg.ids:
        if label_id not in name_map:
            logging.warning('LABEL_ID %s not found in reef layer; skipping', label_id)
            continue
        gbr_name = name_map[label_id]
        for year in cfg.years:
            year_dir = cfg.traces_root / label_id / str(year)
            if not year_dir.exists():
                continue
            pattern = pattern_core.format(label=label_id)
            zarrs = sorted(year_dir.glob(pattern))
            if not zarrs:
                logging.info('No multi-particle runs for %s %d', label_id, year)
                continue
            for store in zarrs:
                try:
                    lon, lat, hours, t0 = load_multi_run(store, cfg.dt_hours)
                except Exception as e:  # noqa: BLE001
                    logging.warning('Failed to load %s: %s', store, e)
                    continue
                # Apply stride thinning
                if args.stride > 1:
                    keep = np.arange(0, lon.size, args.stride)
                    lon = lon[keep]; lat = lat[keep]; hours = hours[keep]
                # Apply max-points cap
                if args.max_points is not None and lon.size > args.max_points > 0:
                    rng = np.random.default_rng(0)  # deterministic sample for reproducibility
                    idx = rng.choice(lon.size, size=args.max_points, replace=False)
                    lon = lon[idx]; lat = lat[idx]; hours = hours[idx]
                # Build output path
                if cfg.plots_root:
                    rel = store.relative_to(cfg.traces_root)
                    out_png = cfg.plots_root / rel.parent / rel.name.replace('.zarr', '_mtrack.png')
                else:
                    out_png = store.with_name(store.name.replace('.zarr', '_mtrack.png'))
                plot_cloud(lon, lat, hours, reef_gdf, label_id, gbr_name, t0, out_png, show=args.show)
                any_plotted = True
    if not any_plotted:
        logging.error('No multi-particle plots produced')
        return 3
    logging.info('Multi-particle plotting complete')
    return 0

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
