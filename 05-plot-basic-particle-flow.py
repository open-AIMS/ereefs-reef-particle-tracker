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

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from utils import (
    configure_logging,
    sanitize_name,
    normalize_years,
    normalize_weeks,
    make_week_tag,
    load_reef_layer as load_reef_layer_util,
    build_label_name_map,
    load_parcels_zarr,
    discover_week_tags,
    particles_zarr_path,
)

    # (All helper implementations replaced by centralized utilities from utils.py)


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

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)
    logging.info("Starting trajectory plotting")

    try:
        years = normalize_years([int(v) for v in args.years]) if isinstance(args.years[0], str) else normalize_years(args.years)
    except Exception as e:  # noqa: BLE001
        logging.error("Invalid years specification: %s", e)
        return 2
    try:
        weeks = normalize_weeks(args.weeks)
    except ValueError as e:  # noqa: BLE001
        logging.error(str(e))
        return 3
    # Load reef layer through utils (keeps logic consistent in one place)
    try:
        reef_gdf = load_reef_layer_util(args.reef_shp)
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to load reef layer: %s", e)
        return 5
    name_map: Dict[str, str] = build_label_name_map(reef_gdf)

    # Decide week tags: if weeks supplied, single consolidated tag. If not, discover all *_w??*_particles.zarr for each reef/year.
    explicit_week_tag = make_week_tag(weeks)
    logging.info("Requested week tag (explicit or placeholder): %s", explicit_week_tag)

    any_plotted = False
    for rid in args.ids:
        if rid not in name_map:
            logging.warning("LABEL_ID %s not found in reef file; skipping", rid)
            continue
        gbr_name = name_map[rid]
        safe_name = sanitize_name(gbr_name)
        for year in years:
            if weeks:
                # Use the consolidated explicit tag only
                tags_to_plot = [explicit_week_tag]
            else:
                # Discover via utility
                yr_dir = args.outdir / rid / str(year)
                tags_to_plot = discover_week_tags(yr_dir, safe_name, rid, year, args.k_index) or [explicit_week_tag]

            for tag in tags_to_plot:
                zarr_dir = particles_zarr_path(args.outdir, rid, year, safe_name, args.k_index, tag)
                if not zarr_dir.exists():
                    logging.warning("Missing trajectory directory: %s", zarr_dir)
                    continue
                try:
                    traj = load_parcels_zarr(zarr_dir, assume_dt_hours=args.assume_dt_hours)
                    lon, lat, times = traj.lon, traj.lat, traj.time
                except SystemExit as e:
                    logging.error(str(e))
                    continue
                png_path = zarr_dir.with_name(zarr_dir.name.replace('_particles.zarr', f'_{tag}_track.png') if tag not in zarr_dir.name else zarr_dir.name.replace('_particles.zarr','_track.png'))
                plot_run(lon, lat, times, reef_gdf, png_path, rid, gbr_name, show=args.show, color_time=args.color_time)
                any_plotted = True

    if not any_plotted:
        logging.error("No plots produced (check inputs / outputs exist)")
        return 4
    logging.info("All plotting completed")
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
