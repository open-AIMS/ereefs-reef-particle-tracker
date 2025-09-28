#!/usr/bin/env python
"""Config-driven plotting of particle simulations (daily-depth naming).

Refactored to use shared TOML config (same file as extractor / simulator). Weeks are deprecated; this
script discovers simulation Zarr outputs based on the new depth-inclusive filename pattern created by
`utils.simulation_filename` used in `04-basic-particle-flow.py`:

        {model_name}_{LABEL_ID}_particles_{depthToken}_{startDateYYYYMMDD}_{runLengthDays}d.zarr

Features:
    * Loads the same TOML config (ids, years, depth_m, model_name, data_root/traces_root, date_periods optional)
    * Discovers simulation Zarr stores per reef/year by globbing for the above pattern
    * Loads lon/lat/time via `utils.load_parcels_zarr`
    * Plots trajectory as dots (optionally coloured by relative time progression)
    * Overlays reef polygon geometry for context
    * Writes PNG next to the Zarr with suffix `_track.png`

Assumptions / Notes:
    * Single-particle simulations (current workflow); multi-particle will just flatten all positions
    * Zarr layout `(trajectory, obs)` or flattened; loader normalises
    * Time axis hourly; relative colour mapping uses first->last timestamp
    * If multiple simulation Zarrs exist for a reef/year (e.g., different start dates), all are plotted
    * Config `date_periods` is not used for filtering here (plots what exists)

CLI now only accepts --config plus optional `--color-time`, `--show`, `--log-file`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from utils import (
    configure_logging,
    load_config,
    load_parcels_zarr,
    load_reef_layer,
    build_label_name_map,
    depth_token,
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
    p = argparse.ArgumentParser(description="Plot particle trajectories (config-only daily-depth naming)")
    p.add_argument('--config', required=True, type=Path, help='Path to TOML config used for simulation')
    p.add_argument('--reef-shp', type=Path, default=Path('data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp'))
    p.add_argument('--color-time', action='store_true', help='Colour points by relative time progression')
    p.add_argument('--show', action='store_true', help='Display plot window instead of only saving PNG')
    p.add_argument('--log-file', type=Path, default=None)
    return p.parse_args(argv)

# ---------------- Main ----------------

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Load config first (logging file may be specified there)
    cfg = load_config(args.config)
    # CLI log_file overrides config if provided
    log_file = args.log_file if args.log_file else cfg.log_file
    configure_logging(log_file)
    logging.info("Starting trajectory plotting (config-driven)")
    # Reef layer
    try:
        reef_gdf = load_reef_layer(args.reef_shp)
    except Exception as e:  # noqa: BLE001
        logging.error("Failed to load reef layer: %s", e)
        return 5
    name_map = build_label_name_map(reef_gdf)
    depth_tok = depth_token(cfg.depth_m)
    any_plotted = False
    for rid in cfg.ids:
        if rid not in name_map:
            logging.warning("LABEL_ID %s not in reef layer; skipping", rid)
            continue
        gbr_name = name_map[rid]
        for year in cfg.years:
            # Glob all simulation zarr stores for this reef/year matching new pattern
            year_dir = cfg.traces_root / rid / str(year)
            if not year_dir.exists():
                logging.info("Year dir missing: %s", year_dir)
                continue
            pattern = f"{cfg.model_name}_{rid}_particles_{depth_tok}_*_*.zarr"
            matches = sorted(year_dir.glob(pattern))
            if not matches:
                logging.info("No simulations found for %s %d", rid, year)
                continue
            for zpath in matches:
                try:
                    traj = load_parcels_zarr(zpath, assume_dt_hours=cfg.dt_hours)
                except Exception as e:  # noqa: BLE001
                    logging.warning("Failed to load %s: %s", zpath, e)
                    continue
                # Derive output plot path: if plots_root configured, mirror directory structure there
                if getattr(cfg, 'plots_root', None):
                    rel = zpath.relative_to(cfg.traces_root)
                    png_path = cfg.plots_root / rel.parent / rel.name.replace('.zarr', '_track.png')
                else:
                    png_path = zpath.with_name(zpath.name.replace('.zarr','_track.png'))
                plot_run(traj.lon, traj.lat, traj.time, reef_gdf, png_path, rid, gbr_name, show=args.show, color_time=args.color_time)
                any_plotted = True
    if not any_plotted:
        logging.error("No plots produced (check simulation outputs exist)")
        return 4
    logging.info("All plotting completed")
    return 0


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
