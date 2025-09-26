"""
Convert a Parcels particle trajectory Zarr store (produced by 04-basic-particle-flow.py)
into a point shapefile with one point per output observation.

Rationale
Keeping shapefile (or other GIS) conversion out of the simulation script keeps the
core advection path lean and makes it easier to iterate on multi‑particle runs
without repeatedly writing heavyweight formats. This helper performs only the
IO and light reshaping required for inspection / GIS ingestion.

Input layout
The Zarr store written by 04-basic-particle-flow.py contains lon(lat,time) variables
in the Parcels (trajectory, obs) layout. For the current single‑particle prototype
this is typically shape (1, N). Future multi‑particle runs will have shape (M, N)
where M is the number of particles and N the number of saved output times.

Time handling
Parcels may write either an array shaped like (trajectory, obs) or (obs,) depending
on version. We flatten both lon/lat/time to 1‑D arrays of equal length. When only
a single timestamp is present for multiple spatial samples (rare, but guarded),
we reconstruct a sequence assuming the provided --dt-hours or default 1 hour.

Usage examples
python 04b-convert-particle-zarr-to-shp.py --reef-shp data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp \
  --id 18-096 --year 2016 --week 01 --k-index 40

If weeks were stitched (e.g. w01-02) use the same tag you passed to simulation:
python 04b-convert-particle-zarr-to-shp.py --id 18-096 --year 2016 --week-range 01 02

Either --week or --week-range can be supplied (mutually exclusive). If neither is
supplied, the script assumes the simulation used all available weeks and the
Zarr name contains 'all'.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import logging
import sys
from datetime import timedelta
from typing import Sequence, List

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from utils import (
    sanitize_name,
    make_week_tag,
    load_reef_layer,
    build_label_name_map,
    load_parcels_zarr,
    discover_week_tags,
)


def configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Parcels trajectory Zarr to shapefile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--reef-shp', type=Path, default=Path('data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp'), help='Reef layer for GBR_NAME lookup (LABEL_ID, GBR_NAME required)')
    p.add_argument('--id', required=True, help='LABEL_ID used in simulation')
    p.add_argument('--year', required=True, type=int, help='Simulation year')
    p.add_argument('--week', type=str, help='Single week number like 01 (mutually exclusive with --week-range)')
    p.add_argument('--week-range', nargs=2, type=str, metavar=('WSTART','WEND'), help='Week range (inclusive) e.g. 01 02')
    p.add_argument('--k-index', type=int, required=True, help='k index used in simulation naming')
    p.add_argument('--outdir', type=Path, default=Path('working/02'), help='Root directory that holds the Zarr output tree (usually same as simulation --outdir)')
    p.add_argument('--dt-hours', type=float, default=1.0, help='Assumed interval if time reconstruction needed')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing shapefile')
    p.add_argument('--output-suffix', default='track', help='Suffix stem for shapefile; final file ends _<suffix>.shp')
    return p.parse_args(argv)


def derive_week_tag(single: str | None, week_range: List[str] | None) -> str:
    if single and week_range:
        raise SystemExit('Provide either --week or --week-range, not both')
    if single:
        return f"w{int(single):02d}"
    if week_range:
        a, b = (int(week_range[0]), int(week_range[1]))
        if a > b:
            a, b = b, a
        if a == b:
            return f"w{a:02d}"
        return f"w{a:02d}-{b:02d}"
    return 'all'


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    if not args.reef_shp.exists():
        logging.error('Reef shapefile not found: %s', args.reef_shp)
        return 2
    try:
        reefs = load_reef_layer(args.reef_shp)
    except Exception as e:
        logging.error(str(e))
        return 3
    ref = build_label_name_map(reefs)
    if args.id not in ref:
        logging.error('LABEL_ID %s not present in reef layer', args.id)
        return 4
    gbr_name = ref[args.id]
    safe = sanitize_name(gbr_name)
    week_tag = derive_week_tag(args.week, args.week_range)

    # If user omitted explicit weeks (week_tag == 'all') but no 'all' file exists, attempt auto-discovery.
    zarr_dir = args.outdir / args.id / str(args.year)
    zarr_path = zarr_dir / f"{safe}_{args.id}_k{args.k_index}_{args.year}{week_tag}_particles.zarr"
    if week_tag == 'all' and not zarr_path.exists():
        candidates = discover_week_tags(zarr_dir, safe, args.id, args.year, args.k_index)
        if len(candidates) == 1:
            week_tag = candidates[0]
            zarr_path = zarr_dir / f"{safe}_{args.id}_k{args.k_index}_{args.year}{week_tag}_particles.zarr"
            logging.info("Auto-detected week tag '%s' -> %s", week_tag, zarr_path.name)
        elif len(candidates) > 1:
            logging.error('Multiple candidate week tags found %s; specify --week/--week-range', candidates)
            return 5
    if not zarr_path.exists():
        logging.error('Zarr store not found: %s', zarr_path)
        return 5

    shp_path = zarr_path.parent / f"{safe}_{args.id}_k{args.k_index}_{args.year}{week_tag}_{args.output_suffix}.shp"
    if shp_path.exists() and not args.overwrite:
        logging.error('Shapefile exists (use --overwrite): %s', shp_path)
        return 6
    if shp_path.exists() and args.overwrite:
        for ext in ('.shp','.shx','.dbf','.prj','.cpg'):
            p = shp_path.with_suffix(ext)
            if p.exists():
                p.unlink()

    logging.info('Opening trajectory %s', zarr_path)
    traj = load_parcels_zarr(zarr_path, assume_dt_hours=args.dt_hours)
    # Convert to Australia/Brisbane timezone
    bris = traj.time.tz_convert('Australia/Brisbane')
    iso_times = [t.isoformat() for t in bris]
    lons = traj.lon
    lats = traj.lat

    gdf = gpd.GeoDataFrame(
        {
            'LABEL_ID': [args.id]*len(lons),
            'GBR_NAME': [gbr_name]*len(lons),
            'HOUR_IDX': list(range(len(lons))),
            'TIME_ISO': iso_times,
        },
        geometry=[Point(float(x), float(y)) for x,y in zip(lons, lats)],
        crs='EPSG:4326'
    )
    gdf.to_file(shp_path)
    logging.info('Wrote shapefile %s (%d points)', shp_path, len(gdf))
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
