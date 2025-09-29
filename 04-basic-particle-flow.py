"""
This script demonstrates the basic setup and execution of a particle simulation using the OceanParcels
library, leveraging the eReefs GBR1 dataset. It focuses on simulating a single particle.

Key Steps:
1. Load the necessary libraries and configuration.
2. Define the simulation parameters, including the subcube to process.
3. Validate the input data and ensure all required attributes are present.
4. Construct the Parcels FieldSet using the f-node grid and subcube data.
5. Initialize and run the particle simulation.
6. Save the output trajectory data in Zarr format.

"""

from __future__ import annotations

import argparse
import contextlib
import logging
from pathlib import Path
from typing import List, Sequence
from datetime import timedelta, date
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd  # reef metadata

from utils import (
    configure_logging,
    load_config,
    expand_date_periods_for_year,
    depth_token,
    simulation_filename,
)

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
import parcels  # needed for parcels.StatusCode in DeleteParticle handler

# ------------------------------ Helpers ---------------------------------

## depth_token now provided by utils

def discover_daily_files(data_root: Path, label_id: str, year: int, model_name: str, depth_tok: str) -> List[Path]:
    year_dir = data_root / label_id / str(year)
    if not year_dir.exists():
        return []
    pattern = f"{model_name}_{label_id}_UV_*{year}*.nc"  # fallback broad pattern; refined filter applied below
    files = sorted(year_dir.glob(pattern))
    return files

def open_time_series(selected_files: List[Path]) -> xr.Dataset:
    """Open all selected daily files as a single time-series dataset.

    Uses by_coords combination and drops conflicting attrs (rare for curated extractions).
    Caller is responsible for closing the returned dataset.
    """
    return xr.open_mfdataset(
        [str(p) for p in selected_files],
        combine='by_coords',
        decode_times=True,
        combine_attrs='drop_conflicts'
    )



# ------------------------------- Core ------------------------------------

def run_simulation(cfg) -> None:
    log = logging.getLogger(__name__)
    # Load reef names once
    reef_shp = cfg.reef_shp
    if not reef_shp.exists():
        raise SystemExit(f"Reef shapefile missing. Have you run 01-download: {reef_shp}")
    reefs = gpd.read_file(reef_shp)
    ref = reefs.drop_duplicates(subset=['LABEL_ID']).set_index('LABEL_ID')['GBR_NAME'].to_dict()
    depth_tok = depth_token(cfg.depth_m)

    for label_id in cfg.ids:
        if label_id not in ref:
            log.warning("LABEL_ID %s not found in reef layer; skipping", label_id)
            continue
        for year in cfg.years:
            # ---------- Create a map of all daily files for this reef/year --------
            daily_files = discover_daily_files(cfg.data_root, label_id, year, cfg.model_name, depth_tok)
            if not daily_files:
                log.warning("No daily files for %s %d", label_id, year)
                continue
            # Filter by date periods if provided
            date_map = {}
            for p in daily_files:
                try:
                    day_token = p.stem.split('_')[-1]
                    d = pd.to_datetime(day_token, format='%Y%m%d').date()
                    date_map[d] = p
                except Exception:
                    continue
            # ------------ Find the date_periods files with the map ------------
            if cfg.date_periods:
                days_allowed = set(expand_date_periods_for_year(year, cfg.date_periods))
                selected_days = sorted(d for d in date_map if d in days_allowed)
            else:
                selected_days = sorted(date_map)
            if not selected_days:
                log.warning("No days selected after period filtering for %s %d", label_id, year)
                continue
            selected_files = [date_map[d] for d in selected_days]

            # Open combined time-series dataset once
            try:
                ds_ts = open_time_series(selected_files)
            except Exception as e:  # noqa: BLE001
                log.error("Failed to open combined dataset for %s %d: %s", label_id, year, e)
                continue
            # --------------------------------------------------------------
            # Assert buffer_km in config matches metadata in extracted files
            # This guards against running simulations on stale subcubes when
            # the spatial buffer was changed. 
            file_buffer_km = ds_ts.attrs.get('buffer_km', None)
            assert file_buffer_km is not None, f"Combined dataset missing buffer_km metadata"

            assert abs(float(file_buffer_km) - float(cfg.buffer_km)) < 1e-6, (
                f"Config buffer_km={cfg.buffer_km} mismatches dataset buffer_km={file_buffer_km}; "
                f"re-run 03-extract-gbr1-subcubes.py with buffer_km={cfg.buffer_km} (or update config to {file_buffer_km})."
            )
            # --------------------------------------------------------------
            # Build fieldset
            # Extract lon/lat/time and U/V variables from combined dataset
            for req in ("longitude", "latitude", "time"):
                if req not in ds_ts:
                    log.error("Combined dataset missing variable %s for %s %d", req, label_id, year)
                    ds_ts.close()
                    continue
            lon_arr = ds_ts['longitude'].values
            lat_arr = ds_ts['latitude'].values
            times_all = pd.to_datetime(ds_ts['time'].values)
            t_start = times_all.min()
            t_end_full = times_all.max()
            # Identify U/V candidate variables
            cand_u = [v for v in ds_ts.data_vars if v.lower().startswith('u')]
            cand_v = [v for v in ds_ts.data_vars if v.lower().startswith('v')]
            if not cand_u or not cand_v:
                log.error("Cannot identify U/V variables in combined dataset for %s %d", label_id, year)
                ds_ts.close()
                continue
            u_name, v_name = cand_u[0], cand_v[0]
            # Build FieldSet from in-memory arrays
            u_vals = ds_ts[u_name].transpose('time','j','i').values
            v_vals = ds_ts[v_name].transpose('time','j','i').values
            data = {'U': u_vals, 'V': v_vals}
            dimensions = {
                'U': {'lon': lon_arr, 'lat': lat_arr, 'time': times_all.values},
                'V': {'lon': lon_arr, 'lat': lat_arr, 'time': times_all.values},
            }
            fieldset = FieldSet.from_data(data=data, dimensions=dimensions, mesh='spherical', allow_time_extrapolation=False)
 
            available_hours = (t_end_full - t_start).total_seconds() / 3600.0
            if cfg.runtime is not None:
                if cfg.runtime > available_hours + 1e-6:  # allow tiny float tolerance
                    log.warning(
                        "Requested runtime %.2f h exceeds available data span %.2f h (start=%s end=%s). Using available span instead.",
                        cfg.runtime, available_hours, t_start, t_end_full
                    )
                runtime_hours = min(cfg.runtime, available_hours)
            else:
                runtime_hours = available_hours
            if runtime_hours <= 0:
                log.warning("No positive runtime available for %s %d (span %.2f h); skipping", label_id, year, available_hours)
                continue
            t_end = t_start + pd.Timedelta(hours=runtime_hours)
            sim_length_days = int(np.ceil(runtime_hours/24.0))
            out_path = simulation_filename(cfg.traces_root, cfg.model_name, label_id, cfg.depth_m, t_start.date(), sim_length_days)
            if out_path.exists() and not cfg.overwrite:
                log.info("Simulation output exists (skip): %s", out_path)
                continue
            if out_path.exists() and cfg.overwrite:
                import shutil
                shutil.rmtree(out_path, ignore_errors=True)
            # Seed in interior
            # lon_arr, lat_arr already defined above
            mask = np.isfinite(lon_arr) & np.isfinite(lat_arr)
            if not mask.any():
                log.error("All lon/lat NaN for %s %d", label_id, year)
                continue
            js, is_ = np.where(mask)
            jmid = int((js.min()+js.max())//2)
            imid = int((is_.min()+is_.max())//2)
            lon0 = float(lon_arr[jmid, imid]); lat0 = float(lat_arr[jmid, imid])
            class Particle(JITParticle):  # noqa: D401
                """Minimal particle class (can extend with custom variables later)."""
                pass
            pset = ParticleSet.from_list(fieldset=fieldset, pclass=Particle, lon=[lon0], lat=[lat0], time=t_start.to_pydatetime())
            with contextlib.suppress(Exception):
                pset.populate_indices()

            output_dt = timedelta(hours=cfg.dt_hours)
            log.debug("Simulation timesteps: integration=%.3f h output=%.3f h", cfg.dt_hours, cfg.dt_hours)
            pfile = pset.ParticleFile(name=str(out_path), outputdt=output_dt)
            try:
                def DeleteParticle(particle, fieldset, time):
                    if particle.state == parcels.StatusCode.ErrorOutOfBounds:
                        particle.delete()
                pset.execute([AdvectionRK4,DeleteParticle], 
                             runtime=timedelta(hours=runtime_hours), 
                             dt=timedelta(hours=cfg.dt_hours), 
                             output_file=pfile, 
                             verbose_progress=False)
            finally:
                with contextlib.suppress(Exception):
                    pfile.close()
            log.info("Wrote simulation %s (runtime_hours=%.2f length_days=%d)", out_path, runtime_hours, sim_length_days)
            # Close combined dataset once finished with this reef/year
            with contextlib.suppress(Exception):
                ds_ts.close()

    log.info("All simulations complete")

# ------------------------------ CLI entry --------------------------------


def main(argv: Sequence[str] | None = None) -> int:

    p = argparse.ArgumentParser(description="Particle simulation over daily eReefs subcubes (config-only)")
    p.add_argument('--config', required=True, type=Path, help='Path to TOML config')
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg.log_file)
    run_simulation(cfg)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
