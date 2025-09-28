"""
Create a minimal OceanParcels simulation over eReefs GBR1 weekly subcubes using a prebuilt f-node grid.

This script demonstrates the basic setup and execution of a particle simulation using the OceanParcels
library, leveraging the eReefs GBR1 dataset. It focuses on a single weekly subcube and uses a precomputed
f-node grid for accurate C-grid interpolation.

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
from dataclasses import dataclass
from typing import List, Sequence
from datetime import timedelta, date
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd  # reef metadata

from utils import (
    configure_logging,
    sanitize_name,
    load_config,
    expand_date_periods_for_year,
    depth_token,
    simulation_filename,
)

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
try:  # noqa: SIM105
    from parcels.tools.error import ErrorCode  # type: ignore
except Exception:  # noqa: BLE001
    ErrorCode = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class SubcubeMeta:
    path: Path
    lon: np.ndarray
    lat: np.ndarray
    k_index: int
    times: np.ndarray  # datetime64


# ------------------------------ Helpers ---------------------------------

## depth_token now provided by utils

def discover_daily_files(data_root: Path, label_id: str, year: int, model_name: str, depth_tok: str) -> List[Path]:
    year_dir = data_root / label_id / str(year)
    if not year_dir.exists():
        return []
    pattern = f"{model_name}_{label_id}_UV_*{year}*.nc"  # fallback broad pattern; refined filter applied below
    files = sorted(year_dir.glob(pattern))
    return files

def load_subcube_meta(path: Path) -> SubcubeMeta:
    try:
        ds = xr.open_dataset(path, decode_times=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Failed to open subcube {path}: {e}") from e
    for v in ("longitude", "latitude", "time"):
        if v not in ds:
            raise SystemExit(f"Subcube {path} missing variable {v}")
    if 'k_index' not in ds.attrs:
        # Accept absence; use -1 sentinel
        k_index = int(ds.attrs.get('k_index', -1))
    else:
        k_index = int(ds.attrs['k_index'])
    meta = SubcubeMeta(
        path=path,
        lon=ds['longitude'].values,
        lat=ds['latitude'].values,
        k_index=k_index,
        times=ds['time'].values,
    )
    ds.close()
    return meta

def build_fieldset(daily_files: List[Path], first_meta: SubcubeMeta) -> FieldSet:
    ds_list = [xr.open_dataset(p, decode_times=True) for p in daily_files]
    try:
        combined = xr.concat(ds_list, dim='time')
        # Heuristic U/V variable names
        cand_u = [v for v in combined.data_vars if v.lower().startswith('u')]
        cand_v = [v for v in combined.data_vars if v.lower().startswith('v')]
        if not cand_u or not cand_v:
            raise SystemExit("Cannot identify U/V variables in daily files")
        u_name, v_name = cand_u[0], cand_v[0]
        u_vals = combined[u_name].transpose('time','j','i').values
        v_vals = combined[v_name].transpose('time','j','i').values
        time_vals = combined['time'].values
    finally:
        for d in ds_list:
            d.close()
    data = {'U': u_vals, 'V': v_vals}
    dimensions = {
        'U': {'lon': first_meta.lon, 'lat': first_meta.lat, 'time': time_vals},
        'V': {'lon': first_meta.lon, 'lat': first_meta.lat, 'time': time_vals},
    }
    return FieldSet.from_data(data=data, dimensions=dimensions, mesh='spherical', allow_time_extrapolation=False)



# ------------------------------- Core ------------------------------------

def run_simulation(cfg) -> None:
    log = logging.getLogger(__name__)
    # Load reef names once
    reef_shp = Path("data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp")
    if not reef_shp.exists():
        raise SystemExit(f"Reef shapefile missing: {reef_shp}")
    reefs = gpd.read_file(reef_shp)
    ref = reefs.drop_duplicates(subset=['LABEL_ID']).set_index('LABEL_ID')['GBR_NAME'].to_dict()
    depth_tok = depth_token(cfg.depth_m)

    for label_id in cfg.ids:
        if label_id not in ref:
            log.warning("LABEL_ID %s not found in reef layer; skipping", label_id)
            continue
        for year in cfg.years:
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
            if cfg.date_periods:
                days_allowed = set(expand_date_periods_for_year(year, cfg.date_periods))
                selected_days = sorted(d for d in date_map if d in days_allowed)
            else:
                selected_days = sorted(date_map)
            if not selected_days:
                log.warning("No days selected after period filtering for %s %d", label_id, year)
                continue
            selected_files = [date_map[d] for d in selected_days]
            meta = load_subcube_meta(selected_files[0])
            # Build fieldset
            fieldset = build_fieldset(selected_files, meta)
            # Start/end times
            ds_cat = xr.open_mfdataset([str(p) for p in selected_files], combine='by_coords')
            try:
                times = pd.to_datetime(ds_cat['time'].values)
                t_start = times.min()
                t_end_full = times.max()
            finally:
                ds_cat.close()
            runtime_hours = cfg.runtime if cfg.runtime is not None else (t_end_full - t_start).total_seconds()/3600.0
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
            lon_arr, lat_arr = meta.lon, meta.lat
            mask = np.isfinite(lon_arr) & np.isfinite(lat_arr)
            if not mask.any():
                log.error("All lon/lat NaN for %s %d", label_id, year)
                continue
            js, is_ = np.where(mask)
            jmid = int((js.min()+js.max())//2)
            imid = int((is_.min()+is_.max())//2)
            lon0 = float(lon_arr[jmid, imid]); lat0 = float(lat_arr[jmid, imid])
            class Particle(JITParticle):
                pass
            pset = ParticleSet.from_list(fieldset=fieldset, pclass=Particle, lon=[lon0], lat=[lat0], time=t_start.to_pydatetime())
            with contextlib.suppress(Exception):
                pset.populate_indices()
            # Domain constants for simple clamp kernel
            finite_lon = lon_arr[mask]; finite_lat = lat_arr[mask]
            for k,v in {
                'DOM_MINLON': float(np.nanmin(finite_lon)),
                'DOM_MAXLON': float(np.nanmax(finite_lon)),
                'DOM_MINLAT': float(np.nanmin(finite_lat)),
                'DOM_MAXLAT': float(np.nanmax(finite_lat)),
            }.items():
                try:
                    fieldset.add_constant(k, v)
                except Exception:
                    pass
            def Clamp(particle, fieldset, time):  # noqa: D401
                if particle.lon < fieldset.DOM_MINLON: particle.lon = fieldset.DOM_MINLON
                elif particle.lon > fieldset.DOM_MAXLON: particle.lon = fieldset.DOM_MAXLON
                if particle.lat < fieldset.DOM_MINLAT: particle.lat = fieldset.DOM_MINLAT
                elif particle.lat > fieldset.DOM_MAXLAT: particle.lat = fieldset.DOM_MAXLAT
            if cfg.integrator == 'euler':
                from parcels import AdvectionEE
                base_adv = AdvectionEE
            else:
                base_adv = AdvectionRK4
            kernel = pset.Kernel(base_adv) + pset.Kernel(Clamp)
            # Use configured timestep for both integration (dt) and output cadence
            output_dt = timedelta(hours=cfg.dt_hours)
            log.debug("Simulation timesteps: integration=%.3f h output=%.3f h", cfg.dt_hours, cfg.dt_hours)
            pfile = pset.ParticleFile(name=str(out_path), outputdt=output_dt)
            try:
                recovery_kwargs = {}
                if ErrorCode is not None:
                    try: recovery_kwargs['recovery'] = {ErrorCode.ErrorOutOfBounds: lambda p, f, t: p.delete()}
                    except Exception: pass
                pset.execute(kernel, runtime=timedelta(hours=runtime_hours), dt=timedelta(hours=cfg.dt_hours), output_file=pfile, verbose_progress=False, **recovery_kwargs)
            finally:
                with contextlib.suppress(Exception):
                    pfile.close()
            log.info("Wrote simulation %s (runtime_hours=%.2f length_days=%d)", out_path, runtime_hours, sim_length_days)

    log.info("All simulations complete")

# ------------------------------ CLI entry --------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Particle simulation over daily eReefs subcubes (config-only)")
    p.add_argument('--config', required=True, type=Path, help='Path to TOML config')
    return p.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(cfg.log_file)
    run_simulation(cfg)
    return 0

if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
