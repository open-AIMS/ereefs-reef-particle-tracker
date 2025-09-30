"""
Multi-Particle Injection Experiments (Script 06)
===============================================

Purpose
-------
Run repeated (potentially overlapping) multi‑particle advection experiments for each reef and year
using previously extracted daily eReefs GBR1 subcubes (script 03 outputs). Each experiment injects
`num_particles` uniformly inside the reef polygon at the beginning of a chosen start date and
advects them for a configured `runtime` (hours). Injections are scheduled at an arbitrary interval in
days (`injection_interval`), derived from the configured `date_periods` span. Only injections that
fully fit within the available date span (start + runtime_days_ceiling - 1 <= last available day)
are launched.

Key Features
------------
* Uniform polygon seeding via rejection sampling in projected CRS (EPSG:3112) for spatial fidelity.
* Deterministic seeding: RNG seed combines (config.random_seed, LABEL_ID, start_date) so adding
  new reefs does not alter existing runs.
* Optional horizontal diffusion (simple 2-D random walk) controlled by `kh_m2s` (m^2/s); 0 disables.
* Overlapping runs allowed (each run self-contained in a dedicated Zarr directory).
* Buffer consistency check (config.buffer_km vs dataset metadata) to avoid stale subcube usage.
* Graceful out-of-bounds deletion (particles removed only when leaving outer subcube domain).
* Atomic Zarr directory creation (write to .tmp then rename) to avoid partial results on interruption.

Explicit Non-Goals (Handled Elsewhere or Deferred)
--------------------------------------------------
* Residence time / survival curve computation (deferred to a future analysis script).
* Animation / plotting (use script 05 or future dedicated visual tools).
* Land / reef re-entry logic (particles are free once seeded; only domain exit deletes).
* Vertical processes or 3-D diffusion (2-D surface advection only).

Configuration Additions (Required / Optional)
--------------------------------------------
Existing (from utils.Config):
	ids, years, date_periods, model_name, depth_m, data_root, traces_root, dt_hours, runtime,
	overwrite, buffer_km, integrator (currently only 'rk4' respected), reef_shp.
New (added in utils for multi-particle runs):
	num_particles (int)        : Number of particles per injection ( >0 required for this script ).
	injection_interval (int)   : Days between successive injections (>=1).
	random_seed (int|None)     : Base seed; combined with reef + start date for deterministic runs.
	kh_m2s (float)             : Horizontal diffusion coefficient (m^2/s); 0 -> disabled.

Output Naming
-------------
Zarr store (one per run):
	{model}_{LABEL_ID}_mparticles_{depthTok}_{YYYYMMDD}_{Nd}_{Np}p.zarr
Where Nd = ceil(runtime_hours/24). The store is placed under:
	traces_root / LABEL_ID / YEAR /

Time Handling
-------------
* Dataset times are assumed UTC (as written by extraction stage); start time for a run is the first
  timestamp of the first daily file for the start day (midnight UTC).
* Runtime is capped silently to available data coverage if extraction span is shorter than requested.
* Exact-multiple safeguard: If runtime is an exact multiple of 24 h (e.g. 72 h), one extra daily file
	is included so that the final hour has supporting velocity data (otherwise max_time - min_time is
	runtime-1, causing an avoidable cap).

Diffusion Model (Simple Approximation)
-------------------------------------
A random walk with per‑component displacement ~ N(0, 2 * K_h * dt_seconds) metres, converted to
geographic degrees using local latitude for longitude scaling (cos φ). This is a light approximation
sufficient for small buffers and short timescales.

Seeding Algorithm
-----------------
1. Extract reef polygon, reproject to EPSG:3112.
2. Rejection sample in projected bounding box until N interior points accepted (boundary inclusive).
3. Transform accepted points back to lon/lat (EPSG:4326) for particle release.

Deterministic RNG Construction
------------------------------
seed_base = config.random_seed (0 if None)
run_seed  = (seed_base ^ hash(label_id) ^ int(start_date.strftime('%Y%m%d'))) & 0xFFFFFFFF

Edge Cases & Safeguards
-----------------------
* If num_particles <= 0: script aborts early.
* If no date_periods specified: exits (needs explicit ranges to bound injections).
* If injection_interval < 1: ValueError.
* If runtime missing or <= 0: exits (multi-run logic requires finite runtime).
* If polygon area extremely small causing repeated rejection failures, raises RuntimeError.

CLI
---
	python 06-multi-particle-flow.py --config path/to/config.toml

Logging summarises each run: path, N particles, runtime hours, diffusion enabled/disabled.

Future Extensions (Not Implemented Yet)
---------------------------------------
* Structured survival curves & residence thresholds output.
* Parallel execution across reefs/years.
* Advanced seeding patterns (Poisson disk, stratified grids).
* Kernel modular selection via config (e.g., choose integrator + diffusion + custom behaviors).
"""
from __future__ import annotations

import argparse
import contextlib
import logging
from math import cos, pi, log, sqrt
from pathlib import Path
from typing import Sequence, List
from datetime import date, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Polygon, Point
from pyproj import Transformer

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, DiffusionUniformKh, Field
import parcels  # for StatusCode & RNG in JIT kernels

from utils import (
	load_config,
	configure_logging,
	expand_date_periods_for_year,
	depth_token,
	multi_particle_simulation_filename,
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def discover_daily_files(data_root: Path, label_id: str, year: int, model_name: str, depth_tok: str) -> List[Path]:
	"""Return sorted list of daily NetCDF paths for one reef/year (depth-aware).

	Pattern mirrors extraction script 03 output naming:
		{model}_{LABEL_ID}_UV_{depthTok}_{YYYYMMDD}.nc
	"""
	year_dir = data_root / label_id / str(year)
	if not year_dir.exists():
		return []
	pattern = f"{model_name}_{label_id}_UV_{depth_tok}_*.nc"
	return sorted(year_dir.glob(pattern))


def map_dates(files: List[Path]) -> dict[date, Path]:
	out: dict[date, Path] = {}
	for p in files:
		try:
			day_token = p.stem.split('_')[-1]
			d = pd.to_datetime(day_token, format='%Y%m%d').date()
			out[d] = p
		except Exception:
			continue
	return out


def open_combined(files: List[Path]) -> xr.Dataset:
	return xr.open_mfdataset([str(p) for p in files], combine='by_coords', decode_times=True, combine_attrs='drop_conflicts')


def _required_run_days(runtime_hours: float) -> int:
	"""Return number of daily files required to supply velocity coverage for the requested runtime.

	Rationale: Daily files contain 24 hourly timestamps 00..23. For a runtime that is an exact
	multiple of 24 h (e.g. 72), three daily files (72/24) provide timestamps spanning 0..71 h; the
	final hour (71->72) would lack a 72 h timestamp boundary. Including one extra daily file ensures
	interpolation support to the requested end time.
	"""
	if runtime_hours <= 0:
		return 0
	days_exact = runtime_hours / 24.0
	if abs(days_exact - round(days_exact)) < 1e-9:  # exact multiple -> add one buffer day
		return int(round(days_exact)) + 1
	return int(np.ceil(days_exact))


def iter_injection_start_dates(all_days: List[date], interval_days: int, runtime_hours: float) -> List[date]:
	"""Return valid injection start dates.

	We require that the full run (ceil(runtime/24) days) fits within the available span, per
	user specification (avoid starting a run whose runtime would overrun the available data span).
	"""
	if not all_days:
		return []
	run_days = _required_run_days(runtime_hours)
	days_set = set(all_days)
	last_day = all_days[-1]
	starts: List[date] = []
	cur = all_days[0]
	while True:
		if cur not in days_set:
			nxt = next((d for d in all_days if d > cur), None)
			if nxt is None:
				break
			cur = nxt
			continue
		if cur + timedelta(days=run_days - 1) <= last_day:
			starts.append(cur)
		else:
			break
		cur = cur + timedelta(days=interval_days)
		if cur > last_day:
			break
	return starts


def build_seed_points(polygon_gdf: gpd.GeoDataFrame, num_particles: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
	"""Return lon, lat arrays of uniformly sampled points inside polygon.

	Reprojects polygon to EPSG:3112, rejection samples inside bounding box, transforms back to EPSG:4326.
	"""
	if num_particles <= 0:
		raise ValueError("num_particles must be > 0")
	poly_gdf = polygon_gdf.to_crs(3112)
	# GeoPandas 0.14+ deprecates unary_union; prefer union_all if present
	geom_series = poly_gdf.geometry
	if hasattr(geom_series, "union_all"):
		geom = geom_series.union_all()
	else:  # fallback for older versions
		geom = geom_series.unary_union  # type: ignore[attr-defined]
	if geom.is_empty:
		raise ValueError("Empty polygon geometry for seeding")
	if isinstance(geom, Polygon):
		polys = [geom]
	else:  # MultiPolygon
		polys = list(geom.geoms)
	minx, miny, maxx, maxy = geom.bounds
	rng = np.random.default_rng(seed)
	xs: List[float] = []
	ys: List[float] = []
	max_attempts = num_particles * 50  # generous for complex shapes
	attempts = 0
	while len(xs) < num_particles and attempts < max_attempts:
		attempts += 1
		x = rng.uniform(minx, maxx)
		y = rng.uniform(miny, maxy)
		# shapely Polygon.covers expects a geometry, not a raw coordinate tuple
		pt = Point(x, y)
		pt_in = any(poly.covers(pt) for poly in polys)
		if pt_in:
			xs.append(x); ys.append(y)
	if len(xs) < num_particles:
		raise RuntimeError(f"Failed to sample {num_particles} points (accepted {len(xs)}) after {attempts} attempts")
	transformer = Transformer.from_crs(3112, 4326, always_xy=True)
	lon, lat = transformer.transform(np.array(xs), np.array(ys))
	return lon.astype(float), lat.astype(float)

def make_delete_kernel():
	def DeleteParticle(particle, fieldset, time):  # noqa: D401
		if particle.state == parcels.StatusCode.ErrorOutOfBounds:
			particle.delete()
	return DeleteParticle

def install_parcels_noise_filter():
	"""Install a logging filter that suppresses the very noisy internal Parcels messages
	about repeated cell search failures ("Correct cell not found ... after 1000000 iterations")
	while still allowing the progress bar (verbose_progress) to be displayed.

	The progress bar writes to stdout (tqdm) and is unaffected by filtering standard log records.
	We filter both the primary line and its follow-up 'Debug info:' lines.
	"""
	class _ParcelsFilter(logging.Filter):  # inner class so it's only defined once
		noisy_fragments = (
			"Correct cell not found for (lat, lon)",
			"Debug info: old particle indices:",
			"Relative particle position:",
		)
		def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
			msg = record.getMessage()
			return not any(frag in msg for frag in self.noisy_fragments)
	root = logging.getLogger()
	# Avoid stacking duplicates
	if not any(isinstance(f, _ParcelsFilter) for f in getattr(root, 'filters', [])):
		root.addFilter(_ParcelsFilter())

# --------------------------------------------------------------------------------------
# Core workflow
# --------------------------------------------------------------------------------------

def run(cfg) -> None:
	log = logging.getLogger(__name__)
	if not cfg.date_periods:
		log.error("date_periods required for multi-particle runs")
		return
	if cfg.num_particles <= 0:
		log.error("num_particles must be > 0 (got %d)", cfg.num_particles)
		return
	if cfg.injection_interval < 1:
		log.error("injection_interval must be >=1 (got %d)", cfg.injection_interval)
		return
	if cfg.runtime is None or cfg.runtime <= 0:
		log.error("runtime (hours) must be positive for multi-particle runs")
		return

	reef_gdf = gpd.read_file(cfg.reef_shp)
	name_map = reef_gdf.drop_duplicates(subset=['LABEL_ID']).set_index('LABEL_ID')['GBR_NAME'].to_dict()
	depth_tok = depth_token(cfg.depth_m)

	for label_id in cfg.ids:
		if label_id not in name_map:
			log.warning("LABEL_ID %s not found in reef layer; skipping", label_id)
			continue
		reef_poly_gdf = reef_gdf[reef_gdf['LABEL_ID'] == label_id]
		for year in cfg.years:
			daily_files = discover_daily_files(cfg.data_root, label_id, year, cfg.model_name, depth_tok)
			if not daily_files:
				log.warning("No daily files for %s %d", label_id, year)
				continue
			date_map = map_dates(daily_files)
			allowed = set(expand_date_periods_for_year(year, cfg.date_periods))
			all_days = sorted(d for d in date_map if d in allowed)
			if not all_days:
				log.warning("No overlapping days after date_periods filter for %s %d", label_id, year)
				continue
			start_dates = iter_injection_start_dates(all_days, cfg.injection_interval, cfg.runtime)
			if not start_dates:
				log.info("No start dates fit runtime for %s %d", label_id, year)
				continue
			log.info("%s %d -> %d injection(s)", label_id, year, len(start_dates))
			for sdate in start_dates:
				runtime_hours_req = cfg.runtime
				run_days = _required_run_days(runtime_hours_req)
				needed_day_set = {sdate + timedelta(days=i) for i in range(run_days)}
				files_for_run = [date_map[d] for d in sorted(needed_day_set) if d in date_map]
				if not files_for_run:
					log.warning("No files for start %s (%s %d)", sdate, label_id, year)
					continue
				try:
					ds_ts = open_combined(files_for_run)
				except Exception as e:  # noqa: BLE001
					log.error("Failed opening dataset for %s %d %s: %s", label_id, year, sdate, e)
					continue
				file_buffer_km = ds_ts.attrs.get('buffer_km')
				if file_buffer_km is None or abs(float(file_buffer_km) - float(cfg.buffer_km)) > 1e-6:
					log.error("buffer_km mismatch (cfg=%s file=%s) for %s %d", cfg.buffer_km, file_buffer_km, label_id, year)
					ds_ts.close(); continue
				for req in ('longitude','latitude','time','u','v'):
					if req not in ds_ts:
						log.error("Missing variable %s in dataset (%s %d)", req, label_id, year)
						ds_ts.close(); continue
				lon_grid = ds_ts['longitude'].values
				lat_grid = ds_ts['latitude'].values
				times_all = pd.to_datetime(ds_ts['time'].values)
				t_start = times_all.min()
				t_end_full = times_all.max()
				available_hours = (t_end_full - t_start).total_seconds() / 3600.0
				if runtime_hours_req > available_hours + 1e-6:
					log.warning("Requested runtime %.2f h exceeds available span %.2f h (capping)", runtime_hours_req, available_hours)
				runtime_hours = min(runtime_hours_req, available_hours)
				if runtime_hours <= 0:
					log.warning("Non-positive runtime after capping for %s %d run %s", label_id, year, sdate)
					ds_ts.close(); continue
				sim_length_days = int(np.ceil(runtime_hours / 24.0))
				out_path = multi_particle_simulation_filename(cfg.traces_root, cfg.model_name, label_id, cfg.depth_m, sdate, sim_length_days, cfg.num_particles)
				if out_path.exists() and not cfg.overwrite:
					log.info("Exists (skip): %s", out_path)
					ds_ts.close(); continue
				if out_path.exists() and cfg.overwrite:
					import shutil
					shutil.rmtree(out_path, ignore_errors=True)
				out_path.parent.mkdir(parents=True, exist_ok=True)
				base_seed = cfg.random_seed if cfg.random_seed is not None else 0
				run_seed = (base_seed ^ hash(label_id) ^ int(sdate.strftime('%Y%m%d'))) & 0xFFFFFFFF
				try:
					lon0, lat0 = build_seed_points(reef_poly_gdf, cfg.num_particles, run_seed)
				except Exception as e:  # noqa: BLE001
					log.error("Seeding failed %s %d %s: %s", label_id, year, sdate, e)
					ds_ts.close(); continue
				u_vals = ds_ts['u'].transpose('time','j','i').values
				v_vals = ds_ts['v'].transpose('time','j','i').values
				data = {'U': u_vals, 'V': v_vals}
				dimensions = {
					'U': {'lon': lon_grid, 'lat': lat_grid, 'time': times_all.values},
					'V': {'lon': lon_grid, 'lat': lat_grid, 'time': times_all.values},
				}
				fieldset = FieldSet.from_data(data=data, dimensions=dimensions, mesh='spherical', allow_time_extrapolation=False)
				if cfg.kh_m2s > 0:
					# Provide uniform diffusion as Fields so DiffusionUniformKh can index them.
					kh_array = np.full(lon_grid.shape, cfg.kh_m2s, dtype='float32')
					Kh_zonal = Field(name='Kh_zonal', data=kh_array, lon=lon_grid, lat=lat_grid, mesh='spherical')
					Kh_meridional = Field(name='Kh_meridional', data=kh_array, lon=lon_grid, lat=lat_grid, mesh='spherical')
					fieldset.add_field(Kh_zonal)
					fieldset.add_field(Kh_meridional)
				class MParticle(JITParticle):
					pass
				pset = ParticleSet.from_list(fieldset=fieldset, pclass=MParticle, lon=list(lon0), lat=list(lat0), time=t_start.to_pydatetime())
				with contextlib.suppress(Exception):
					pset.populate_indices()
				kernels = [AdvectionRK4]
				if cfg.kh_m2s > 0:
					kernels.append(DiffusionUniformKh)
				kernels.append(make_delete_kernel())
				# Atomic write directory logic:
				# Previous approach appended '.tmp' after the final '.zarr' (foo.zarr.tmp); Parcels then
				# auto-added '.zarr' again, yielding foo.zarr.tmp.zarr and a failed rename. Here we ensure
				# the temporary directory already ends with '.zarr' so Parcels does not append another.
				tmp_path = out_path.parent / f"{out_path.stem}.tmp.zarr"  # stem strips trailing .zarr
				if tmp_path.exists():
					import shutil
					shutil.rmtree(tmp_path, ignore_errors=True)
				output_dt = timedelta(hours=cfg.dt_hours)
				pfile = pset.ParticleFile(name=str(tmp_path), outputdt=output_dt)
				try:
					# Install filter to suppress noisy repeated cell-search debug lines while keeping progress bar.
					install_parcels_noise_filter()
					pset.execute(
						kernels,
						runtime=timedelta(hours=runtime_hours),
						dt=timedelta(hours=cfg.dt_hours),
						output_file=pfile,
						verbose_progress=True,  # progress bar re-enabled
					)
				finally:
					with contextlib.suppress(Exception):
						pfile.close()
				try:
					if out_path.exists():
						import shutil
						shutil.rmtree(out_path, ignore_errors=True)
					tmp_path.rename(out_path)
				except Exception as e:  # noqa: BLE001
					log.error("Atomic rename failed for %s: %s", out_path, e)
				log.info("Run complete %s (N=%d runtime=%.2f h diffusion=%s)", out_path, cfg.num_particles, runtime_hours, 'on' if cfg.kh_m2s>0 else 'off')
				with contextlib.suppress(Exception):
					ds_ts.close()
	log.info("All multi-particle runs complete")


def parse_args(argv: Sequence[str] | None = None):
	p = argparse.ArgumentParser(description="Multi-particle injection experiments (config-driven)")
	p.add_argument('--config', required=True, type=Path, help='Path to TOML config file')
	p.add_argument('--log-file', type=Path, default=None, help='Override log file (else config log_file)')
	return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	cfg = load_config(args.config)
	log_file = args.log_file if args.log_file else cfg.log_file
	configure_logging(log_file)
	run(cfg)
	return 0


if __name__ == '__main__':  # pragma: no cover
	raise SystemExit(main())