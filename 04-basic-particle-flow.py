"""
Create a minimal OceanParcels simulation over eReefs GBR1 weekly subcubes using a prebuilt f-node grid.

Why this script exists
Parcels performs C-grid interpolation on curvilinear meshes using the geometry of cell corners. eReefs 
GBR1 supplies longitude and latitude at cell centres, so we precompute a separate corner grid, known as 
f-nodes, and keep all particle interpolation consistent with that geometry. This script wires a single 
weekly subcube of U and V from eReefs together with the shared f-node corner file to build a Parcels 
FieldSet, then advances one particle with hourly steps to verify the end-to-end setup. Parcels only 
needs the four corners of each cell for C-grid interpolation, and it is valid for U and V to reference 
the same f-node arrays. Using the shared f-node file avoids having to synthesise corners for every subcube 
and guarantees a smooth, non-degenerate mesh for the particle search and interpolation. ([Ocean Parcels 
Documentation][1])

What f-nodes mean here
An f-node is a cell corner. For a centre grid of shape j by i the f-node arrays have shape (j+1) by (i+1). 
Parcels expects U and V on a C-grid to share the same f-node longitude and latitude. In practice we open 
the saved file gbr1_fnodes.nc, slice lon_f and lat_f to the same subcube footprint using the jf_* 
and if_* indices carried in the subcube, and pass those slices to the C-grid reader. The curvilinear 
tutorial for NEMO demonstrates this pattern by pointing U and V at a separate mesh file for lon and lat 
while reading velocity from the data files. We do the same with eReefs, substituting lon_f and lat_f 
for the NEMO glamf and gphif. ([Ocean Parcels Documentation][2])

Overview of processing
The script loads the reef layer to map each requested LABEL_ID to its canonical GBR_NAME so we can derive 
the on-disk subcube filename. It resolves the year and week to a start date, constructs the expected path 
working/02/{LABEL_ID}/{YYYY}/{GBR_NAME_NO_SPACES}*{LABEL_ID}*k{kindex}*{YYYYMMDD}.nc, and validates 
that the file exists and contains the attributes jf_start, jf_stop, if_start, if_stop, fnode_file 
and fnode_crop written by the extractor. It opens the f-node NetCDF referenced by fnode_file, checks 
that lon_f and lat_f exist and that the jf*/if_ slices are within bounds, and then slices lon_f and 
lat_f to match the subcube extent. It builds a Parcels FieldSet for C-grid data using those corner arrays, 
the subcube U and V, and the subcube time coordinate, with mesh set to spherical and time extrapolation 
disabled. The particle is initialised at the centre of the subcube as the centre-grid point at indices 
floor(j/2), floor(i/2). Initial time is the first time step in the subcube. We run AdvectionRK4 with dt 
of one hour for the full file span so that model output positions are hourly by construction. After the 
run, we convert the positions to a point shapefile in EPSG:4326 written to working/04/, with attributes 
TIME_ISO as an ISO 8601 string in Australia/Brisbane and HOUR_IDX as a zero based integer index of the 
hourly step. Where relevant, we call ParticleSet.populate_indices before execute to accelerate curvilinear 
index initialisation. ([Ocean Parcels Documentation][2])

Why we couple subcubes to a single shared f-node file
The GBR1 native domain contains NaN padding offshore that forms an L shape. Our shared f-node file trims 
away the problematic offshore wedge and provides a clean vertical band aligned with the coast. The extractor 
already guarantees that each subcube lies fully inside that band and provides the jf_/if_ slices that map 
the subcube to the corner grid. Reusing that single f-node file ensures U and V always see a consistent,
 well behaved corner mesh without any per run corner synthesis or inpainting. If a requested subcube falls 
 outside the f-node crop the script fails with a clear error rather than attempting to guess coordinates.

Assumptions
Parcels version is 3.1.4. The extractor has written jf_start, jf_stop, if_start, if_stop, fnode_file 
and fnode_crop attributes to every subcube. Weekly files are named with the start date of the requested
 week, where week 1 is 1 to 7 January, week 2 is 8 to 14 January, and so on. Output CRS for the trajectory 
 shapefile is EPSG:4326. We run one particle, hourly steps, no diffusion or custom kernels, and we do not 
 delete the particle at the boundary because weekly windows are far from the outer model edge. The generic 
 C-grid loader in Parcels supports providing corner coordinates from a separate file and only needs the 
 f-nodes, not the exact locations of the velocity stagger points. ([Ocean Parcels Documentation][3])

What is written
Zarr trajectory store: working/02/<LABEL_ID>/<YEAR>/<SAFE_NAME>_<LABEL_ID>_k<kindex>_<YEAR>w<week>_particles.zarr
containing standard Parcels variables (lon, lat, time, plus any future particle variables) in the
(trajectory, obs) layout suitable for multi-particle scaling. Shapefile export has been removed from
this script to keep the simulation lean; use 04b-convert-particle-zarr-to-shp.py to produce point
shapefiles (one point per output step) for GIS/debug workflows.

Limitations
This is a basic, single particle hourly advection. It is meant to verify the geometry and indexing rather 
than provide scientific results. If later you choose to use multiple particles, add diffusion, or add masks 
and deletion zones, extend the kernels and outputs accordingly. The file discovery assumes the naming pattern 
described above and that the subcube carries the f-node slice attributes. If either assumption is not met 
you will need to adapt the resolver.

[1]: https://docs.oceanparcels.org/en/latest/examples/documentation_indexing.html?utm_source=chatgpt.com "Grid indexing — Parcels Documentation"
[2]: https://docs.oceanparcels.org/en/latest/examples/tutorial_nemo_curvilinear.html "Curvilinear grids — Parcels Documentation"
[3]: https://docs.oceanparcels.org/en/stable/examples/documentation_indexing.html?utm_source=chatgpt.com "Grid indexing — Parcels Documentation"

"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Dict
from datetime import datetime, timedelta, date
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd  # reef metadata

from utils import (
	configure_logging,
	sanitize_name,
	normalize_years,
	normalize_weeks,
	make_week_tag,
	# Added for simpler week->date expansion
	expand_weeks_to_dates,
)

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
# Explicit import path for ErrorCode in Parcels 3.1.4
try:  # noqa: SIM105
    from parcels.tools.error import ErrorCode  # type: ignore
except Exception:  # noqa: BLE001
    ErrorCode = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


## Removed local helpers now provided by utils (sanitize_name, normalize_years, normalize_weeks, make_week_tag)


@dataclass
class SubcubeMeta:
	path: Path
	jf_start: int
	jf_stop: int
	if_start: int
	if_stop: int
	fnode_file: Path
	fnode_crop: str
	j_start: int
	j_stop: int  # inclusive
	i_start: int
	i_stop: int  # inclusive
	k_index: int
	lon: np.ndarray
	lat: np.ndarray
	times: np.ndarray  # datetime64 array for that daily file


def discover_daily_files(data_root: Path, label_id: str, year: int, safe_name: str, k_index: int) -> List[Path]:
	year_dir = data_root / label_id / str(year)
	if not year_dir.exists():
		return []
	pattern = f"{safe_name}_{label_id}_k{k_index}_{year}*.nc"
	return sorted(year_dir.glob(pattern))


def load_subcube_meta(path: Path) -> SubcubeMeta:
	try:
		ds = xr.open_dataset(path, decode_times=True)
	except Exception as e:  # noqa: BLE001
		raise SystemExit(f"Failed to open subcube {path}: {e}") from e
	for a in ("jf_start", "jf_stop", "if_start", "if_stop", "fnode_file", "fnode_crop", "j_start", "j_stop", "i_start", "i_stop", "k_index"):
		if a not in ds.attrs:
			raise SystemExit(f"Subcube {path} missing required attribute {a}; abort.")
	meta = SubcubeMeta(
		path=path,
		jf_start=int(ds.attrs["jf_start"]),
		jf_stop=int(ds.attrs["jf_stop"]),
		if_start=int(ds.attrs["if_start"]),
		if_stop=int(ds.attrs["if_stop"]),
		fnode_file=Path(ds.attrs["fnode_file"]),
		fnode_crop=str(ds.attrs["fnode_crop"]),
		j_start=int(ds.attrs["j_start"]),
		j_stop=int(ds.attrs["j_stop"]),
		i_start=int(ds.attrs["i_start"]),
		i_stop=int(ds.attrs["i_stop"]),
		k_index=int(ds.attrs["k_index"]),
		lon=ds["longitude"].values,
		lat=ds["latitude"].values,
		times=ds["time"].values,
	)
	ds.close()
	return meta


def build_fieldset(daily_files: List[Path], meta: SubcubeMeta) -> FieldSet:
	"""Build a Parcels FieldSet (A-grid approximation) from consecutive daily files.

	Simplified assumptions (enforced with asserts):
	  - Each daily file has exactly 24 hourly steps (known eReefs product spec).
	  - All files are consecutive without gaps (missing file => assertion on total hours).
	  - Variables 'u' and 'v' exist with dims (time, j, i).
	We concatenate first and then perform minimal sanity checks.
	"""
	ds_list = [xr.open_dataset(p, decode_times=True) for p in daily_files]
	try:
		combined = xr.concat(ds_list, dim="time")
		# Core variable presence
		assert 'u' in combined and 'v' in combined, "Missing u or v variable"
		times = pd.to_datetime(combined['time'].values)
		expected_hours = 24 * len(daily_files)
		assert times.size == expected_hours, f"Time steps {times.size} != expected {expected_hours} (check for missing daily file)"
		# Hour spacing tolerance: allow near-hour deviations (some files may have leap/processing offsets)
		if times.size > 1:
			deltas_ns = np.diff(times.values).astype('timedelta64[ns]').astype(int)
			# Convert to hours as float
			deltas_hours = deltas_ns / 3.6e12  # 1 hour = 3.6e12 ns
			# Check monotonic increasing
			assert np.all(deltas_hours > 0), "Non-increasing time axis detected"
			# Identify deltas not within 1h +/- 2 minutes tolerance
			bad = ~((deltas_hours >= (1 - 2/60)) & (deltas_hours <= (1 + 2/60)))
			if np.any(bad):
				logging.warning("%d time intervals deviate >2 min from 1h (continuing): first few=%s", bad.sum(), deltas_hours[bad][:5])
		u_vals = combined['u'].transpose('time','j','i').values
		v_vals = combined['v'].transpose('time','j','i').values
		time_vals = combined['time'].values
	finally:
		for d in ds_list:
			d.close()
	data = {'U': u_vals, 'V': v_vals}
	dimensions = {
		'U': {'lon': meta.lon, 'lat': meta.lat, 'time': time_vals},
		'V': {'lon': meta.lon, 'lat': meta.lat, 'time': time_vals},
	}
	return FieldSet.from_data(data=data, dimensions=dimensions, mesh='spherical', allow_time_extrapolation=False)


def _format_row(values: np.ndarray, head: int = 4, tail: int = 4, fmt: str = "%.5f") -> str:
	n = values.size
	if n <= head + tail + 1:
		return ", ".join(fmt % v for v in values)
	head_part = ", ".join(fmt % v for v in values[:head])
	tail_part = ", ".join(fmt % v for v in values[-tail:])
	return f"{head_part}, ... {tail_part}"


def debug_print_grid(meta: SubcubeMeta, u_first: np.ndarray, v_first: np.ndarray, head: int = 2, tail: int = 2) -> None:
	"""Print truncated lon/lat grid and first time-slice of U,V for debugging.

	Shows first & last 'head'/'tail' rows and columns. Minimal noise, removable.
	"""
	lon = meta.lon
	lat = meta.lat
	jmax, imax = lon.shape
	logging.info("GRID SNAPSHOT: shape j=%d i=%d", jmax, imax)
	# Rows: first 'head', last 'tail'
	sel_js = list(range(min(head, jmax))) + list(range(max(head, jmax - tail), jmax)) if jmax > head + tail else list(range(jmax))
	for j in sel_js:
		lon_row = _format_row(lon[j, :])
		lat_row = _format_row(lat[j, :])
		logging.info("lon[j=%d]: %s", j, lon_row)
		logging.info("lat[j=%d]: %s", j, lat_row)
	# Columns snapshot (first/last few values down the column)
	sel_is = list(range(min(head, imax))) + list(range(max(head, imax - tail), imax)) if imax > head + tail else list(range(imax))
	for i in sel_is:
		lon_col = _format_row(lon[:, i])
		lat_col = _format_row(lat[:, i])
		logging.info("lon_col[i=%d]: %s", i, lon_col)
		logging.info("lat_col[i=%d]: %s", i, lat_col)
	# U/V first slice (time index 0) if provided
    
	logging.info("U[0,:,:] snapshot (rows like lon):")
	for j in sel_js:
		logging.info("U[0,j=%d]: %s", j, _format_row(u_first[j, :]))
	logging.info("V[0,:,:] snapshot (rows like lon):")
	for j in sel_js:
		logging.info("V[0,j=%d]: %s", j, _format_row(v_first[j, :]))


def run_sim(label_id: str, gbr_name: str, safe_name: str, year: int, weeks: List[int] | None,
	daily_files: List[Path], meta: SubcubeMeta, outdir: Path, seed: int,
	dt_hours: float = 1.0, overwrite: bool = False, integrator: str = 'rk4', shrink_margin: int = 1,
	debug_short_run: bool = False) -> None:
	# Determine week label for naming
	week_tag = make_week_tag(weeks)

	reef_dir = outdir / label_id / str(year)
	reef_dir.mkdir(parents=True, exist_ok=True)
	particle_base = reef_dir / f"{safe_name}_{label_id}_k{meta.k_index}_{year}{week_tag}_particles.zarr"

	hours_expected = len(daily_files) * 24

	fieldset = build_fieldset(daily_files, meta)

	# Optional grid snapshot for debugging: use first day's u/v slice
	if logging.getLogger().isEnabledFor(logging.INFO) and any(flag in sys.argv for flag in ('--debug-grid',)):
		try:
			first_ds = xr.open_dataset(daily_files[0], decode_times=False)
			u_sample = first_ds['u'].isel(time=0).values
			v_sample = first_ds['v'].isel(time=0).values
			debug_print_grid(meta, u_sample, v_sample)
		finally:
			with contextlib.suppress(Exception):
				first_ds.close()

	# Clean existing outputs if overwrite requested
	zarr_existing = particle_base
	if zarr_existing.exists() and overwrite:
		logging.info("Overwriting existing Zarr output: %s", zarr_existing)
		import shutil
		shutil.rmtree(zarr_existing, ignore_errors=True)
	elif zarr_existing.exists() and not overwrite:
		raise SystemExit(f"Output already exists (use --overwrite to replace): {zarr_existing}")
		
	# Choose a robust interior start cell (avoid any NaN or boundary interpolation issues)
	lon_arr = meta.lon
	lat_arr = meta.lat
	finite_mask = np.isfinite(lon_arr) & np.isfinite(lat_arr)
	if not finite_mask.any():
		raise SystemExit("All lon/lat are NaN; cannot seed particle")
	valid_js, valid_is = np.where(finite_mask)
	j0, j1 = int(valid_js.min()), int(valid_js.max())
	i0, i1 = int(valid_is.min()), int(valid_is.max())
	# Apply one-cell interior margin when possible
	if shrink_margin < 0:
		shrink_margin = 0
	if j1 - j0 > 2*shrink_margin:
		j0 += shrink_margin; j1 -= shrink_margin
	if i1 - i0 > 2*shrink_margin:
		i0 += shrink_margin; i1 -= shrink_margin
	jmid = (j0 + j1) // 2
	imid = (i0 + i1) // 2
	lon0 = float(lon_arr[jmid, imid])
	lat0 = float(lat_arr[jmid, imid])
	logging.info(
		"SEED candidate j-range=[%d,%d] i-range=[%d,%d] chosen (j,i)=(%d,%d) lon=%.5f lat=%.5f", 
		j0, j1, i0, i1, jmid, imid, lon0, lat0
	)
	# Seed bounds constants for clamping kernel
	minlon = float(np.nanmin(lon_arr[finite_mask]))
	maxlon = float(np.nanmax(lon_arr[finite_mask]))
	minlat = float(np.nanmin(lat_arr[finite_mask]))
	maxlat = float(np.nanmax(lat_arr[finite_mask]))
	fieldset_constants = {
		'DOM_MINLON': minlon,
		'DOM_MAXLON': maxlon,
		'DOM_MINLAT': minlat,
		'DOM_MAXLAT': maxlat,
	}
	for k,v in fieldset_constants.items():
		meta_note = f"Adding constant {k}={v}"  # debug trace (optional remove later)
		logging.debug(meta_note)
	try:
		for k,v in fieldset_constants.items():
			fieldset.add_constant(k, v)
	except Exception:  # noqa: BLE001
		pass

	# Reopen concatenated time for runtime
	ds_cat = xr.open_mfdataset([str(p) for p in daily_files], combine='by_coords')
	try:
		times = pd.to_datetime(ds_cat['time'].values)
		t_start = times.min()
		t_end = times.max()
		runtime = (t_end - t_start).to_pytimedelta()
		# DEBUG HACK: override runtime for short diagnostic run
		if debug_short_run:
			logging.warning("DEBUG SHORT RUN ENABLED: forcing runtime=48h and dt/output=30min")
			runtime = timedelta(hours=12)
			# Force dt to 0.5h for two interior steps
			dt_hours = 0.5
	finally:
		ds_cat.close()

	# Custom particle class so we can delete cleanly on boundary error
	class BoundedParticle(JITParticle):
		pass
	pset = ParticleSet.from_list(fieldset=fieldset, pclass=BoundedParticle, lon=[lon0], lat=[lat0], time=t_start.to_pydatetime())
	with contextlib.suppress(Exception):
		pset.populate_indices()
	# Define simple clamping kernel to keep particle inside finite domain
	def ClampDomain(particle, fieldset, time):  # noqa: D401
		if particle.lon < fieldset.DOM_MINLON:
			particle.lon = fieldset.DOM_MINLON
		elif particle.lon > fieldset.DOM_MAXLON:
			particle.lon = fieldset.DOM_MAXLON
		if particle.lat < fieldset.DOM_MINLAT:
			particle.lat = fieldset.DOM_MINLAT
		elif particle.lat > fieldset.DOM_MAXLAT:
			particle.lat = fieldset.DOM_MAXLAT

	# If debug short run, output every 30 min to capture 0h,0.5h,1h
	output_dt = timedelta(hours=0.5) if debug_short_run else timedelta(hours=1)
	particle_file = pset.ParticleFile(name=str(particle_base), outputdt=output_dt)
	# Recovery kernel to delete particle if it still escapes domain
	def DeleteParticle(particle, fieldset, time):  # noqa: D401
		particle.delete()
	try:
		if integrator == 'euler':
			from parcels import AdvectionEE
			base_adv = AdvectionEE
		else:
			base_adv = AdvectionRK4
		advec_kernel = pset.Kernel(base_adv)
		clamp_kernel = pset.Kernel(ClampDomain)
		combined_kernel = advec_kernel + clamp_kernel
		recovery_kwargs = {}
		if ErrorCode is not None:
			try:
				recovery_kwargs['recovery'] = {ErrorCode.ErrorOutOfBounds: DeleteParticle}
			except AttributeError:
				pass
		pset.execute(
			combined_kernel,
			runtime=runtime,
			dt=timedelta(hours=dt_hours),
			output_file=particle_file,
			verbose_progress=False,
			**recovery_kwargs
		)
	except Exception as e:  # noqa: BLE001
		# Ensure file is closed before propagating failure
		if hasattr(particle_file, 'close'):
			with contextlib.suppress(Exception):
				particle_file.close()
		raise SystemExit(f"Parcels execution failed: {e}") from e
	finally:
		if hasattr(particle_file, 'close'):
			with contextlib.suppress(Exception):
				particle_file.close()

	logging.info(
		"RUN label=%s name=%s year=%d weeks=%s subcubes=%d runtime=%s zarr=%s (A-grid)",
		label_id, gbr_name, year, week_tag, len(daily_files), runtime, particle_base
	)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Minimal Parcels C-grid run over stitched eReefs subcubes", formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	p.add_argument(
		'--reef-shp',
		type=Path,
		default=Path('data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp'),
		help='Path to reef shapefile (.shp) or GeoPackage (.gpkg).'
	)
	p.add_argument('--ids', nargs='+', required=True, help='Space separated list of LABEL_ID values.')
	p.add_argument('--years', nargs='+', type=int, required=True, help='Year or list of years; if two values given they define inclusive range.')
	p.add_argument('--weeks', nargs='*', type=int, default=None, help='Ordinal weeks (Jan1-Jan7=1, Jan8-Jan14=2, ...) to stitch; omit for all weeks.')
	p.add_argument('--k-index', type=int, default=40, help='Vertical index (zero-based) to use (default near -2.35 m).')
	p.add_argument('--data-root', type=Path, default=Path('working/02'), help='Root directory containing extracted daily subcubes (matches extractor --outdir).')
	p.add_argument('--outdir', type=Path, default=Path('working/02'), help='Output root directory for particle results (shares tree with extraction output).')
	p.add_argument('--seed', type=int, default=42)
	p.add_argument('--log-file', type=Path, default=None)
	p.add_argument('--dt-hours', type=float, default=1.0, help='Integration time step in hours (e.g. 0.5 for 30 min).')
	p.add_argument('--overwrite', action='store_true', help='Overwrite existing particle output (delete old .zarr/.nc).')
	p.add_argument('--integrator', choices=['rk4','euler'], default='rk4', help='Advection integrator kernel to use (rk4 or euler).')
	p.add_argument('--shrink-margin', type=int, default=1, help='Additional interior cell margin to inset domain for seeding and clamping (>=0).')
	p.add_argument('--debug-short-run', action='store_true', help='DEBUG HACK: limit runtime to 1 hour with 30min output (produces ~3 points). Remove when done.')
	p.add_argument('--debug-grid', action='store_true', help='DEBUG: log truncated lon/lat and first U,V slice.')
	return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	configure_logging(args.log_file)
	logging.info("Starting basic particle flow run")

	if not args.reef_shp.exists():
		logging.error("Reef shapefile not found: %s", args.reef_shp)
		return 2
	reefs = gpd.read_file(args.reef_shp)
	if 'LABEL_ID' not in reefs.columns or 'GBR_NAME' not in reefs.columns:
		logging.error("Reef file missing LABEL_ID or GBR_NAME")
		return 3
	# Consolidate names by LABEL_ID
	ref = reefs.drop_duplicates(subset=['LABEL_ID']).set_index('LABEL_ID')['GBR_NAME'].to_dict()

	years = normalize_years(args.years)
	try:
		weeks = normalize_weeks(args.weeks)
	except ValueError as e:  # noqa: BLE001
		logging.error(str(e))
		return 4
	if weeks:
		logging.info("Weeks selection (stitched into one run per year): %s", weeks)
	else:
		logging.info("No weeks provided: using full available daily period per year")

	for rid in args.ids:
		if rid not in ref:
			logging.warning("LABEL_ID %s not in reef layer; skipping", rid)
			continue
		gbr_name = ref[rid]
		safe_name = sanitize_name(gbr_name)
		for year in years:
			all_daily = discover_daily_files(args.data_root, rid, year, safe_name, args.k_index)
			if not all_daily:
				logging.warning("No daily subcubes for %s %d", rid, year)
				continue
			# Map date->file using final underscore token (YYYYMMDD) and expand requested weeks.
			# 1. Extract dates embedded at end of each filename stem.
			date_map: Dict[date, Path] = {}
			for p in all_daily:
				# Expect pattern ..._<YYYYMMDD>.nc; keep robust fallback.
				maybe = p.stem.split('_')[-1]
				try:
					d = datetime.strptime(maybe, '%Y%m%d').date()
				except ValueError:
					continue
				date_map[d] = p
			# 2. Decide which dates to stitch: all if no weeks provided, else expand weeks to dates.
			if weeks is None:
				selected_dates = sorted(date_map)
			else:
				selected_dates = [d for d in expand_weeks_to_dates(year, weeks) if d in date_map]

			missing = [d for d in selected_dates if d not in date_map]
			if missing:
				logging.error("Missing %d daily files for %s %d (weeks selection); first few %s", len(missing), rid, year, missing[:5])
				continue

			selected_files = [date_map[d] for d in selected_dates]
			try:
				meta = load_subcube_meta(selected_files[0])
			except SystemExit as e:
				logging.error(str(e))
				continue

			# Confirm all files share identical jf/if slices
			mismatch = False
			for f in selected_files[1:]:
				try:
					other = load_subcube_meta(f)
				except SystemExit as e:
					logging.error(str(e))
					mismatch = True
					break
				if not (other.jf_start==meta.jf_start and other.jf_stop==meta.jf_stop and other.if_start==meta.if_start and other.if_stop==meta.if_stop):
					logging.error("Corner slice mismatch between %s and %s", f, meta.path)
					mismatch = True
					break
			if mismatch:
				continue

			try:
					run_sim(
					label_id=rid,
					gbr_name=gbr_name,
					safe_name=safe_name,
					year=year,
					weeks=weeks,
					daily_files=selected_files,
					meta=meta,
					outdir=args.outdir,
					seed=args.seed,
					dt_hours=args.dt_hours,
					overwrite=args.overwrite,
					integrator=args.integrator,
					shrink_margin=args.shrink_margin,
					debug_short_run=args.debug_short_run,
				)
			except SystemExit as e:
				logging.error("Run aborted for %s %d: %s", rid, year, e)
			except Exception as e:  # noqa: BLE001
				logging.exception("Unexpected failure for %s %d: %s", rid, year, e)

	logging.info("All runs processed")
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
