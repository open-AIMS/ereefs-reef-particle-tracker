"""
03-extract-gbr1-subcubes.py

Purpose
This script downloads grid-aligned NetCDF subcubes from the eReefs GBR1 2.0 OPeNDAP aggregation for named 
reefs on the GBR and saves one file per reef per year at a single depth slice. The goal is to prefetch hourly 
u and v currents on a curvilinear grid in a rectangular i-j window that fully contains a 10 km buffered 
bounding box around each reef. These local files are the inputs for the particle modelling stage.

Inputs
A reef boundary dataset with polygons in EPSG:4326 and at least two attributes, LABEL_ID and GBR_NAME. The 
script filters the layer by the list of LABEL_ID values supplied on the command line. The eReefs OPeNDAP URL 
default is https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml but should remain configurable.

Core behaviour
The script reads the static model grid once to obtain two-dimensional longitude and latitude arrays and the 
vertical coordinate zc. For each target reef polygon it constructs a 10 km outward buffer in a projected CRS 
and converts it back to WGS84. It computes the minimal rectangular index window on the curvilinear grid that 
covers the buffered geographic bounding box. This is achieved by masking lon/lat with the buffered bbox, 
finding all matching j,i pairs, then expanding to j=[jmin..jmax] and i=[imin..imax]. The script opens the 
OPeNDAP dataset with xarray and the netCDF4 engine, selects the requested depth index (k_index, default 40 
which corresponds to zc near -2.35 m), subsets to the i-j window, and slices time to the requested calendar 
year with hourly resolution preserved. By default only u, v, longitude, latitude, zc and time are written to 
disk. The file is compressed using zlib with moderate compression and shuffle to keep sizes reasonable without 
slowing reads.

Outputs
One NetCDF file per reef per year named {GBR_NAME_no_spaces}_{LABEL_ID}_gbr1_k{k_index}_{year}.nc in an 
output directory organised as outdir/LABEL_ID/. Each file includes data variables u and v on dimensions 
time, j, i, plus coordinates longitude[j,i], latitude[j,i] and scalar coordinate zc. Global attributes include
 gbr_name, label_id, k_index, zc_m, j_start, j_stop, i_start, i_stop and buffer_km so the particle stage can 
 reconstruct the modelling envelope. The time coordinate remains hourly CF-compliant as served by eReefs. No 
 resampling is performed.

Assumptions and constraints
The reef polygons may be multipolygons. The script should dissolve multipart geometries into a single polygon 
for each LABEL_ID. The grid is curvilinear so geographic bboxes are only used to find indices; the data 
actually written are a rectangular j-i slab. The script must not attempt to rotate or realign the grid. 
Time is left in the dataset’s native units; decoding with xarray is acceptable and recommended. Network 
access is via OPeNDAP; users may choose to run with dask chunking or serial reads depending on their environment.

Command line interface
Provide arguments for --reef-shp, --ids (space-separated list of LABEL_ID values), --years (start and end or 
an explicit list), --buffer-km (default 10), --k-index (default 40), --outdir, and --opendap. The script 
should validate that each LABEL_ID exists in the shapefile and that at least one hour of data was written. 
Logging should print the chosen i-j window, zc in metres and the output path for traceability.

Performance and robustness
Open the grid once, reuse it across reefs, and only open the full dataset when subsetting a given window. 
Use small time chunks, for example 24 hours, to keep memory steady. Catch and report OPeNDAP and network 
errors, and write partial outputs only after a successful to_netcdf. Optionally add a --dry-run that prints 
windows and output paths without downloading.

Example
python 02-extract-gbr1-subcubes.py --reef-shp reefs.shp --ids 16-068 18-032 18-096 19-019 --years 2016 2023 --buffer-km 10 --k-index 40 --outdir data/subcubes
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import warnings
import os
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
from netCDF4 import Dataset as NCDS  # For reading precomputed f-node grid crop
from shapely.geometry import Polygon, box

# Filter noisy library warnings for cleaner logs (especially on HPC)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_OPENDAP = "https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml"


@dataclass(frozen=True)
class GridArrays:
	"""Container for static grid arrays loaded once per run.

	Attributes
	----------
	lon : np.ndarray
		2-D longitude array (degrees east).
	lat : np.ndarray
		2-D latitude array (degrees north).
	zc : np.ndarray
		1-D vertical coordinate (metres, negative downward).
	"""

	lon: np.ndarray
	lat: np.ndarray
	zc: np.ndarray


def configure_logging(log_file: Path | None = None) -> None:
	"""Configure root logger with stdout and optional file handlers.

	Parameters
	----------
	log_file : Path | None
		If provided, a file handler is added writing INFO level messages.
	"""

	root = logging.getLogger()
	if root.handlers:  # Avoid duplicate handlers in interactive sessions
		return
	root.setLevel(logging.INFO)

	fmt = logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
	)
	sh = logging.StreamHandler(sys.stdout)
	sh.setFormatter(fmt)
	root.addHandler(sh)
	if log_file is not None:
		log_file.parent.mkdir(parents=True, exist_ok=True)
		fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
		fh.setFormatter(fmt)
		root.addHandler(fh)


def load_grid(opendap_url: str = DEFAULT_OPENDAP) -> GridArrays:
	"""Load static curvilinear grid arrays from the remote eReefs dataset.

	The grid (lon, lat, zc) is loaded once and reused for all reef subsets.

	Parameters
	----------
	opendap_url : str
		OPeNDAP aggregation URL for GBR1 2.0.

	Returns
	-------
	GridArrays
		Dataclass with longitude, latitude and vertical coordinate arrays.
	"""

	logging.info("Loading static grid arrays from OPeNDAP: %s", opendap_url)
	try:
		ds = xr.open_dataset(opendap_url, decode_times=True)
	except OSError as e:  # Network / opendap resolution errors
		raise RuntimeError(f"Failed to open OPeNDAP dataset: {e}") from e

	required = {"longitude", "latitude", "zc"}
	missing = required - set(ds.variables)
	if missing:
		raise ValueError(f"Dataset missing required variables: {missing}")

	lon = ds["longitude"].astype("float64").values
	lat = ds["latitude"].astype("float64").values
	zc = ds["zc"].astype("float64").values

	if lon.ndim != 2 or lat.ndim != 2:
		raise ValueError("Expected 2-D longitude and latitude arrays")
	if zc.ndim != 1:
		raise ValueError("Expected 1-D vertical coordinate zc")

	logging.info("Grid loaded: lon/lat shape %s, vertical levels %d", lon.shape, zc.size)
	ds.close()
	return GridArrays(lon=lon, lat=lat, zc=zc)


def _buffer_polygon_epsg3112(poly: Polygon, buffer_km: float) -> Polygon:
	"""Return outward buffer of polygon using EPSG:3112 then back to WGS84.

	Parameters
	----------
	poly : Polygon
		Input polygon in EPSG:4326.
	buffer_km : float
		Buffer distance in kilometres.
	"""

	if buffer_km <= 0:
		return poly
	gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326").to_crs("EPSG:3112")
	buffered = gdf.buffer(buffer_km * 1000.0)
	# Convert back
	buffered_wgs84 = buffered.to_crs("EPSG:4326").iloc[0]
	return Polygon(buffered_wgs84.exterior)


def ij_window_for_buffered_bbox(
	grid: GridArrays, poly: Polygon, buffer_km: float
) -> Tuple[int, int, int, int]:
	"""Compute minimal rectangular i-j window covering buffered polygon bbox.

	Steps
	-----
	1. Buffer polygon in projected CRS (EPSG:3112) by buffer_km.
	2. Take geographic bounding box of buffered polygon.
	3. Identify all (j,i) where lon/lat falls within that bbox.
	4. Return (j_start, j_stop, i_start, i_stop) inclusive indices spanning range.

	Parameters
	----------
	grid : GridArrays
		Static grid arrays.
	poly : Polygon
		Reef polygon in EPSG:4326.
	buffer_km : float
		Buffer distance in kilometres.

	Returns
	-------
	tuple[int, int, int, int]
		Inclusive index bounds (j_start, j_stop, i_start, i_stop).
	"""

	buffered = _buffer_polygon_epsg3112(poly, buffer_km)
	minx, miny, maxx, maxy = buffered.bounds
	lon = grid.lon
	lat = grid.lat

	mask = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
	# Defensive improvement: exclude NaN padding so windows never straddle invalid centres
	finite_mask = np.isfinite(lat) & np.isfinite(lon)
	mask &= finite_mask
	if not mask.any():
		raise ValueError("Buffered bbox does not intersect grid; check inputs")

	js, is_ = np.where(mask)
	j_start, j_stop = int(js.min()), int(js.max())
	i_start, i_stop = int(is_.min()), int(is_.max())
	logging.info(
		"Computed window j:[%d,%d] i:[%d,%d] (buffer %.1f km)",
		j_start,
		j_stop,
		i_start,
		i_stop,
		buffer_km,
	)
	return j_start, j_stop, i_start, i_stop


def read_fnode_crop(path: str) -> tuple[int, int, int, int]:
	"""Read crop indices (centre-grid) from precomputed f-node NetCDF.

	Expected global attribute: crop_j0_i0_j1_i1 with CSV 'j0,i0,j1,i1' where j1/i1 are exclusive.
	"""
	with NCDS(path, "r") as ds:  # noqa: SIM115
		attr = getattr(ds, "crop_j0_i0_j1_i1", None)
		if not attr:
			raise RuntimeError(f"f-node file {path} missing crop_j0_i0_j1_i1 attribute")
		try:
			j0, i0, j1, i1 = map(int, str(attr).split(","))
		except Exception as e:  # noqa: BLE001
			raise RuntimeError(f"Unable to parse f-node crop '{attr}' in {path}") from e
		# Sanity: ensure lon_f / lat_f present
		_ = ds.variables["lon_f"]
		_ = ds.variables["lat_f"]
	return j0, i0, j1, i1


def _sanitize_name(name: str) -> str:
	return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name.replace(" ", "_"))


def _expected_hours_in_year(year: int) -> int:
	# Leap year simple test
	leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
	return 366 * 24 if leap else 365 * 24


def extract_reef_year(
	opendap_url: str,
	grid: GridArrays,
	label_id: str,
	gbr_name: str,
	geom: Polygon,
	buffer_km: float,
	k_index: int,
	year: int,
	outdir: Path,
	fnode_nc: str,
	weeks: List[int] | None = None,
	time_chunk_hours: int = 24,
	daily_mode: bool = True,
) -> Path:
	"""Extract a reef-year as either a single file or daily incremental files.

	When daily_mode=True:
		Writes one NetCDF per day named {safe}_{label}_k{k}_YYYYMMDD.nc inside reef directory.
		Skips days already present. Provides progress logging.
		Also writes a manifest text file listing completed days.

	When daily_mode=False:
		Falls back to single-year behaviour (legacy path).
	"""

	j_start, j_stop, i_start, i_stop = ij_window_for_buffered_bbox(grid, geom, buffer_km)
	# Convert inclusive stops to exclusive for containment logic
	j_stop_ex = j_stop + 1
	i_stop_ex = i_stop + 1
	# Read f-node crop and validate containment
	j0, i0, j1, i1 = read_fnode_crop(fnode_nc)
	if not (j_start >= j0 and j_stop_ex <= j1 and i_start >= i0 and i_stop_ex <= i1):
		raise SystemExit(
			f"Requested subcube window j[{j_start}:{j_stop_ex}) i[{i_start}:{i_stop_ex}) "
			f"is outside f-node crop j[{j0}:{j1}) i[{i0}:{i1}). "
			"Choose a reef closer to the coast or rebuild the f-node grid."
		)
	# Derive f-node slice indices per +1 rule (relative to crop origin) documented in spec
	jf_start = (j_start - j0) + 1
	jf_stop = (j_stop_ex - j0) + 1
	if_start = (i_start - i0) + 1
	if_stop = (i_stop_ex - i0) + 1
	logging.info(
		"Window centres j[%d:%d) i[%d:%d); f-node slices jf[%d:%d) if[%d:%d) from %s",
		j_start,
		j_stop_ex,
		i_start,
		i_stop_ex,
		jf_start,
		jf_stop,
		if_start,
		if_stop,
		fnode_nc,
	)
	depth_m = float(grid.zc[k_index])
	safe_name = _sanitize_name(gbr_name)
	label_dir = outdir / label_id / str(year)
	label_dir.mkdir(parents=True, exist_ok=True)
	manifest_path = label_dir / "manifest.txt"

	weeks_attr = None
	if weeks:
		weeks_sorted = sorted(set(weeks))
		weeks_attr = ",".join(str(w) for w in weeks_sorted)
		logging.info("Restricting extraction to weeks (Jan1-Jan7=1): %s", weeks_sorted)

	if not daily_mode:
		# Fallback to yearly single file (reuse earlier logic by toggling daily_mode flag off)
		yearly_dir = outdir / label_id
		yearly_dir.mkdir(parents=True, exist_ok=True)
		final_path = yearly_dir / f"{safe_name}_{label_id}_gbr1_k{k_index}_{year}.nc"
		if final_path.exists():
			logging.info("Year file exists, skipping: %s", final_path)
			return final_path
		logging.info(
			"(Non-daily mode) Extracting %s (%s) year %d depth %.2f m window j[%d:%d] i[%d:%d]",\
			label_id, gbr_name, year, depth_m, j_start, j_stop, i_start, i_stop
		)
		# Re-open dataset per earlier method
		try:
			ds = xr.open_dataset(opendap_url, decode_times=True)
		except OSError as e:  # noqa: BLE001
			raise RuntimeError(f"Failed to open dataset: {e}") from e
		required_vars = ["u", "v", "longitude", "latitude", "zc", "time"]
		if any(v not in ds.variables for v in required_vars):
			raise ValueError("Dataset missing required variables for yearly extraction")
		ds_sub = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23:00:00"))
		# If weeks restriction apply a mask to time dimension (ordinal week numbers Jan1-Jan7=1)
		if weeks:
			_time_vals = ds_sub['time'].values
			_allowed = set(weeks)
			# Convert to ordinal weeks: week = (day_of_year-1)//7 + 1
			# Convert to UTC date first
			secs = (_time_vals - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1,'s')
			py_dates = [_dt.datetime.utcfromtimestamp(float(s)) for s in secs]
			day_of_year = np.array([d.timetuple().tm_yday for d in py_dates])
			ord_weeks = ((day_of_year - 1) // 7) + 1
			mask_time = np.isin(ord_weeks, list(_allowed))
			if mask_time.sum() == 0:
				logging.warning("No times found for requested ordinal weeks %s in year %d", weeks, year)
			ds_sub = ds_sub.isel(time=mask_time)
		spatial = ds_sub.isel(k=k_index, j=slice(j_start, j_stop + 1), i=slice(i_start, i_stop + 1))
		out = xr.Dataset(
			{
				"u": spatial["u"].transpose("time", "j", "i"),
				"v": spatial["v"].transpose("time", "j", "i"),
				"longitude": ds["longitude"].isel(j=slice(j_start, j_stop + 1), i=slice(i_start, i_stop + 1)),
				"latitude": ds["latitude"].isel(j=slice(j_start, j_stop + 1), i=slice(i_start, i_stop + 1)),
				"zc": xr.DataArray(depth_m, dims=()),
				"time": spatial["time"],
			}
		)
		out.attrs.update(
			dict(
				gbr_name=gbr_name,
				label_id=label_id,
				k_index=k_index,
				zc_m=depth_m,
				j_start=j_start,
				j_stop=j_stop,
				i_start=i_start,
				i_stop=i_stop,
				buffer_km=buffer_km,
				source_opendap=opendap_url,
				history="Extracted (year mode) with 02-extract-gbr1-subcubes.py",
				fnode_file=os.path.abspath(fnode_nc),
				fnode_crop=f"{j0},{i0},{j1},{i1}",
				jf_start=int(jf_start),
				jf_stop=int(jf_stop),
				if_start=int(if_start),
				if_stop=int(if_stop),
				weeks=weeks_attr if weeks_attr else "all",
			)
		)
		# Extend / create description attribute with provenance note
		desc = out.attrs.get("description", "")
		out.attrs["description"] = (desc + " Subcube verified to lie inside f-node crop; jf_*/if_* give matching corner slices.").strip()
		comp = dict(zlib=True, complevel=4, shuffle=True)
		encoding = {"u": comp, "v": comp, "longitude": comp, "latitude": comp, "time": {}, "zc": {}}
		out = out.chunk({"time": time_chunk_hours})
		tmp_path = final_path.with_suffix(".tmp.nc")
		out.to_netcdf(tmp_path, format="NETCDF4", encoding=encoding)
		tmp_path.replace(final_path)
		logging.info("Year file written: %s", final_path)
		ds.close()
		return final_path

	# DAILY MODE
	logging.info(
		"Daily mode extraction %s (%s) year %d depth %.2f m window j[%d:%d] i[%d:%d] (%d x %d cells)",
		label_id,
		gbr_name,
		year,
		depth_m,
		j_start,
		j_stop,
		i_start,
		i_stop,
		(j_stop - j_start + 1),
		(i_stop - i_start + 1),
	)
	try:
		ds_full = xr.open_dataset(opendap_url, decode_times=True)
	except OSError as e:  # noqa: BLE001
		raise RuntimeError(f"Failed to open dataset: {e}") from e
	required_vars = ["u", "v", "longitude", "latitude", "zc", "time"]
	if any(v not in ds_full.variables for v in required_vars):
		raise ValueError("Dataset missing required variables")

	# Pre-select spatial slice for efficiency (still remote but reduces transfer for each day)
	spatial_base = ds_full.isel(
		k=k_index, j=slice(j_start, j_stop + 1), i=slice(i_start, i_stop + 1)
	)

	# Build list of all days of year (will filter by weeks if provided)
	start = np.datetime64(f"{year}-01-01")
	end = np.datetime64(f"{year}-12-31")
	all_days = np.arange(start, end + np.timedelta64(1, "D"), dtype="datetime64[D]")
	if weeks:
		_allowed = set(weeks)
		_filtered = []
		for d in all_days:
			# Convert to python date
			py_date = (d.astype('datetime64[D]')).astype(object)
			ordinal_week = ((py_date.timetuple().tm_yday - 1) // 7) + 1
			if ordinal_week in _allowed:
				_filtered.append(d)
		all_days = np.array(_filtered, dtype='datetime64[D]')
		logging.info("Filtered days by ordinal weeks %s -> %d days", sorted(_allowed), all_days.size)

	completed = set()
	if manifest_path.exists():
		completed = set(d.strip() for d in manifest_path.read_text().splitlines() if d.strip())
	logging.info("Days already completed: %d", len(completed))

	comp = dict(zlib=True, complevel=4, shuffle=True)
	encoding = {"u": comp, "v": comp, "longitude": comp, "latitude": comp, "time": {}, "zc": {}}

	for idx, day in enumerate(all_days, start=1):
		day_str = str(day)  # YYYY-MM-DD
		if day_str in completed:
			continue
		day_path = label_dir / f"{safe_name}_{label_id}_k{k_index}_{day_str.replace('-', '')}.nc"
		if day_path.exists():
			logging.info("Skipping existing day file %s", day_path.name)
			completed.add(day_str)
			continue
		day_start = f"{day_str}T00:00:00"
		day_end = f"{day_str}T23:00:00"
		logging.info(
			"[%s %d/%d] Downloading day %s", label_id, idx, len(all_days), day_str
		)
		day_slice = spatial_base.sel(time=slice(day_start, day_end))
		if day_slice.sizes.get("time", 0) == 0:
			logging.warning("No data for day %s; creating empty placeholder", day_str)
			continue
		# Build minimal dataset
		out = xr.Dataset(
			{
				"u": day_slice["u"].transpose("time", "j", "i"),
				"v": day_slice["v"].transpose("time", "j", "i"),
				"longitude": ds_full["longitude"].isel(j=slice(j_start, j_stop + 1), i=slice(i_start, i_stop + 1)),
				"latitude": ds_full["latitude"].isel(j=slice(j_start, j_stop + 1), i=slice(i_start, i_stop + 1)),
				"zc": xr.DataArray(depth_m, dims=()),
				"time": day_slice["time"],
			}
		)
		out.attrs.update(
			dict(
				gbr_name=gbr_name,
				label_id=label_id,
				k_index=k_index,
				zc_m=depth_m,
				j_start=j_start,
				j_stop=j_stop,
				i_start=i_start,
				i_stop=i_stop,
				buffer_km=buffer_km,
				source_opendap=opendap_url,
				history="Daily extraction with 02-extract-gbr1-subcubes.py",
				fnode_file=os.path.abspath(fnode_nc),
				fnode_crop=f"{j0},{i0},{j1},{i1}",
				jf_start=int(jf_start),
				jf_stop=int(jf_stop),
				if_start=int(if_start),
				if_stop=int(if_stop),
				weeks=weeks_attr if weeks_attr else "all",
			)
		)
		desc = out.attrs.get("description", "")
		out.attrs["description"] = (desc + " Subcube verified to lie inside f-node crop; jf_*/if_* give matching corner slices.").strip()
		tmp_day = day_path.with_suffix(".tmp.nc")
		if tmp_day.exists():
			tmp_day.unlink(missing_ok=True)
		try:
			out.to_netcdf(tmp_day, format="NETCDF4", encoding=encoding)
			tmp_day.replace(day_path)
			completed.add(day_str)
			manifest_path.write_text("\n".join(sorted(completed)))
		except Exception:  # noqa: BLE001
			with contextlib.suppress(Exception):
				tmp_day.unlink(missing_ok=True)
			raise
	ds_full.close()
	logging.info(
		"Completed daily extraction for %s %d (%d/%d days).", label_id, year, len(completed), len(all_days)
	)
	# Return manifest path for reference
	return manifest_path


def extract_all_years(
	opendap_url: str,
	grid: GridArrays,
	reefs: gpd.GeoDataFrame,
	ids_ordered: List[str],
	years: List[int],
	buffer_km: float,
	k_index: int,
	outdir: Path,
	fnode_nc: str,
	weeks: List[int] | None = None,
) -> None:
	"""Loop over reefs and years extracting subcubes.

	Aborts entire run on first critical failure to surface issues early (per spec preference to abort).
	Structured for future parallelisation: outer loops explicit and side-effect limited to file writes.
	"""

	for rid in ids_ordered:
		row = reefs[reefs["LABEL_ID"] == rid].iloc[0]
		gbr_name = row["GBR_NAME"]
		geom: Polygon = row.geometry  # type: ignore[assignment]
		for year in years:
			try:
				extract_reef_year(
					opendap_url=opendap_url,
					grid=grid,
					label_id=rid,
					gbr_name=gbr_name,
					geom=geom,
					buffer_km=buffer_km,
					k_index=k_index,
					year=year,
					outdir=outdir,
					fnode_nc=fnode_nc,
					weeks=weeks,
					daily_mode=True,
				)
			except Exception as e:  # noqa: BLE001
				logging.error("Extraction failed for %s %d: %s", rid, year, e)
				raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command line arguments for extraction script.

	Returns
	-------
	argparse.Namespace
		Parsed arguments.
	"""

	p = argparse.ArgumentParser(
		description="Extract grid-aligned GBR1 subcubes for specified reefs and years.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	p.add_argument(
		"--reef-shp",
		type=Path,
		default=Path(
			"data/in-3p/GBR_AIMS_Complete-GBR-feat_V1b/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp"
		),
		help="Path to reef shapefile (.shp) or GeoPackage (.gpkg).",
	)
	p.add_argument("--ids", nargs="+", required=True, help="Space separated list of LABEL_ID values.")
	p.add_argument(
		"--years",
		nargs="+",
		type=int,
		required=True,
		help="Year or list of years; if two values given they define inclusive range.",
	)
	p.add_argument(
		"--buffer-km", type=float, default=10.0, help="Buffer distance applied around each reef polygon (km)."
	)
	p.add_argument(
		"--k-index", type=int, default=40, help="Vertical index (zero-based) to extract (default near -2.35 m)."
	)
	p.add_argument(
		"--depth-m",
		type=float,
		default=None,
		help="Optional target depth in metres (negative downward). If provided overrides k-index by nearest match.",
	)
	p.add_argument(
		"--opendap",
		type=str,
		default=DEFAULT_OPENDAP,
		help="eReefs GBR1 2.0 OPeNDAP aggregation URL.",
	)
	p.add_argument(
		"--outdir",
		type=Path,
		default=Path("working/02"),
		help="Output root directory (subfolders by LABEL_ID).",
	)
	p.add_argument(
		"--weeks",
		nargs="*",
		type=int,
		default=None,
		help="Ordinal weeks (Jan1-Jan7=1, Jan8-Jan14=2, ...) to process; two values = inclusive range; omit for all weeks.",
	)
	p.add_argument(
		"--fnode-nc",
		type=str,
		default="working/02/gbr1_fnodes.nc",
		help="Path to precomputed f-node NetCDF containing lon_f/lat_f and crop_j0_i0_j1_i1",
	)
	p.add_argument(
		"--seed", type=int, default=42, help="Random seed (reserved for reproducibility, not heavily used here)."
	)
	p.add_argument(
		"--log-file",
		type=Path,
		default=None,
		help="Optional path to log file; if omitted only stdout is used.",
	)
	return p.parse_args(argv)


def _years_list(year_args: List[int]) -> List[int]:
	if len(year_args) == 1:
		return year_args
	if len(year_args) == 2:
		a, b = year_args
		if a > b:
			a, b = b, a
		return list(range(a, b + 1))
	return sorted(set(year_args))


def _weeks_list(week_args: List[int] | None) -> List[int] | None:
	"""Normalize weeks argument (ordinal weeks: Jan1-Jan7=1, Jan8-Jan14=2, ...).

	Rules:
	- None -> None (all weeks)
	- 1 value -> list with that value
	- 2 values -> inclusive range
	- >2 -> unique sorted list
	Validation: weeks must be >=1; no fixed upper bound because last partial week can be 52 or 53 depending on year length.
	"""
	if not week_args:
		return None
	if len(week_args) == 1:
		weeks = week_args
	elif len(week_args) == 2:
		a, b = week_args
		if a > b:
			a, b = b, a
		weeks = list(range(a, b + 1))
	else:
		weeks = sorted(set(week_args))
	for w in weeks:
		if w < 1:
			raise ValueError(f"Week number {w} invalid; must be >= 1")
	return weeks


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	configure_logging(args.log_file)
	logging.info("Starting extraction run")

	# Validate reef shapefile exists
	if not args.reef_shp.exists():
		logging.error("Reef shapefile not found: %s", args.reef_shp)
		return 2

	years = _years_list(args.years)
	try:
		weeks = _weeks_list(args.weeks)
	except ValueError as e:
		logging.error(str(e))
		return 7
	if weeks:
		logging.info("Requested weeks filter: %s", weeks)
		# Determine maximum ordinal week for each year (based on day count)
		for y in years:
			# Days in year
			leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
			days_in_year = 366 if leap else 365
			max_week = ((days_in_year - 1) // 7) + 1
			too_high = [w for w in weeks if w > max_week]
			logging.info("Year %d has %d ordinal weeks (last may be partial)", y, max_week)
			if too_high:
				logging.warning("Weeks %s exceed max %d for year %d and will match zero days", too_high, max_week, y)
	else:
		logging.info("No weeks filter: processing all weeks of each year")
	logging.info("Years to process: %s", years)
	try:
		grid = load_grid(args.opendap)
	except Exception as e:  # noqa: BLE001
		logging.exception("Failed to load grid: %s", e)
		return 3

	# If depth specified override k-index
	k_index = args.k_index
	if args.depth_m is not None:
		diffs = np.abs(grid.zc - args.depth_m)
		k_index = int(np.argmin(diffs))
		logging.info(
			"Depth override requested: target %.2f m -> nearest zc %.2f m at k_index=%d",
			args.depth_m,
			grid.zc[k_index],
			k_index,
		)
	else:
		logging.info("Using provided k_index=%d (depth %.2f m)", k_index, grid.zc[k_index])

	# Load reef polygons (only shapefile/gpkg). We assume LABEL_ID and GBR_NAME exist.
	reefs = gpd.read_file(args.reef_shp)
	required_cols = {"LABEL_ID", "GBR_NAME"}
	missing_cols = required_cols - set(reefs.columns)
	if missing_cols:
		logging.error("Reef file missing required columns: %s", missing_cols)
		return 4

	# Filter by requested IDs preserving order of user list
	id_set = set(args.ids)
	subset = reefs[reefs["LABEL_ID"].isin(id_set)].copy()
	if subset.empty or subset["LABEL_ID"].nunique() != len(id_set):
		missing_ids = [rid for rid in args.ids if rid not in subset["LABEL_ID"].unique()]
		logging.error("One or more LABEL_ID not found: %s", missing_ids)
		return 5
	# Reorder by user provided list
	subset["_order"] = subset["LABEL_ID"].apply(lambda v: args.ids.index(v))
	subset = subset.sort_values("_order")

	logging.info("Processing %d reefs", subset.shape[0])
	try:
		extract_all_years(
			opendap_url=args.opendap,
			grid=grid,
			reefs=subset,
			ids_ordered=args.ids,
			years=years,
			buffer_km=args.buffer_km,
			k_index=k_index,
			outdir=args.outdir,
			fnode_nc=args.fnode_nc,
			weeks=weeks,
		)
	except Exception:
		return 6
	logging.info("Extraction run complete.")
	return 0


if __name__ == "__main__":  # pragma: no cover
	raise SystemExit(main())
