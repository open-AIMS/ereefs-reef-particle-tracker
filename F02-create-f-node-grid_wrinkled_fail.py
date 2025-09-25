"""
This version of the script works, but does not produce a usable grid. In an attempt to
fill in the NaN areas of the grid it extrapolates to cover the offshore areas, which
results in a grid that wrinkles and folds over itself in places. This makes it unusable
in OceanParcels.

Build GBR1 f-nodes for OceanParcels, plus a full f-node point shapefile for visual QA.

Purpose:
This script downloads the static curvilinear grid for eReefs GBR1 Hydro v2 from the NCI OPeNDAP catalogue and synthesises the corner coordinates (f-nodes) required by OceanParcels on an Arakawa C-grid. It writes a compact NetCDF containing lon_f[jf,if] and lat_f[jf,if] ready to be referenced when constructing a Parcels FieldSet.from_c_grid_dataset, and also writes a point shapefile with every f-node for validation and map overlays.

Why f-nodes:
Parcels uses barycentric C-grid interpolation and therefore requires the cell-corner coordinates shared by U, V (and W) on C-grids. Parcels does not need the exact U/V node positions when the corner grid is provided. Do not attempt to pass centre coordinates for a C-grid FieldSet.from_c_grid_dataset. 
docs.oceanparcels.org

Data source:
OPeNDAP only, using the curated GBR1 v2 grid aggregation:
https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml.html

Only the 2-D latitude[j,i] and longitude[j,i] arrays are read. No time-varying variables are fetched. This mirrors the AIMS grid-to-shapefile tooling that extracts only latitude, longitude (and sometimes botz) because the grid is static. 
GitHub

Inputs:

latitude[j,i], longitude[j,i] from the OPeNDAP dataset above. The GBR1 centre grid is curvilinear and may include NaN “padding” in the Coral Sea. Those NaNs delineate the outer edge of the computational domain and must be handled as boundaries, because the valid ocean cells form an L-shaped subset of the full rectangular index space.

Outputs:

NetCDF file (default gbr1_fnodes.nc) with:

lon_f[jf,if], lat_f[jf,if] of shape (j+1, i+1), dtype float32, zlib compression level 4.

CF attributes: standard_name, units, and long_name clarifying these are cell corner coordinates for eReefs GBR1 Hydro v2.

Global attributes: source_url (the OPeNDAP URL), crs_epsg (the EPSG used for projection maths), created_utc, repo_url (link to your Git repository), and a short description. Also include a parcels_note briefly explaining how to wire these arrays into FieldSet.from_c_grid_dataset. 
docs.oceanparcels.org

Shapefile (default gbr1_fnodes.shp), point geometry in WGS84 (EPSG:4326), with attributes:

JF and IF as zero-based indices for the corner grid. Parcels itself does not consume the shapefile; it is for visualisation and validation.

Corner synthesis method:

Project the centre lon/lat to a planar CRS to avoid spherical averaging artefacts. Use EPSG:3577 (GDA94 Australian Albers). Record the CRS in the NetCDF global attrs.

Interior corners: for each interior f-node, average the four surrounding centre points in projected coordinates:
Xf[j+1,i+1] = 0.25*(Xc[j,i] + Xc[j+1,i] + Xc[j,i+1] + Xc[j+1,i+1]), and similarly for Yf.

Exterior ring: extrapolate one cell outward by first-order differences along rows and columns:

Xf[1:-1, 0] = Xf[1:-1, 1] - (Xf[1:-1, 2] - Xf[1:-1, 1])

Xf[1:-1, -1] = Xf[1:-1, -2] + (Xf[1:-1, -2] - Xf[1:-1, -3])

Xf[0, :] = Xf[1, :] - (Xf[2, :] - Xf[1, :])

Xf[-1, :] = Xf[-2, :] + (Xf[-2, :] - Xf[-3, :])
Apply the same for Yf.

Handling NaNs in GBR1’s L-shape: treat NaN centres as outside the domain. Compute interior corners only where all four centres are finite. For any corners adjacent to NaN centres, fill by extrapolating from the nearest computed interior corner along the i and j directions using the same first-difference logic as the exterior ring. This treats the NaN region as an outer boundary. Finally, back-project Xf,Yf to lon_f,lat_f in EPSG:4326.

Indexing and orientation:
Use NumPy zero-based indexing. Name the corner dimensions jf and if with sizes j+1 and i+1. This matches Python conventions and aligns with Parcels’ expectation that U and V share the same f-node grid. The shapefile attributes JF and IF are zero-based to match the arrays used in Python. 
docs.oceanparcels.org

Validation:
Run a geometric sanity check on a configurable random sample N of interior cells. For each sampled cell, take the mean of its four f-node corners in projected coordinates and compare with the original centre point. Report the maximum and percentile offsets in metres. Positive, realistic cell areas and monotonic progression of indices are implicitly checked by the synthesis and can be optionally asserted later in workflows concerned with stuck particles. 
docs.oceanparcels.org

Performance and IO:
Use the NetCDF4 library directly to read from OPeNDAP and to write the output NetCDF. Only fetch the latitude and longitude variables and their attributes. Memory use is modest for GBR1. The shapefile is written with GeoPandas/Shapely using EPSG:4326 for portability.

How this plugs into Parcels:
When building your C-grid FieldSet, point the U and V lon and lat dimensions to the lon_f and lat_f variables from the coordinates file produced by this script, so that U and V share the corner grid. Example patterns are shown in the Parcels indexing tutorial with a NEMO coordinates file and are directly transferable by replacing glamf/gphif with lon_f/lat_f. 
docs.oceanparcels.org

Hard-coded configuration:
At the top of this script, define constants for the OPeNDAP URL, output NetCDF path, output shapefile path, CRS EPSG (3577), and the validation sample size.

Limitations:
The edge extrapolation is first order and may not preserve shapes around highly skewed boundary cells, but is consistent with common centre-to-corner synthesis used in ncWMS and in the AIMS GBR1/GBR4 grid shapefile derivations. GBR1 does not cross the antimeridian, so no longitude unwrapping is required. 
GitHub

"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
from datetime import datetime, timezone
from typing import Tuple
from collections import deque

import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer

try:  # Optional heavy deps
	import geopandas as gpd
	from shapely.geometry import Point
except ImportError:  # pragma: no cover
	gpd = None  # type: ignore
	Point = None  # type: ignore



# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
OPENDAP_URL = "https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml"
OUT_NC = "gbr1_fnodes.nc"
OUT_SHP = "gbr1_fnodes.shp"
CRS_EPSG = 3577  # GDA94 Australian Albers (projected)
VALIDATE_SAMPLES = 1000
REPO_URL = "<your-repo-url>"
DESCRIPTION = "GBR1 Hydro v2 f-node corner coordinates for OceanParcels C-grid usage"
MAX_OFFSET_THRESHOLD_M = 200.0  # tolerance for centre vs corner-mean offset (95th percentile gate below)
MAX_FAIL_FRACTION = 0.01  # 1 percent
RANDOM_SEED = 42
SHAPE_PROGRESS_EVERY = 50000  # log every N points when streaming shapefile
FILL_PROGRESS_EVERY = 200  # rows/cols per progress log in fill pass
OUT_CENTRE_SHP = "gbr1_centres.shp"  # new: centre points shapefile


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s %(message)s",
	datefmt="%H:%M:%S",
)
logger = logging.getLogger("gbr1_fnodes")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def read_gbr1_latlon(opendap_url: str = OPENDAP_URL) -> Tuple[np.ndarray, np.ndarray]:
	"""Read only the static longitude[j,i] and latitude[j,i] arrays via OPeNDAP.

	Returns
	-------
	longitude : np.ndarray (j, i)
	latitude  : np.ndarray (j, i)
	NOTE: Return order is (lon, lat) to match common usage downstream.
	"""
	logger.info("Opening OPeNDAP dataset: %s", opendap_url)
	with Dataset(opendap_url) as ds:  # read-only by default
		# Defensive: ensure dims j,i ordering (some grids use y,x or similar names).
		if "latitude" not in ds.variables or "longitude" not in ds.variables:
			raise KeyError("Dataset missing required latitude/longitude variables")
		lat_var = ds.variables["latitude"]
		lon_var = ds.variables["longitude"]
		# Avoid reading any slice with time/k dims. We assume pure 2D.
		if lat_var.ndim != 2 or lon_var.ndim != 2:
			raise ValueError("Expected 2D latitude/longitude arrays")
		# Use masked arrays if present to preserve NaNs for padding/land
		latitude = np.array(lat_var[:], dtype=np.float64)
		longitude = np.array(lon_var[:], dtype=np.float64)
	logger.info("Read lon/lat with shape %s", latitude.shape)
	return longitude, latitude


# ---------------------------------------------------------------------------
# Corner synthesis
# ---------------------------------------------------------------------------
def project_centres(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Transformer, Transformer]:
	"""Project geographic centre coords to planar CRS (EPSG:3577)."""
	logger.info("Projecting centres to EPSG:%s (size=%d points)", CRS_EPSG, lon.size)
	to_proj = Transformer.from_crs(4326, CRS_EPSG, always_xy=True)
	to_geo = Transformer.from_crs(CRS_EPSG, 4326, always_xy=True)
	Xc, Yc = to_proj.transform(lon, lat)  # vectorised
	return np.array(Xc, dtype=np.float64), np.array(Yc, dtype=np.float64), to_proj, to_geo


def compute_interior_corners(Xc: np.ndarray, Yc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute interior corner points. Primary method requires four finite centres.

	If zero corners are formed (pathological), fall back to a nanmean of available
	finite centres among the 2x2 stencil to seed values for subsequent extrapolation.
	"""
	jdim, idim = Xc.shape
	Xf = np.full((jdim + 1, idim + 1), np.nan, dtype=np.float64)
	Yf = np.full((jdim + 1, idim + 1), np.nan, dtype=np.float64)

	finite = np.isfinite(Xc) & np.isfinite(Yc)
	valid = finite[:-1, :-1] & finite[1:, :-1] & finite[:-1, 1:] & finite[1:, 1:]
	primary_count = int(valid.sum())
	if primary_count > 0:
		avgX = 0.25 * (Xc[:-1, :-1] + Xc[1:, :-1] + Xc[:-1, 1:] + Xc[1:, 1:])
		avgY = 0.25 * (Yc[:-1, :-1] + Yc[1:, :-1] + Yc[:-1, 1:] + Yc[1:, 1:])
		Xf_view = Xf[1:-1, 1:-1]
		Yf_view = Yf[1:-1, 1:-1]
		Xf_view[valid] = avgX[valid]
		Yf_view[valid] = avgY[valid]
		logger.info("Interior corners (4-centre) computed: %d", primary_count)
		return Xf, Yf

	# Fallback path
	logger.warning("No interior corners met 4-centre finite criterion; activating fallback nanmean strategy")
	# Build stacked arrays (4, j-1, i-1)
	stacksX = np.stack([Xc[:-1, :-1], Xc[1:, :-1], Xc[:-1, 1:], Xc[1:, 1:]], axis=0)
	stacksY = np.stack([Yc[:-1, :-1], Yc[1:, :-1], Yc[:-1, 1:], Yc[1:, 1:]], axis=0)
	with np.errstate(invalid="ignore"):
		avgX = np.nanmean(stacksX, axis=0)
		avgY = np.nanmean(stacksY, axis=0)
	fallback_valid = np.isfinite(avgX) & np.isfinite(avgY)
	Xf[1:-1, 1:-1][fallback_valid] = avgX[fallback_valid]
	Yf[1:-1, 1:-1][fallback_valid] = avgY[fallback_valid]
	logger.info("Fallback seeded %d interior corners (nanmean of available centres)", int(fallback_valid.sum()))
	return Xf, Yf


def _fill_pass(arr2d: np.ndarray, axis: int):
	"""Directional fill + linear interpolation/extrapolation along 1D slices.

	axis=1: operate across columns for each row.
	axis=0: operate down rows for each column.

	Strategy:
	  * Identify finite indices.
	  * Use numpy.interp for interior NaNs.
	  * Extrapolate leading/trailing NaNs with first-order difference (or constant if only one finite value).
	  * Single pass per slice keeps complexity O(n*m) with vectorised inner ops.
	"""
	if axis not in (0, 1):
		raise ValueError("axis must be 0 or 1")

	n_outer = arr2d.shape[0] if axis == 1 else arr2d.shape[1]
	n_inner = arr2d.shape[1] if axis == 1 else arr2d.shape[0]
	idx_all = np.arange(n_inner)
	log_every = max(1, FILL_PROGRESS_EVERY)
	start_time = datetime.now()
	for outer in range(n_outer):
		slice_view = arr2d[outer, :] if axis == 1 else arr2d[:, outer]
		finite_mask = np.isfinite(slice_view)
		if finite_mask.all():
			continue
		finite_idx = idx_all[finite_mask]
		if finite_idx.size == 0:
			# nothing finite; cannot fill
			continue
		finite_vals = slice_view[finite_mask]
		# Interior interpolation (np.interp broadcasts) - handles interior NaNs only
		nan_mask = ~finite_mask
		interior_targets = idx_all[nan_mask]
		if finite_idx.size == 1:
			# single value -> fill entire slice
			slice_view[:] = finite_vals[0]
		else:
			# Fill interior (will also extend endpoints constant; adjust afterwards)
			slice_view[nan_mask] = np.interp(interior_targets, finite_idx, finite_vals)
			# Leading extrapolation if needed
			first_idx = finite_idx[0]
			if first_idx > 0:
				step = finite_vals[1] - finite_vals[0]
				# vectorised arithmetic progression backwards
				k = np.arange(first_idx - 1, -1, -1)
				slice_view[k] = finite_vals[0] - step * (first_idx - k)
			# Trailing extrapolation
			last_idx = finite_idx[-1]
			if last_idx < n_inner - 1:
				step = finite_vals[-1] - finite_vals[-2]
				k = np.arange(last_idx + 1, n_inner)
				slice_view[k] = finite_vals[-1] + step * (k - last_idx)
		# progress log
		if outer % log_every == 0:
			done_pct = 100.0 * outer / n_outer
			logger.info(
				"Fill pass axis=%d progress %d/%d (%.1f%%) elapsed=%s", axis, outer, n_outer, done_pct, datetime.now() - start_time
			)
	# Final log
	logger.info(
		"Fill pass axis=%d completed (total slices=%d) total_time=%s", axis, n_outer, datetime.now() - start_time
	)


def extrapolate_edges(Xf: np.ndarray, Yf: np.ndarray):
	"""Extrapolate outer ring one cell outward using first differences."""
	# Horizontal edges (excluding corners to avoid duplication until final corner set)
	Xf[1:-1, 0] = Xf[1:-1, 1] - (Xf[1:-1, 2] - Xf[1:-1, 1])
	Xf[1:-1, -1] = Xf[1:-1, -2] + (Xf[1:-1, -2] - Xf[1:-1, -3])
	Yf[1:-1, 0] = Yf[1:-1, 1] - (Yf[1:-1, 2] - Yf[1:-1, 1])
	Yf[1:-1, -1] = Yf[1:-1, -2] + (Yf[1:-1, -2] - Yf[1:-1, -3])

	# Vertical edges
	Xf[0, :] = Xf[1, :] - (Xf[2, :] - Xf[1, :])
	Xf[-1, :] = Xf[-2, :] + (Xf[-2, :] - Xf[-3, :])
	Yf[0, :] = Yf[1, :] - (Yf[2, :] - Yf[1, :])
	Yf[-1, :] = Yf[-2, :] + (Yf[-2, :] - Yf[-3, :])


def fill_and_extrapolate(Xf: np.ndarray, Yf: np.ndarray):
	"""Fill NaNs adjacent to domain edges, then extrapolate outer ring (in-place)."""
	t0 = datetime.now()
	def _nan_count():
		return int(np.isnan(Xf).sum() + np.isnan(Yf).sum())
	logger.info("Starting fill passes (initial NaNs=%d)", _nan_count())
	for round_idx in range(2):
		for arr in (Xf, Yf):
			_fill_pass(arr, axis=1)
			_fill_pass(arr, axis=0)
		logger.info("After fill round %d NaNs=%d", round_idx + 1, _nan_count())
	# Final brute-force nearest-neighbour style fill for any residual isolated NaNs using simple averaging of finite neighbors
	for arr in (Xf, Yf):
		nan_mask = np.isnan(arr)
		if np.any(nan_mask):
			# Replace with mean of finite 3x3 neighborhood if available
			filled = 0
			for (j, i) in zip(*np.where(nan_mask)):
				j0, j1 = max(0, j - 1), min(arr.shape[0], j + 2)
				i0, i1 = max(0, i - 1), min(arr.shape[1], i + 2)
				window = arr[j0:j1, i0:i1]
				finite_vals = window[np.isfinite(window)]
				if finite_vals.size:
					arr[j, i] = float(np.mean(finite_vals))
					filled += 1
			if filled:
				logger.info("Residual fill pass replaced %d isolated NaNs", filled)

	extrapolate_edges(Xf, Yf)
	logger.info("Fill + edge extrapolation completed in %s", datetime.now() - t0)


def propagate_remaining_nans(Xf: np.ndarray, Yf: np.ndarray):
	"""Propagate nearest finite corner values outward using BFS until no NaNs remain.

	This is a conservative nearest-neighbour fill for residual undefined corners
	(typically outside the L-shaped wet domain) so downstream shapefile writing
	does not fail. These synthetic corners are only used for QA visualisation;
	Parcels uses interior cell geometry.
	"""
	nan_mask = ~(np.isfinite(Xf) & np.isfinite(Yf))
	remaining = int(nan_mask.sum())
	if remaining == 0:
		logger.info("No residual NaNs after extrapolation; skipping propagation")
		return
	logger.info("Propagating %d residual NaN corner(s) via BFS nearest fill", remaining)
	jmax, imax = Xf.shape
	finite_mask = np.isfinite(Xf) & np.isfinite(Yf)
	dq = deque(zip(*np.where(finite_mask)))
	visited = finite_mask.copy()
	steps = 0
	while dq and remaining > 0:
		j, i = dq.popleft()
		baseX = Xf[j, i]
		baseY = Yf[j, i]
		for nj, ni in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
			if 0 <= nj < jmax and 0 <= ni < imax and not visited[nj, ni]:
				if not (np.isfinite(Xf[nj, ni]) and np.isfinite(Yf[nj, ni])):
					Xf[nj, ni] = baseX
					Yf[nj, ni] = baseY
					remaining -= 1
				visited[nj, ni] = True
				dq.append((nj, ni))
		steps += 1
		if steps % 200000 == 0:
			logger.info("  propagation queue progress: remaining=%d queue=%d", remaining, len(dq))
	if remaining != 0:
		logger.warning("Propagation ended with %d NaNs still present", remaining)
	else:
		logger.info("Propagation complete; all corners finite")


def build_corners(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""High-level corner synthesis pipeline returning lon_f, lat_f in EPSG:4326."""
	t0 = datetime.now()
	Xc, Yc, _to_proj, to_geo = project_centres(lon, lat)
	logger.info("Centre finite fraction: %.4f", np.isfinite(Xc).mean())
	Xf, Yf = compute_interior_corners(Xc, Yc)
	fill_and_extrapolate(Xf, Yf)
	# Propagate any stubborn NaNs (should be small in count)
	propagate_remaining_nans(Xf, Yf)
	if np.isnan(Xf).any() or np.isnan(Yf).any():
		missing = int(np.isnan(Xf).sum() + np.isnan(Yf).sum())
		raise RuntimeError(f"Unfilled NaNs remain in corner arrays after propagation (count={missing})")
	lon_f, lat_f = to_geo.transform(Xf, Yf)
	logger.info("Corner build complete in %s", datetime.now() - t0)
	return lon_f.astype(np.float32), lat_f.astype(np.float32)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_netcdf(lon_f: np.ndarray, lat_f: np.ndarray, path: str = OUT_NC):
	logger.info("Writing NetCDF: %s", path)
	jf, i_f = lon_f.shape
	with Dataset(path, "w") as ds:
		ds.createDimension("jf", jf)
		ds.createDimension("if", i_f)

		vlon = ds.createVariable(
			"lon_f", "f4", ("jf", "if"), zlib=True, complevel=4, shuffle=True
		)
		vlat = ds.createVariable(
			"lat_f", "f4", ("jf", "if"), zlib=True, complevel=4, shuffle=True
		)
		vlon[:, :] = lon_f
		vlat[:, :] = lat_f

		vlon.standard_name = "longitude"
		vlon.units = "degrees_east"
		vlon.long_name = "GBR1 Hydro v2 cell corner longitudes (f-nodes)"
		vlat.standard_name = "latitude"
		vlat.units = "degrees_north"
		vlat.long_name = "GBR1 Hydro v2 cell corner latitudes (f-nodes)"

		ds.source_url = OPENDAP_URL
		ds.crs_epsg = CRS_EPSG
		ds.created_utc = datetime.now(timezone.utc).isoformat()
		ds.repo_url = REPO_URL
		ds.description = DESCRIPTION
		ds.parcels_note = (
			"Use lon_f/lat_f as the common corner grid for U and V in "
			"FieldSet.from_c_grid_dataset. U and V must share the same f-node dimensions."
		)


def write_shapefile(lon_f: np.ndarray, lat_f: np.ndarray, path: str = OUT_SHP):
	jf, i_f = lon_f.shape
	total_pts = jf * i_f
	if gpd is None or Point is None:
		logger.warning("GeoPandas/Shapely not available; skipping f-node shapefile output")
		return
	logger.info("Writing f-node shapefile %s with %d points (GeoPandas)", path, total_pts)
	JF, IF = np.meshgrid(np.arange(jf, dtype=np.int32), np.arange(i_f, dtype=np.int32), indexing="ij")
	points = [Point(lon_f[j, i], lat_f[j, i]) for j in range(jf) for i in range(i_f)]
	gdf = gpd.GeoDataFrame({"JF": JF.ravel(), "IF": IF.ravel()}, geometry=points, crs="EPSG:4326")
	out_dir = os.path.dirname(os.path.abspath(path)) or "."
	os.makedirs(out_dir, exist_ok=True)
	gdf.to_file(path)
	logger.info("F-node shapefile write complete")


def write_centres_shapefile(lon_c: np.ndarray, lat_c: np.ndarray, path: str = OUT_CENTRE_SHP):
	"""Write the raw centre grid (finite points only) as a point shapefile for comparison.

	Attributes:
	  J, I : zero-based indices of centre nodes.
	NaN centre locations are skipped (logged) to avoid invalid geometries.
	"""
	jdim, idim = lon_c.shape
	total = jdim * idim
	finite_mask = np.isfinite(lon_c) & np.isfinite(lat_c)
	finite_count = int(finite_mask.sum())
	skipped = total - finite_count
	if gpd is None or Point is None:
		logger.warning("GeoPandas/Shapely not available; skipping centre shapefile output")
		return
	logger.info("Writing centre shapefile %s with %d finite points (skipped %d NaNs)", path, finite_count, skipped)
	Js, Is = np.where(finite_mask)
	points = [Point(lon_c[j, i], lat_c[j, i]) for j, i in zip(Js, Is)]
	gdf = gpd.GeoDataFrame({"J": Js.astype(np.int32), "I": Is.astype(np.int32)}, geometry=points, crs="EPSG:4326")
	out_dir = os.path.dirname(os.path.abspath(path)) or "."
	os.makedirs(out_dir, exist_ok=True)
	gdf.to_file(path)
	logger.info("Centre shapefile write complete")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_offsets(
	lon_f: np.ndarray,
	lat_f: np.ndarray,
	lon_c: np.ndarray,
	lat_c: np.ndarray,
	samples: int = VALIDATE_SAMPLES,
	seed: int = RANDOM_SEED,
) -> Tuple[float, float, float, float, float]:
	"""Sample interior (cell) centres and compare to mean of their four f-node corners.

	Acceptance sampling rule: pick j in [0, jdim-2], i in [0, idim-2] such that the 2x2
	block of centre points is fully finite and the four corresponding corners exist.

	Returns tuple: (max_offset, p50, p95, fail_fraction, mean_offset)
	"""
	to_proj = Transformer.from_crs(4326, CRS_EPSG, always_xy=True)
	Xf, Yf = to_proj.transform(lon_f, lat_f)
	Xc, Yc = to_proj.transform(lon_c, lat_c)

	jdim, idim = lon_c.shape
	rng = random.Random(seed)
	centre_valid = np.isfinite(lon_c) & np.isfinite(lat_c)
	# Pre-mask interior cell candidates requiring all four centres finite
	cell_valid = (
		centre_valid[:-1, :-1]
		& centre_valid[1:, :-1]
		& centre_valid[:-1, 1:]
		& centre_valid[1:, 1:]
	)
	valid_indices = np.array(np.where(cell_valid)).T  # shape (N,2) of (j,i)
	if valid_indices.size == 0:
		raise RuntimeError("No finite interior cells for validation")

	requested = samples
	if valid_indices.shape[0] < samples:
		logger.warning(
			"Requested %d validation samples but only %d interior valid cells; reducing",
			samples,
			valid_indices.shape[0],
		)
		samples = valid_indices.shape[0]

	offsets: list[float] = []
	for n in range(samples):
		j, i = valid_indices[rng.randrange(0, valid_indices.shape[0])]
		# Four f-node corners for this cell are at (j+1+i_offset, i+1+j_offset)
		c11 = (j + 1, i + 1)
		c12 = (j + 1, i + 2)
		c21 = (j + 2, i + 1)
		c22 = (j + 2, i + 2)
		if not (
			np.isfinite(Xf[c11])
			and np.isfinite(Xf[c12])
			and np.isfinite(Xf[c21])
			and np.isfinite(Xf[c22])
		):
			# Skip if any corner still invalid (should be rare after propagation); choose another
			continue
		xm = 0.25 * (Xf[c11] + Xf[c12] + Xf[c21] + Xf[c22])
		ym = 0.25 * (Yf[c11] + Yf[c12] + Yf[c21] + Yf[c22])
		dx = xm - Xc[j, i]
		dy = ym - Yc[j, i]
		offsets.append(math.hypot(dx, dy))
		if (n + 1) % 200 == 0:
			logger.info("Validation progress: %d/%d samples collected", n + 1, samples)

	if not offsets:
		raise RuntimeError("Unable to collect any validation offsets (all skipped)")

	arr = np.array(offsets, dtype=np.float64)
	max_o = float(np.max(arr))
	p50 = float(np.percentile(arr, 50))
	p95 = float(np.percentile(arr, 95))
	mean_o = float(np.mean(arr))
	fail_fraction = float(np.mean(arr > MAX_OFFSET_THRESHOLD_M))
	logger.info(
		"Validation samples_used=%d (requested=%d) max=%.2f m p50=%.2f m p95=%.2f m mean=%.2f m fail_fraction=%.4f",
		len(arr),
		requested,
		max_o,
		p50,
		p95,
		mean_o,
		fail_fraction,
	)
	return max_o, p50, p95, fail_fraction, mean_o


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main():  # pragma: no cover - high level orchestration
	t0 = datetime.now()
	lon_c, lat_c = None, None
	try:
		lon_c_orig, lat_c_orig = read_gbr1_latlon()
		lon_c = lon_c_orig
		lat_c = lat_c_orig
		lon_f, lat_f = build_corners(lon_c, lat_c)

		write_netcdf(lon_f, lat_f, OUT_NC)
		write_shapefile(lon_f, lat_f, OUT_SHP)
		write_centres_shapefile(lon_c, lat_c, OUT_CENTRE_SHP)

		# Self-test: reopen NetCDF and confirm shapes
		with Dataset(OUT_NC) as ds:
			lf = ds.variables["lon_f"]
			la = ds.variables["lat_f"]
			assert lf.shape == lon_f.shape == la.shape, "Shape mismatch in output NetCDF"
		validate_offsets(lon_f, lat_f, lon_c, lat_c, VALIDATE_SAMPLES)
		logger.info("Completed in %s", datetime.now() - t0)
	except Exception as exc:  # noqa
		logger.exception("Failure in main: %s", exc)
		raise


if __name__ == "__main__":  # pragma: no cover
	main()



