"""
Build an OceanParcels-ready corner grid (f-nodes) for eReefs GBR1 by cropping to the
vertical of the L and synthesising cell corners.

Why this script exists
OceanParcels performs C-grid interpolation using the geometry of cell corners rather than only 
the cell centres. On curvilinear meshes like eReefs GBR1 the shape and orientation of each cell 
vary across the domain. Parcels therefore needs a complete, non-degenerate set of corner 
coordinates that U and V can share. The native eReefs files publish longitude and latitude at 
cell centres, with NaN padding in unused parts of the Coral Sea. Those NaNs form an L-shaped 
valid region inside the rectangular index frame. If you pass Parcels a corner grid that contains 
NaNs or folded quadrilaterals, the particle cell-search and interpolation can fail. This utility 
builds a clean f-node grid that Parcels can consume directly.

What f-nodes mean here
An f-node is the geographic location of a cell corner. For a centre grid of shape (j, i) there 
are (j+1, i+1) corners. Parcels expects U and V on a C-grid to reference the same f-node longitude 
and latitude so that it can weight face-centred velocities correctly on a curving mesh. The arrays 
produced by this script are called lon_f and lat_f and have those corner dimensions.

Overview of the processing
The script reads only the static two-dimensional longitude and latitude from the NCI OPeNDAP 
aggregation for GBR1. It then finds a simple rectangular crop that preserves all rows and 
keeps the widest possible west-origin band of finite columns. Concretely, for each row it 
counts the number of leading finite columns from i=0 and takes the minimum of those counts 
across all rows. The crop becomes i in [0, I_end) and j in [0, jmax). This is the vertical of 
the L that hugs the coastline and contains most reefs. Within this crop the centre grid is 
NaN-free by construction.

Corners are synthesised in a projected coordinate system to avoid artefacts from averaging 
angular degrees. Centre longitudes and latitudes are projected to EPSG:3577. For each interior 
corner the script averages the four surrounding centres to place the corner midway between them. 
A single outer ring of corners is then extrapolated by first-order differences along rows and 
columns so that the corner grid is one larger in each direction than the centre grid. The 
projected corners are transformed back to WGS84 longitudes and latitudes and written to a 
small NetCDF file with CF-style metadata. An optional point shapefile of all corners is written 
for quick visual checks. A lightweight validation reports the distance, in metres in the 
projected space, between each sampled centre and the mean of its four corners.

Why we crop rather than inpaint
The public GBR1 centre grid contains NaN padding that creates an L-shaped footprint. Filling 
that wedge with naive extrapolation tends to produce wrinkled or overlapping cells near the 
boundary, which breaks the topology Parcels assumes. Because you will not simulate near the 
outer model edge, the simplest robust solution is to discard the problematic wedge entirely 
and keep a single interior band that is fully finite and still aligned with the coast. This
 avoids inpainting, avoids folded cells, and yields a clean corner mesh for Parcels. When 
 you later subset windows around reefs and slice U and V, use the same j and i crop recorded 
 in the output attributes so that velocities and corners are consistent.

What gets written and how to use it
The NetCDF output contains lon_f and lat_f with dimensions jf=j+1 and if=i+1 for the cropped 
band. Global attributes record the OPeNDAP source, the projection used for the maths, the 
creation time, a short description, and the j and i crop indices. In OceanParcels you construct 
a C-grid FieldSet by pointing the U and V fields to your velocity files sliced to the same j 
and i window, and by supplying lon_f and lat_f from this file as the shared corner grid. The 
optional shapefile of points is only for visual validation and does not enter the Parcels workflow.

Limitations and scope
This corner grid intentionally excludes the offshore wedge and any ragged edges of the original 
L-shape. It is not suitable for simulations that rely on flows or particles near the model 
boundary. If you later need coverage that extends into the padded side you will need a 
different strategy, such as a larger crop or a proper inpainting method that preserves cell 
orientation. For the present goal of reef-centred simulations well inside the shelf, this 
cropped corner grid keeps the implementation simple and prevents geometry-related failures 
in OceanParcels.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
from netCDF4 import Dataset
from pyproj import Transformer

try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:
    gpd = None
    Point = None


# -----------------------------------------------------------------------------
# Configuration (hard coded for simplicity)
# -----------------------------------------------------------------------------
OPENDAP_URL = "https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml"
OUT_NC = "working/02/gbr1_fnodes.nc"
OUT_SHP = "working/02/gbr1_fnodes.shp"      # optional; skipped if GeoPandas missing
CRS_EPSG = 3577                  # GDA94 Australian Albers (planar maths)
DESCRIPTION = "GBR1 Hydro v2 f-node corner coordinates cropped to the vertical of the L (widest west-origin band finite for all rows) for OceanParcels C-grid usage"
REPO_URL = "https://github.com/open-AIMS/ereefs-reef-water-residency"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("gbr1_fnodes_vertical")


# -----------------------------------------------------------------------------
# Data access
# -----------------------------------------------------------------------------
def read_latlon(opendap_url: str = OPENDAP_URL) -> Tuple[np.ndarray, np.ndarray]:
    """Read latitude[j,i], longitude[j,i] from OPeNDAP as float64 numpy arrays."""
    log.info("Opening OPeNDAP dataset: %s", opendap_url)
    with Dataset(opendap_url) as ds:
        if "latitude" not in ds.variables or "longitude" not in ds.variables:
            raise KeyError("latitude/longitude variables not found")
        lat = np.array(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.array(ds.variables["longitude"][:], dtype=np.float64)
    if lat.ndim != 2 or lon.ndim != 2:
        raise ValueError("Expected 2D latitude/longitude arrays")
    log.info("Centre grid shape j,i = %s", lat.shape)
    return lon, lat


# -----------------------------------------------------------------------------
# Crop to the vertical of the L
# -----------------------------------------------------------------------------
def leading_finite_run_length(row_mask: np.ndarray) -> int:
    """
    Return the number of leading True values from index 0 in a 1D boolean array.
    If row is all True, returns len(row). If row[0] is False, returns 0.
    """
    # np.argmax returns 0 when the first element is the max; invert logic by appending a False sentinel
    # and finding the first False.
    inv = ~row_mask
    # Find first False; if none, result is len(row_mask)
    idx = np.argmax(np.concatenate([inv, [True]]))  # first True in 'inv' i.e., first False in row_mask
    return idx


def crop_vertical_band(lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """
    Keep all rows, and crop columns to the widest west-origin band that is finite for every row.

    For each row j, compute R_j = number of leading finite values from i=0.
    Let I_end = min_j R_j. Crop to i in [0, I_end). This preserves the vertical of the L.

    Returns cropped lon, lat and crop indices (j0, j1, i0, i1) in source space.
    """
    jdim, idim = lon.shape
    finite = np.isfinite(lon) & np.isfinite(lat)

    # Guard: require at least some finite data in column 0 for most rows
    if not finite[:, 0].any():
        raise RuntimeError("Column 0 has no finite values; cannot form west-origin band")

    run_lengths = np.fromiter(
        (leading_finite_run_length(finite[j, :]) for j in range(jdim)),
        dtype=np.int64, count=jdim
    )
    I_end = int(run_lengths.min())

    if I_end < 2:
        raise RuntimeError(
            f"Computed west-origin finite band too narrow (I_end={I_end}). "
            f"Check grid orientation or NaN padding assumptions."
        )

    i0, i1 = 0, I_end
    j0, j1 = 0, jdim  # retain all rows
    lon_c = lon[j0:j1, i0:i1]
    lat_c = lat[j0:j1, i0:i1]

    # Sanity: crop must be fully finite by construction
    if not (np.isfinite(lon_c).all() and np.isfinite(lat_c).all()):
        raise RuntimeError("Vertical-band crop unexpectedly contains NaNs")

    log.info("Cropped vertical band: j=[%d:%d) (all rows), i=[%d:%d) -> shape %s", j0, j1, i0, i1, lon_c.shape)
    return lon_c, lat_c, (j0, j1, i0, i1)


# -----------------------------------------------------------------------------
# Corner synthesis on the cropped band
# -----------------------------------------------------------------------------
def build_corners(lon_c: np.ndarray, lat_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build f-node corners for a fully finite centre grid by:
      1) projecting to EPSG:3577,
      2) averaging 2x2 centres for interior corners,
      3) extrapolating a single outer ring by first differences,
      4) projecting back to lon/lat (EPSG:4326).
    """
    to_proj = Transformer.from_crs(4326, CRS_EPSG, always_xy=True)
    to_geo = Transformer.from_crs(CRS_EPSG, 4326, always_xy=True)

    Xc, Yc = to_proj.transform(lon_c, lat_c)  # vectorised
    jdim, idim = Xc.shape

    Xf = np.empty((jdim + 1, idim + 1), dtype=np.float64)
    Yf = np.empty((jdim + 1, idim + 1), dtype=np.float64)

    # Interior corners
    Xf[1:-1, 1:-1] = 0.25 * (Xc[:-1, :-1] + Xc[1:, :-1] + Xc[:-1, 1:] + Xc[1:, 1:])
    Yf[1:-1, 1:-1] = 0.25 * (Yc[:-1, :-1] + Yc[1:, :-1] + Yc[:-1, 1:] + Yc[1:, 1:])

    # Single outer ring by first differences
    Xf[1:-1, 0]   = Xf[1:-1, 1] - (Xf[1:-1, 2] - Xf[1:-1, 1])
    Xf[1:-1, -1]  = Xf[1:-1, -2] + (Xf[1:-1, -2] - Xf[1:-1, -3])
    Xf[0, :]      = Xf[1, :]     - (Xf[2, :]     - Xf[1, :])
    Xf[-1, :]     = Xf[-2, :]    + (Xf[-2, :]    - Xf[-3, :])

    Yf[1:-1, 0]   = Yf[1:-1, 1] - (Yf[1:-1, 2] - Yf[1:-1, 1])
    Yf[1:-1, -1]  = Yf[1:-1, -2] + (Yf[1:-1, -2] - Yf[1:-1, -3])
    Yf[0, :]      = Yf[1, :]     - (Yf[2, :]     - Yf[1, :])
    Yf[-1, :]     = Yf[-2, :]    + (Yf[-2, :]    - Yf[-3, :])

    lon_f, lat_f = to_geo.transform(Xf, Yf)
    return lon_f.astype(np.float32), lat_f.astype(np.float32)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
def write_netcdf(lon_f: np.ndarray, lat_f: np.ndarray, crop: Tuple[int,int,int,int], path: str = OUT_NC):
    j0, j1, i0, i1 = crop
    with Dataset(path, "w") as ds:
        ds.createDimension("jf", lon_f.shape[0])
        ds.createDimension("if", lon_f.shape[1])

        vlon = ds.createVariable("lon_f", "f4", ("jf", "if"), zlib=True, complevel=4, shuffle=True)
        vlat = ds.createVariable("lat_f", "f4", ("jf", "if"), zlib=True, complevel=4, shuffle=True)
        vlon[:, :] = lon_f
        vlat[:, :] = lat_f

        vlon.standard_name = "longitude"
        vlon.units = "degrees_east"
        vlon.long_name = "GBR1 Hydro v2 cell corner longitudes (f-nodes, vertical band crop)"
        vlat.standard_name = "latitude"
        vlat.units = "degrees_north"
        vlat.long_name = "GBR1 Hydro v2 cell corner latitudes (f-nodes, vertical band crop)"

        ds.source_url = OPENDAP_URL
        ds.crs_epsg = CRS_EPSG
        ds.created_utc = datetime.now(timezone.utc).isoformat()
        ds.repo_url = REPO_URL
        ds.description = DESCRIPTION
        ds.crop_j0_i0_j1_i1 = f"{j0},{i0},{j1},{i1}"
        ds.parcels_note = (
            "Use lon_f/lat_f as the common corner grid for U and V with FieldSet.from_c_grid_dataset. "
            "U/V must be sliced to the same j,i window [j0:j1, i0:i1] as recorded in crop_j0_i0_j1_i1."
        )


def write_shapefile_points(lon_f: np.ndarray, lat_f: np.ndarray, path: str = OUT_SHP):
    if gpd is None or Point is None:
        log.warning("GeoPandas/Shapely not available; skipping shapefile output")
        return
    jf, i_f = lon_f.shape
    JF, IF = np.meshgrid(np.arange(jf, dtype=np.int32), np.arange(i_f, dtype=np.int32), indexing="ij")
    pts = [Point(lon_f[j, i], lat_f[j, i]) for j in range(jf) for i in range(i_f)]
    gdf = gpd.GeoDataFrame({"JF": JF.ravel(), "IF": IF.ravel()}, geometry=pts, crs="EPSG:4326")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    gdf.to_file(path)
    log.info("Shapefile written: %s (%d points)", path, jf * i_f)


# -----------------------------------------------------------------------------
# Lightweight validation (projected mean of corners vs centres)
# -----------------------------------------------------------------------------
def quick_validate(lon_f: np.ndarray, lat_f: np.ndarray, lon_c: np.ndarray, lat_c: np.ndarray) -> None:
    to_proj = Transformer.from_crs(4326, CRS_EPSG, always_xy=True)
    Xf, Yf = to_proj.transform(lon_f, lat_f)
    Xc, Yc = to_proj.transform(lon_c, lat_c)

    jdim, idim = lon_c.shape
    jj = np.linspace(0, jdim - 2, 400, dtype=int)
    ii = np.linspace(0, idim - 2, 400, dtype=int)

    dists = []
    for j in jj:
        for i in ii:
            xm = 0.25 * (Xf[j + 1, i + 1] + Xf[j + 1, i + 2] + Xf[j + 2, i + 1] + Xf[j + 2, i + 2])
            ym = 0.25 * (Yf[j + 1, i + 1] + Yf[j + 1, i + 2] + Yf[j + 2, i + 1] + Yf[j + 2, i + 2])
            dists.append(np.hypot(xm - Xc[j, i], ym - Yc[j, i]))
    dists = np.asarray(dists)
    log.info(
        "Validation: max=%.1f m  p50=%.1f m  p95=%.1f m  mean=%.1f m",
        float(np.max(dists)), float(np.percentile(dists, 50)),
        float(np.percentile(dists, 95)), float(np.mean(dists))
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    lon, lat = read_latlon()
    lon_c, lat_c, crop = crop_vertical_band(lon, lat)
    lon_f, lat_f = build_corners(lon_c, lat_c)

    write_netcdf(lon_f, lat_f, crop, OUT_NC)
    write_shapefile_points(lon_f, lat_f, OUT_SHP)
    quick_validate(lon_f, lat_f, lon_c, lat_c)
    log.info("Done. NetCDF: %s", OUT_NC)
    if gpd is not None:
        log.info("Shapefile: %s", OUT_SHP)


if __name__ == "__main__":
    main()
