## Setting up Python
1. Create the Conda environment. 
```bash
cd {path to this folder} 
conda env create -f environment.yml
```
2. Activate the environment
```bash
conda activate reef-residency
```
3. Run the code:
```bash
python 02-extract-gbr1-subcubes.py
```

You are helping implement two Python CLI scripts intended to run in a conda environment for reef particle tracking on the GBR using OceanParcels and eReefs GBR1 2.0. Use Australian English in all comments and messages. Follow the docstrings in 02-extract-gbr1-subcubes.py and 03-simulate-residence.py exactly.

Environment. Target Python 3.13 or later. Assume conda with dependencies xarray, netCDF4, dask, numpy, pandas, geopandas, shapely, pyproj, parcels and matplotlib. Use the most current version of each library. Prefer the conda-forge channel for all libraries. The scripts should import only from the standard library and the listed packages.

Style. Write modern, type hinted Python with clear function boundaries. Keep everything deterministic where random numbers are used by accepting a seed parameter and seeding NumPy. Use pathlib.Path for file paths. Use logging rather than print for progress. Expose a clean argparse CLI for each script. Validate inputs early and report helpful error messages. 

Data model. The eReefs GBR1 2.0 OPeNDAP aggregation exposes curvilinear longitude and latitude as 2-D arrays, time as hourly CF time in AEST and currents u and v with dimensions [time, k, j, i]. For stage 1 work at a single level using k_index=40 by default (-2.35 m). Construct a 10 km buffer (10 grid cells) around each reef polygon using EPSG:3577 and convert back to EPSG:4326. Compute the minimal rectangular i-j window that covers the buffered geographic bounding box and write that rectangular slab to disk. Include longitude, latitude, zc and time. Compress NetCDF output with moderate zlib compression. Record attributes gbr_name, label_id, k_index, zc_m, j_start, j_stop, i_start, i_stop and buffer_km.

Particle modelling. For stage 2 build a FieldSet from a single subcube file with mesh="spherical". Add uniform diffusion by inserting constant fields named Kh_zonal and Kh_meridional when kh>0. Build a ReefMask field by rasterising the reef polygon on the subcube’s grid. Build a DomainMask that is ones everywhere except zeros along the outermost N cells where N=edge_shrink_cells. Kernels are RK4 advection, optional DiffusionUniformKh, a kernel that writes an integer inside flag at outputs by sampling ReefMask, and a kernel that deletes particles when DomainMask is zero. Use dt=5 minutes, outputdt=1 day and runtime of 21 days by default. Seed particles uniformly across the polygon interior by rejection sampling in the polygon bbox. Weekly injections occur at local time Australia/Brisbane at the chosen hour and are converted to naive datetimes before calling Parcels. Particles are not deleted when they leave the reef polygon. At each daily output count how many remain inside and compute a fraction relative to the initial population. The threshold residence time tau10 is the first day when the fraction is less than or equal to 0.10. Save a CSV per reef per year with the weekly curves and tau10. Write a quick MP4 animation for at least the first week in a year to help debug release positions and masks.

Quality. Add unit-scale utility functions with docstrings and type hints: load_grid, ij_window_for_buffered_bbox, extract_reef_year, extract_all_years, fieldset_from_local, polygon_to_mask_field, add_domain_mask, sample_points_in_polygon, weekly_injections and animate_week. Put the CLI in a main() guarded by if __name__ == "__main__". Use small xarray time chunks such as 24 hours and keep memory steady. Handle OPeNDAP and IO exceptions gracefully and ensure temporary or partial files are cleaned up on failure. Where possible, reuse FieldSet objects across weekly runs within a year.

Validation. Log the depth in metres at k_index and the i-j window for each file. After writing a subcube, open it and assert that the time dimension spans the year with hourly cadence. During simulation, log the first three weekly tau10 values per reef to confirm behaviour. Provide a --preview mode that processes a single reef and week to verify the environment and plotting.

Deliverables. Two executable scripts with the CLIs described in the docstrings, ready to run on conda-forge environments. No notebooks. 
