You are helping implement two Python CLI scripts intended to run in a conda environment for reef particle tracking on the GBR using OceanParcels and eReefs GBR1 2.0. Use Australian English in all comments and messages. Follow the docstrings in 02-extract-gbr1-subcubes.py and 03-simulate-residence.py exactly.

Environment. Target Python 3.13 or later. Assume conda with dependencies xarray, netCDF4, dask, numpy, pandas, geopandas, shapely, pyproj, parcels and matplotlib. Use the most current version of each library. Prefer the conda-forge channel for all libraries. The scripts should import only from the standard library and the listed packages.

Style. Write modern, type hinted Python with clear function boundaries. Keep everything deterministic where random numbers are used by accepting a seed parameter and seeding NumPy. Use pathlib.Path for file paths. Use logging rather than print for progress. Expose a clean argparse CLI for each script. Validate inputs early and report helpful error messages. 

Data model. The eReefs GBR1 2.0 OPeNDAP aggregation exposes curvilinear longitude and latitude as 2-D arrays, time as hourly CF time in AEST and currents u and v with dimensions [time, k, j, i]. For stage 1 work at a single level using k_index=40 by default (-2.35 m). Construct a 10 km buffer (10 grid cells) around each reef polygon using EPSG:3577 and convert back to EPSG:4326. Compute the minimal rectangular i-j window that covers the buffered geographic bounding box and write that rectangular slab to disk. Include longitude, latitude, zc and time. Compress NetCDF output with moderate zlib compression. Record attributes gbr_name, label_id, k_index, zc_m, j_start, j_stop, i_start, i_stop and buffer_km.

Particle modelling. For stage 2 build a FieldSet from a single subcube file with mesh="spherical". Add uniform diffusion by inserting constant fields named Kh_zonal and Kh_meridional when kh>0. Build a ReefMask field by rasterising the reef polygon on the subcube’s grid. Build a DomainMask that is ones everywhere except zeros along the outermost N cells where N=edge_shrink_cells. Kernels are RK4 advection, optional DiffusionUniformKh, a kernel that writes an integer inside flag at outputs by sampling ReefMask, and a kernel that deletes particles when DomainMask is zero. Use dt=5 minutes, outputdt=1 day and runtime of 21 days by default. Seed particles uniformly across the polygon interior by rejection sampling in the polygon bbox. Weekly injections occur at local time Australia/Brisbane at the chosen hour and are converted to naive datetimes before calling Parcels. Particles are not deleted when they leave the reef polygon. At each daily output count how many remain inside and compute a fraction relative to the initial population. The threshold residence time tau10 is the first day when the fraction is less than or equal to 0.10. Save a CSV per reef per year with the weekly curves and tau10. Write a quick MP4 animation for at least the first week in a year to help debug release positions and masks.

Quality. Add unit-scale utility functions with docstrings and type hints: load_grid, ij_window_for_buffered_bbox, extract_reef_year, extract_all_years, fieldset_from_local, polygon_to_mask_field, add_domain_mask, sample_points_in_polygon, weekly_injections and animate_week. Put the CLI in a main() guarded by if __name__ == "__main__". Use small xarray time chunks such as 24 hours and keep memory steady. Handle OPeNDAP and IO exceptions gracefully and ensure temporary or partial files are cleaned up on failure. Where possible, reuse FieldSet objects across weekly runs within a year.

Validation. Log the depth in metres at k_index and the i-j window for each file. After writing a subcube, open it and assert that the time dimension spans the year with hourly cadence. During simulation, log the first three weekly tau10 values per reef to confirm behaviour. Provide a --preview mode that processes a single reef and week to verify the environment and plotting.

Deliverables. Two executable scripts with the CLIs described in the docstrings, ready to run on conda-forge environments. No notebooks. 



I am trying to use the oceanparcels library to simulate particle movement using eReefs GBR1 hydrodynamic. I am trying to use the hourly data on NCI. This has a curvilinear grid. OceanParcels supports curvilinear grids, but the documentation seems to focus on NEMO models. It indicates that Parcels needs to read the f-nodes on the corners. This is not how the grid in eReefs is arranged. The latitute and longitudes correspond to the centroids of the grids. Do I need to synthesise f-nodes before I can run parcels.

I want to create a script that will create f-nodes for the GBR1 curvilinear grid hydro 2. The C-grids are available from the OpenDap service: https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml.html.
Here is the R code that was used to convert the eReefs grids to polygons. This provides more details on structure of the grid:
# Eric Lawrey - Australian Institute of Marine Science
# Converts the eReefs curvilinear grid in the NetCDF file to a shapefile
# This conversion just converts the grid into a set of polygons 
# so that the grid can be rendered on a GIS map. Attached to the polygons
# is the value of the model depth (botz from the NetCDF file, dfData in the shapefile). 
# Where this variable is no NaN the grid cell is wet in the model and so should
# have data available at the surface.

# This script is written to create shapefiles for the two grids
# used in eReefs modelling (https://ereefs.org.au).
# This conversion needs an example GBR4 and GBR1 data files from 
# which to extract the grids. These files are so large that they
# are not saved as part of these scripts (as they are >30GB). We don't
# download them in the script due to their size.
# They can be downloaded from
# http://dapds00.nci.org.au/thredds/catalogs/fx3/catalog.html
# Example files are:
# http://dapds00.nci.org.au/thredds/fileServer/fx3/gbr4_v2/gbr4_simple_2020-01.nc
# http://dapds00.nci.org.au/thredds/fileServer/fx3/gbr1_2.0/gbr1_simple_2020-01-14.nc
# 
# These files should be saved in C:\temp for the script to access them. 
# Alternatively adjust the path in the script to the files.


library(ncdf4)
library(maptools)


# Make the working directory match the location of this script. This
# will ensure the relative paths make sense.
# https://stackoverflow.com/questions/3452086/getting-path-of-an-r-script/35842176#35842176
setwd(utils::getSrcDirectory(function(x) {x})[1])

gbr4ncfile <- 'C:/temp/gbr4_simple_2020-01.nc'
gbr1ncfile <- 'C:/temp/gbr1_simple_2020-01-14.nc'


#
# Given the original points x generate the 'o' points using the distances
# between the a's
#  o1    o2    o3
#     a1    a2
#  o4    o5    o6
#     a3    a4  
#  o7    o8    o9
# Note the array would normally be much bigger than that shown above
# and the algorithm must work in this case.
#x <- matrix(c(0,10,20,1,11,21,2,12,23),nrow=3,ncol=3)
#x <- matrix(c(0,0,0,10,10,10,20,20,20),nrow=3,ncol=3)
#x <- matrix(c(0,1,2,3,10,11,12,13,20,21,22,23),nrow=4,ncol=3)
# Take a maxtrix of X values or Y values
# Adapted from NcWMS - CurvilinearGrid.java
makeCorners <- function (midPoints) {
    ncols <- ncol(midPoints)
    nrows <- nrow(midPoints)
    edges <- matrix(0, nrows+1, ncols+1)
    a1 <- midPoints[1:nrows-1,1:ncols-1]
    # Shift down 1
    a3 <- midPoints[2:nrows,1:ncols-1]
    # Shift across 1
    a2 <- midPoints[1:nrows-1,2:ncols]
    # Shift down and across 1
    a4 <- midPoints[2:nrows,2:ncols]

    edges[2:nrows,2:ncols] <- (a1+a2+a3+a4)/4
    
    # Extrapolate to left and right exterior points
    edges[2:nrows,1] <- edges[2:nrows,2] - (edges[2:nrows,3] - edges[2:nrows,2])
    edges[2:nrows,ncols+1] <- edges[2:nrows,ncols] + (edges[2:nrows,ncols] - edges[2:nrows,ncols-1])
    
    # Extrapolate to the first and last rows
    edges[1,] <- edges[2,] - (edges[3,] - edges[2,])
    edges[nrows+1,] <- edges[nrows,] + (edges[nrows,] - edges[nrows-1,])
    
    return(edges)
}

# Create a polygon around each four corner coordinates (o1 ...). 
# The corners can be calculated using makeCorners
# lat and long should correspond to the coordinates for o1 ...
#  o1    o2    o3
#     a1    a2
#  o4    o5    o6
#     a3    a4  
#  o7    o8    o9
# The centre coordinates a1 ... correspond to the centroids of 
# the original grid.
# 
# From
# http://stackoverflow.com/questions/26620373/spatialpolygons-creating-a-set-of-polygons-in-r-from-coordinates
# 
#polys <- SpatialPolygons(
#    mapply(
#        function(poly, id) {
#            xy <- matrix(poly, ncol=2, byrow=TRUE)
#            Polygons(list(Polygon(xy)), ID=id)
#        }, 
#        split(square, row(square)), ID)
#    )

# Create a matrix of polygon coordinates
# Returns a SpatialPolygonsDataFrame with a data frame containing
# the grid cell indicies of each polygon.
# Data must be full grid with one row per grid cell.
toPolygons <- function(lat, long, data) {
    pts <- proc.time()
    # Calculate corners around the lat and long locations
    # These matricies will be one row and column larger
    # than lat and long.
    longGrid <- makeCorners(long)
    latGrid <- makeCorners(lat)
    nrows <- nrow(latGrid)
    ncols <- ncol(longGrid)
    
    # Assertion checks
    if (nrow(latGrid) != nrow(longGrid)) {
        stop(paste("nrows of latGrid and longGrid don't match:",nrow(latGrid),nrow(longGrid)))
    }
    if (ncol(latGrid) != ncol(longGrid)) {
        stop(paste("ncol of latGrid and longGrid don't match:",ncol(latGrid),ncol(longGrid)))
    }
    if (nrow(lat) != nrow(long)) {
        stop(paste("nrows of lat and long don't match:",nrow(lat),nrow(long)))
    }
    if (ncol(lat) != ncol(long)) {
        stop(paste("ncol of lat and long don't match:",ncol(lat),ncol(long)))
    }
    if (nrow(latGrid)!=(nrow(lat)+1)) {
        stop(paste0("latGrid (",nrow(latGrid),
            ") should have one more row than lat (",nrow(latGrid),"), it doesn't"))
    }
    if (ncol(latGrid)!=(ncol(lat)+1)) {
        stop(paste0("latGrid (",ncol(latGrid),
            ") should have one more column than lat (",ncol(latGrid),"), it doesn't"))
    }
    if (nrow(longGrid)!=(nrow(long)+1)) {
        stop(paste0("longGrid (",nrow(longGrid),
            ") should have one more row than long (",nrow(longGrid),"), it doesn't"))
    }
    if (ncol(longGrid)!=(ncol(long)+1)) {
        stop(paste0("longGrid (",ncol(longGrid),
            ") should have one more column than long (",ncol(longGrid),"), it doesn't"))
    }
    
    if (ncol(lat) *nrow(lat) != nrow(data)) {
        stop(paste0("Data length does not match the grid size"))
    }
    
    y1 <- as.vector(latGrid[1:nrows-1,1:ncols-1])
    # Shift down 1
    y2 <- as.vector(latGrid[2:nrows,1:ncols-1])
    # Shift across 1
    y4 <- as.vector(latGrid[1:nrows-1,2:ncols])
    # Shift down and across 1
    y3 <- as.vector(latGrid[2:nrows,2:ncols])
    
    x1 <- as.vector(longGrid[1:nrows-1,1:ncols-1])
    # Shift down 1
    x2 <- as.vector(longGrid[2:nrows,1:ncols-1])
    # Shift across 1
    x4 <- as.vector(longGrid[1:nrows-1,2:ncols])
    # Shift down and across 1
    x3 <- as.vector(longGrid[2:nrows,2:ncols])
    
    boxesRaw <- cbind(x1,y1,x2,y2,x3,y3,x4,y4,x1,y1)
    
    # Assertion
    if (nrow(boxesRaw) != (nrow(lat) * ncol(lat))) {
        stop("There should be one boxesRaw per cell in lat, there aren't")
    }
    
    # Some boxes will have NaN elements in them.
    # Should remove all boxes with any NaN corners as
    # the polygon creation will fail.
    goodboxIndicies <- complete.cases(boxesRaw)
    
    boxes <- boxesRaw[goodboxIndicies, ]
    
    ID <- paste0(row(lat),"_",col(lat))[goodboxIndicies]
    print(paste("Prepared for polygon creation: ",round((proc.time()-pts)[3],1),"sec"))
    # Remove cells with NaN values. Assume lat and long are same
    
    require(maps)
    polys <- SpatialPolygons(
        mapply(
            function(poly, id) {
                xy <- matrix(poly, ncol=2, byrow=TRUE)
                Polygons(list(Polygon(xy)), ID=id)
            }, 
            split(boxes, row(boxes)), ID),
        proj4string=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    )
    # Trim off the last column and row as these are extensions to
    # create the polygon corners.
    rowData <- as.vector(row(lat))
    colData <- as.vector(col(lat))
    dfLatLong <- data.frame(row=rowData, col=colData)[goodboxIndicies,]
    
    # Using depth as the variable name here is a bit of a hack.
    # This is the name of the variable that will appear in the final
    # shapefile. Ideally this name would be passed in as a parameter
    # to this function, however I could quite workout how to get this
    # to work. In this script we are only extracting 'depth' so this
    # hack is OK.
    depth <- data[goodboxIndicies,]
    dataTable <- cbind(depth,dfLatLong)
    row.names(dataTable) <- ID
    spp <- SpatialPolygonsDataFrame(polys,data=dataTable)
    print(paste("SpatialPolygonsDataFrame complete: ",round((proc.time()-pts)[3],1),"sec"))
    return(spp)
} 


# Utility function for coordinating the creation of the output shapefile.
createDepthShpFile <- function (outname, ncfile) {
  # Setup outputs, checking if they have already been created
  outdir <- dirname(outname)
  dir.create(outdir, showWarnings = FALSE)
  
  # If the output files already exist then skip regenerating them
  # this is done as each step can take a while. This makes the
  # script more restartable.
  outfile <- paste0(outname,'.shp')
  if (file.exists(outfile)) {
	  print(paste0('Skipping shapefile creation as output already exists: ',outfile))
  } else {
  
    if (!file.exists(ncfile)) {
  	  stop(paste0("NetCDF file not found. See script comments for where to download the input data. ",ncfile))
    }
    
    nc <- nc_open(ncfile)
    varData <- ncvar_get( nc, nc$var$botz, start=c(1,1), count=nc$var$botz$varsize )
    lat <- ncvar_get(nc, 'latitude')
    long <- ncvar_get(nc, 'longitude')
    polys <- toPolygons(lat, long, data.frame(botz=as.vector(varData)))
    
    
    prjStr <-'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
    cat(prjStr, file=paste0(outname,'.prj'))
    writePolyShape(polys,outname)
    
    print(paste0("Shapefile generated: ",outname,".shp"))

  }
}

outname <- 'export/GBR_AIMS_eReefs-grid-shapefiles_GBR4-hydro-v2'
print('===== Creating GBR 4 grid shapefile ====')
createDepthShpFile(outname, gbr4ncfile)


outname <- 'export/GBR_AIMS_eReefs-grid-shapefiles_GBR1-hydro-v2'
print('===== Creating GBR 1 grid shapefile ====')
createDepthShpFile(outname, gbr1ncfile)

I want to create a Python files that will download the curvilinear grid from the opendap service, then convert the grid to f-nodes in a form suitable for use in OceanParcels. I don't want you to generate the code, but create a prompt with all the relevant information. I will be generating the code in VSCode using GPT-5. I want from you is a detailed script specification with additional relevent technical information.

Great, I can give you a tight, copy-paste-ready prompt/spec you can drop into VS Code for GPT-5 to generate a Python tool that builds f-nodes for GBR1. I will include the exact dataset, variable names, shapes, the f-node maths, CF metadata you should write, validation steps, and how the result plugs into OceanParcels.

# Prompt for GPT-5 in VS Code

You are writing a production-quality Python script that builds an OceanParcels-ready f-node grid for the eReefs GBR1 hydrodynamic model (Hydro v2). The script must read the curvilinear grid from the NCI THREDDS OPeNDAP endpoint and produce cell-corner longitude and latitude arrays on the model’s native curvilinear grid. The output will be a small NetCDF file with the corner arrays, plus optional GeoJSON for visual QA. The script must not download the massive time-varying velocity arrays. It must be deterministic, robust to NaNs at the domain perimeter, and memory-efficient.

The upstream grid is available via OPeNDAP at:
[https://thredds.nci.org.au/thredds/dodsC/fx3/model\_data/gbr1\_2.0.ncml.html](https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml.html)

Confirm the following before proceeding. This dataset exposes two 2-D coordinate variables `latitude[j,i]` and `longitude[j,i]` with shapes j=2389 and i=510 at the time of writing. It also contains `botz[j,i]` and time-varying fields such as `u[time,k,j,i]` and `v[time,k,j,i]`. The `u` and `v` variables carry CF standard\_names `eastward_sea_water_velocity` and `northward_sea_water_velocity`, respectively, and use the same `latitude` and `longitude` coordinates. The time units are `days since 1990-01-01 00:00:00 +10`. Do not load `u`, `v`, or `eta` etc for this task. Only read the static `latitude`, `longitude` (and optionally `botz` for masking) from OPeNDAP. ([thredds.nci.org.au][1])

Explain in comments why Parcels needs f-nodes. Parcels performs C-grid interpolation using barycentric coordinates with `FieldSet.from_c_grid_dataset`, which requires the four cell corners (f-points) and expects U, V (and W) to share the same corner grid. Parcels does not need the exact U or V node positions when the corner grid is supplied. Cite the Parcels grid indexing docs and the enforcement that U, V, W must share the f-point dimensions. ([docs.oceanparcels.org][2])

Also note in comments that eReefs GBR1 is generated by SHOC on an orthogonal curvilinear C-grid, but the public NCI files advertise `u` and `v` as eastward and northward components on the same `latitude` and `longitude` as other fields. It is therefore possible to drive Parcels on an A-grid using centre coordinates, but barycentric C-grid interpolation generally yields better accuracy on curvilinear meshes. This script therefore synthesises f-nodes even though centre-only use is viable. Add the source note that GBR1 is 510 x 2390 horizontally. ([thredds.nci.org.au][1])

Describe the outputs. The primary output is a NetCDF file, default name `gbr1_fnodes.nc`, containing two 2-D arrays `lon_f[jf, if]` and `lat_f[jf, if]` with shapes `jf = j + 1` and `if = i + 1`. Include CF attributes on each variable: `standard_name = longitude` or `latitude`, `units = degrees_east` or `degrees_north`, and a short `long_name` clarifying that these are cell corner coordinates for GBR1 Hydro v2. Include the original grid sizes as global attributes and provenance metadata indicating the OPeNDAP URL and the creation date in UTC. Optionally write CF “bounds style” corner variables `lon_vertices[j, i, nv]` and `lat_vertices[j, i, nv]` with `nv = 4` ordered consistently around each cell for compatibility with other tools. Reference that corner arrays are a common CF pattern and are sometimes called `*_bounds` or `*_vertices`. ([ncl.ucar.edu][3])

Specify the algorithm to construct corners from centres. Do not compute arithmetic means in geographic coordinates near the antimeridian. For GBR1 the domain is entirely in eastern longitudes and far from the antimeridian, but you must nevertheless implement the computation in a projected planar CRS and transform back. Use `pyproj` to project centre coordinates from `EPSG:4326` to an Australia-appropriate projection. Prefer `EPSG:3577` (GDA94 Australian Albers) or `EPSG:3112` if you want a conformal alternative. Document the chosen CRS in metadata. After projecting to x and y with shapes `[j,i]`, compute the interior corners by averaging the four surrounding centres:
Xf\[j+1, i+1] = 0.25 \* (Xc\[j, i] + Xc\[j+1, i] + Xc\[j, i+1] + Xc\[j+1, i+1])
and the same for Y. Then extrapolate to the exterior rows and columns by first-order differences to obtain the outer ring of corners:
Xf\[1:-1, 0] = Xf\[1:-1, 1] - (Xf\[1:-1, 2] - Xf\[1:-1, 1])
Xf\[1:-1, -1] = Xf\[1:-1, -2] + (Xf\[1:-1, -2] - Xf\[1:-1, -3])
Xf\[0, :] = Xf\[1, :] - (Xf\[2, :] - Xf\[1, :])
Xf\[-1, :] = Xf\[-2, :] + (Xf\[-2, :] - Xf\[-3, :])
Apply the same for Y. This is the same logic used by ncWMS style curvilinear corner synthesis from centre points. After extrapolation, transform Xf, Yf back to lon, lat. State explicitly that j increases south to north and i increases west to east, and keep the indexing consistent so that `lon_f[0,0]` and `lat_f[0,0]` are the north-west outer corner of the full grid if that matches the input row ordering. Cite ncWMS style approaches and CF corner practices. ([reading-escience-centre.gitbooks.io][4])

Spell out masking and NaN handling. Expect `latitude` and `longitude` to be finite everywhere on the structured grid. If the centre arrays include NaNs at padding rows or columns in some model vintages, restrict the interior averaging to valid four-centre neighbourhoods only and fall back to nearest neighbour propagation when any of the four are NaN. For the exterior ring, if either of the difference terms becomes NaN, propagate from the nearest valid interior along that row or column. If a `botz` threshold mask is desired for later QA, treat `botz <= -98` or the `botz` attribute `missing_value` as land. Do not use `botz` to mask the grid when writing corner arrays, because Parcels expects a complete corner grid even where velocities are masked. The `botz` metadata shows `missing_value = -99.0` and `outside = 9999`. Cite the upstream attributes. ([thredds.nci.org.au][1])

Describe the file IO and performance expectations. Use xarray backed by netCDF4 to open the OPeNDAP dataset with chunking on `j` and `i` only, for example blocks in the order of a few hundred in `j` and one to two hundred in `i`. Immediately subset the dataset to `["latitude", "longitude"]` (and optionally `["botz"]`) to avoid pulling time or depth variables. Convert to NumPy for the corner computation. The centre grid size 2389 x 510 fits comfortably in memory on a laptop. The corner arrays will be 2390 x 511. Compress the output NetCDF with zlib compression level 4 to keep it small.

Detail the requested CLI, configuration, and outputs. The script should be callable as a module and as a command line tool. Accept arguments for the OPeNDAP URL, an optional local path to a NetCDF file in case OPeNDAP is unavailable, the target CRS EPSG code, the output NetCDF path, and an optional GeoJSON path for writing a decimated polygon mesh for sanity checking. In GeoJSON mode, thin the grid by a user factor in both j and i to avoid huge files and write each surviving cell polygon using the four corners in anti-clockwise order. Add an option to save `lon_vertices` and `lat_vertices` alongside `lon_f` and `lat_f`. For reproducibility, log the SHA-256 of the input file if a local file is used, or record the HTTP Last-Modified header if available for OPeNDAP.

Provide thorough validation within the script. After constructing corners, sample at least a hundred random interior cells. For each, compute the arithmetic mean of the four corners in projected coordinates and confirm that it is within a small tolerance of the original centre point. Report the maximum centre-minus-mean distance in metres. Compute geodesic areas for a small sample of cells using `pyproj.Geod` to confirm strictly positive areas and reasonable magnitudes for 1 km cells. Also confirm monotonicity of i and j across the f-grid by checking that adjacent corner longitudes are generally increasing eastward along rows and adjacent corner latitudes increasing northward along columns, allowing for local curvature. Assert the expected shapes and write them into the global attributes.

Explain clearly how the result is meant to be used in Parcels. The output `lon_f` and `lat_f` arrays are to be passed as the f-node `lon` and `lat` to `FieldSet.from_c_grid_dataset`, using the same f-node grid for U and V, and providing the U and V data files with their native variables `u` and `v`. Because the NCI public files advertise `u` and `v` as eastward and northward components already, no rotation is required. If a user prefers not to use Parcels’ C-grid indexing, they could instead build an A-grid `FieldSet` with 2-D `lon` and `lat` using centre points and bilinear interpolation, but that is not the focus of this tool. Cite Parcels docs for c-grid expectations and the advisory that U and V share the same corner grid. ([docs.oceanparcels.org][2])

Document acceptance criteria in code comments. First, opening the OPeNDAP URL must only fetch the coordinate arrays and relevant attributes. Second, the script must write a NetCDF file with `lon_f` and `lat_f` of shape `(j+1, i+1)` and the expected CF metadata. Third, the geometric validation must print a summary with the maximum centre-minus-corner-mean offset and the number of cells sampled. Fourth, if GeoJSON output is requested, the file must contain polygons with four vertices per cell in a consistent winding order. Fifth, the output must include global attributes listing the source OPeNDAP URL, the CRS used for computation, and the date and time the file was created.

Add inline documentation pointing to helpful background. Cite Parcels’ curvilinear and indexing tutorials for context, and the eAtlas grid-to-polygon reference that uses an equivalent corner synthesis logic from the same GBR1 data, to justify the averaging and extrapolation scheme. Provide URLs in comments to assist users who want to cross-check the shapes and grid description. ([docs.oceanparcels.org][5])

Finally, include a brief section in comments on limitations and extensions. Note that the extrapolation scheme is first order on the outer ring and could be replaced by a shape-preserving approach near highly skewed cells. Note also that because GBR1 does not cross the antimeridian, unwrapping longitudes is not required here. If a future eReefs grid does cross the antimeridian, add a longitude unwrap step before projection. Suggest writing a small unit test that loads the published GBR1 and GBR4 shapefile products and checks that the derived polygon corners match within a small tolerance after cell-centre snapping. Link users to the AIMS shapefile dataset as an external cross-check. ([catalogue.eatlas.org.au][6])

# Notes and sources you should weave into comments

The NCI OPeNDAP page for `gbr1_2.0.ncml` shows `latitude[j,i]`, `longitude[j,i]`, `botz[j,i]`, and velocities `u` and `v` with standard CF names and coordinates referencing the same latitude and longitude arrays. This confirms that the grid is curvilinear with coordinates at cell centres and that velocities are eastward and northward components. Use only those coordinate arrays from OPeNDAP. ([thredds.nci.org.au][1])

Parcels’ grid indexing documentation states that for Arakawa C-grids it interpolates on barycentric coordinates using the four corners, and that `FieldSet.from_c_grid_dataset` sets `cgrid_velocity` interpolation and expects U, V (and W) on the same f-node grid. The FieldSet code enforces that U, V, W share identical f-node dimensions. ([docs.oceanparcels.org][2])

Corner arrays are a well-established pattern in CF workflows. If you choose to emit `*_vertices` alongside the f-node arrays, the shape should be `[j,i,4]` with a consistent vertex order. ([ncl.ucar.edu][3])

The AIMS eAtlas repository and metadata show a prior centre-to-corner conversion for GBR1 and GBR4 that underpins released shapefiles of the grids, reinforcing that centre-to-corner synthesis is appropriate for these meshes and that the grid is static over time for a given run. ([GitHub][7])

The CSIRO and eReefs pages provide context on grid size and model configuration and may be referenced in script comments that describe the domain and dimensions. ([CSIRO Research][8])

If you implement the script according to this spec, it will produce a compact NetCDF file with `lon_f` and `lat_f` that can be supplied directly to Parcels when building a `FieldSet` on a C-grid, while also providing optional CF vertices and a light GeoJSON for QC maps.

[1]: https://thredds.nci.org.au/thredds/dodsC/fx3/model_data/gbr1_2.0.ncml.html "OPeNDAP Dataset Query Form"
[2]: https://docs.oceanparcels.org/en/latest/examples/documentation_indexing.html?utm_source=chatgpt.com "Grid indexing — Parcels Documentation"
[3]: https://www.ncl.ucar.edu/Document/Functions/ESMF/curvilinear_to_SCRIP.shtml?utm_source=chatgpt.com "curvilinear_to_SCRIP - NCAR Command Language (NCL)"
[4]: https://reading-escience-centre.gitbooks.io/ncwms-user-guide/content/?utm_source=chatgpt.com "ncWMS User Guide"
[5]: https://docs.oceanparcels.org/en/latest/examples/tutorial_nemo_curvilinear.html?utm_source=chatgpt.com "Curvilinear grids — Parcels Documentation"
[6]: https://catalogue.eatlas.org.au/geonetwork/srv/api/records/43ff162c-8132-41cd-8547-76a1acf58105?utm_source=chatgpt.com "eReefs GBR1 and GBR4 model boundary and grid in shapefile format (AIMS)"
[7]: https://github.com/eatlas/GBR_AIMS_eReefs-grid-shapefiles?utm_source=chatgpt.com "eatlas/GBR_AIMS_eReefs-grid-shapefiles - GitHub"
[8]: https://research.csiro.au/ereefs/models/models-about/models-hydrodynamics/?utm_source=chatgpt.com "Hydrodynamics – eReefs Research"
