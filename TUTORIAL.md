# Tutorial: estimating reef water residency with OceanParcels and eReefs

This tutorial explains the background needed to understand and adapt the two scripts you will run. It assumes you are an ecologist rather than an ocean modeller. By the end you should be comfortable with what the code is doing, why it is structured in two stages, and what the main limitations are.

## What we are trying to measure

Residency time, in this context, is a way of describing how long water remains associated with an individual reef before being flushed away by currents and mixing. If you release a passive tracer such as eDNA on a reef at midday, the amount of tracer present on the reef usually declines through time as water is stirred and advected. We treat particles as tiny packets of water and ask a simple question at the end of each day: how many of the originally released packets are still over the reef. Tracking that fraction across days gives a survival curve. A threshold, such as the day when only 10 percent remain, is our operational definition of residency time. Repeating the release week by week and year by year lets us see seasonal patterns and interannual variability.

## How we will achieve it

The workflow has two stages. First we extract, from the national eReefs model, a tidy subset of hourly currents around each reef. This produces compact NetCDF files on your computer that are quick to read. Second we use the OceanParcels toolbox to run weekly particle simulations on those local files. Particles are released inside the reef polygon, move with the modelled currents plus a simple representation of small scale stirring, and are counted as inside the reef whenever they lie within the polygon. We do not delete a particle just because it steps outside the reef, because tides commonly push water off and back on again. We only delete particles that leave the outer edge of the modelling domain. The daily fraction of particles inside is our proxy for concentration. The threshold day is the residency time for that weekly snapshot.

## What OceanParcels is and what it does

OceanParcels is an open source Lagrangian tracking library that simulates the movement of many independent particles through time. Lagrangian means we follow the paths of the particles themselves rather than solving equations on a fixed grid. The library reads modelled currents on a grid and interpolates them to each particle’s location at each time step, then integrates the movement forward. We use the fourth order Runge Kutta scheme for accurate advection and add an optional uniform horizontal diffusivity to represent unresolved stirring. Parcels lets us add small “kernels” of logic, for example to flag whether a particle is inside the reef polygon at the current time, or to delete it if it reaches the edge of the domain. Parcels writes out NetCDF files containing the positions and any variables we attach to each particle.

## What eReefs provides

eReefs is a suite of hydrodynamic and biogeochemical models for the Great Barrier Reef. We use the hydrodynamic component known as GBR1, which resolves the shelf at roughly 1 km grid spacing and provides hourly currents. The model contains tides and wind driven flow, and represents the coastline and reefs at the scale of a kilometre. At this resolution the model cannot resolve the smallest channels or reef flats but it does a reasonable job of the larger scale currents that govern flushing. The dataset is large and long in time, so rather than stream the entire domain we take small spatial windows centred on each reef and save them locally.

## A short note on NetCDF and OPeNDAP

NetCDF is a common scientific file format used for multidimensional arrays such as longitude by latitude by time. It stores metadata like variable names and units, which helps tools read the data correctly. OPeNDAP is a web service that lets you read parts of large NetCDF datasets over the internet as if they were local. In our first stage we connect to eReefs via OPeNDAP to read only the variables and the grid cells we need. We then save that subset to a local NetCDF file. Working locally is faster and makes debugging the second stage repeatable.

# How the eReefs grid is organised

GBR1 uses a curvilinear grid that follows the coastline and shelf. The dataset stores two two dimensional coordinate arrays called longitude and latitude. Each element of those arrays is the geographic location of the centre of a model cell. Think of these as the centroids of a gently tilted mesh rather than the corners of a perfect rectangle. The valid ocean cells form an L shaped patch within the full rectangular index space. Outside that patch the centre coordinates are NaN, which is a deliberate way to mark unused parts of the computational frame in the Coral Sea. 


## Stage 1 - Data extraction

The extraction script reads your reef layer, which contains polygons and identifiers, and for each selected reef builds a 10 km buffer in metres using a projected coordinate system. It then looks up the model’s curvilinear longitude and latitude arrays and finds the minimal grid aligned rectangle that covers the buffered reef box. We select a single depth level representative of a near surface sample, k=40 which is about −2.5 m, and pull hourly u and v currents for the chosen year. The script writes a small NetCDF file to disk and records in the file’s attributes the grid window and depth used. Because the original domain is huge, this local file is tiny by comparison and can be reopened instantly in the second stage. Extracting once and reusing the result is the key to quick iteration when you are tuning parameters and checking outputs.

## Stage 2 - Simulation

The simulation script opens one of those subcubes and builds a Parcels field set from it. It rasterises the reef polygon onto the model grid to make a mask whose value is 1 inside the reef and 0 outside. It also creates a domain mask that is 1 everywhere except for the outermost one or two grid cells, which are set to 0. This gives a clean boundary where particles are deleted if they drift too far away. Each week of the year, at a fixed local time, the script randomly places a large number of particles uniformly across the reef interior. It then advances them every five minutes using the model currents and an optional small horizontal diffusivity that represents stirring by eddies smaller than the grid. At the end of each day it checks each particle against the reef mask and notes whether it is inside. A particle is allowed to move in and out with the tide. Only when it reaches the domain edge is it removed from the experiment. From the daily tally of particles inside the reef we form a fraction that starts at one and typically declines. The day when the fraction first drops to a threshold such as 0.10 is recorded as the residency time for that week. The script also saves a simple animation so you can see that the release locations, the mask and the flow field look sensible.

## Why one depth and what you lose by doing that

We work at a single depth near the surface, approximated by the model’s level k=40 which is close to −2.5 m. This choice lines up with many sampling programs for eDNA and makes the extraction and the tracking fast and easy to understand. The trade off is that we ignore vertical shear and vertical mixing. In many shelf settings the near surface currents capture most of the horizontal flushing signal, but if your question depends strongly on vertical processes you may later extend the approach to two or three depths or to full 3D advection. As you move in that direction the setup becomes more complex and you will need to choose a vertical mixing coefficient and consider surface and bottom boundary conditions.

