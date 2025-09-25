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

## Why the grid looks tilted and what a curvilinear grid means

GBR1 uses a curvilinear grid. Instead of longitudes and latitudes forming perfect straight lines, the grid lines curve gently to follow the coastline and fit more cells into interesting places. This is efficient for the model but it makes simple geographic boxes a poor fit. The solution is to find the smallest rectangle in grid index space that covers the geographic area we care about. Conceptually you draw a geographic bounding box around a reef buffered by about 10 km, find which curvilinear cells intersect that box, then take the minimum and maximum i and j indices that contain them. The result is a neat i-j rectangle that exactly matches the model’s storage layout. We use that rectangle for extraction and for the later particle runs.

## Stage 1 in plain language

The extraction script reads your reef layer, which contains polygons and identifiers, and for each selected reef builds a 10 km buffer in metres using a projected coordinate system. It then looks up the model’s curvilinear longitude and latitude arrays and finds the minimal grid aligned rectangle that covers the buffered reef box. We select a single depth level representative of a near surface sample, k=40 which is about −2.5 m, and pull hourly u and v currents for the chosen year. The script writes a small NetCDF file to disk and records in the file’s attributes the grid window and depth used. Because the original domain is huge, this local file is tiny by comparison and can be reopened instantly in the second stage. Extracting once and reusing the result is the key to quick iteration when you are tuning parameters and checking outputs.

## Stage 2 in plain language

The simulation script opens one of those subcubes and builds a Parcels field set from it. It rasterises the reef polygon onto the model grid to make a mask whose value is 1 inside the reef and 0 outside. It also creates a domain mask that is 1 everywhere except for the outermost one or two grid cells, which are set to 0. This gives a clean boundary where particles are deleted if they drift too far away. Each week of the year, at a fixed local time, the script randomly places a large number of particles uniformly across the reef interior. It then advances them every five minutes using the model currents and an optional small horizontal diffusivity that represents stirring by eddies smaller than the grid. At the end of each day it checks each particle against the reef mask and notes whether it is inside. A particle is allowed to move in and out with the tide. Only when it reaches the domain edge is it removed from the experiment. From the daily tally of particles inside the reef we form a fraction that starts at one and typically declines. The day when the fraction first drops to a threshold such as 0.10 is recorded as the residency time for that week. The script also saves a simple animation so you can see that the release locations, the mask and the flow field look sensible.

## Why one depth and what you lose by doing that

We work at a single depth near the surface, approximated by the model’s level k=40 which is close to −2.5 m. This choice lines up with many sampling programs for eDNA and makes the extraction and the tracking fast and easy to understand. The trade off is that we ignore vertical shear and vertical mixing. In many shelf settings the near surface currents capture most of the horizontal flushing signal, but if your question depends strongly on vertical processes you may later extend the approach to two or three depths or to full 3D advection. As you move in that direction the setup becomes more complex and you will need to choose a vertical mixing coefficient and consider surface and bottom boundary conditions.

## Why repeat across weeks and years

Hydrodynamic conditions on the shelf vary over tidal, weather and seasonal timescales, and also from year to year. A single release tells you what happened on that day only. By repeating the release once a week through the year you sample a wide range of tides and wind patterns, which produces a distribution of residency times. Aggregating by month or season reveals the climatology. Extending the analysis across multiple years lets you compare periods with different climate drivers and check whether results are robust.

## Choosing particle numbers and other parameters

Particle tracking is a Monte Carlo method, which means results stabilise as the number of particles increases. A few thousand particles per weekly release is usually enough for stable fractions at daily timescales. The time step of five minutes is short compared with the tidal period, which keeps numerical errors small. The horizontal diffusivity controls the rate at which particles spread beyond what the grid can resolve. Values between about 5 and 20 m² s⁻¹ are commonly used at 1 km resolution. The runtime sets how long after release you keep counting. Three weeks is a useful default because many reefs have flushed substantially by then. The edge shrink parameter trims away the outermost cells of the subcube to avoid keeping particles that graze the boundary.

## Limitations of the approach

There are several important limitations to keep in mind when interpreting results. The model grid is about 1 km, so the exact shape of a small reef, passages through the reef crest and narrow channels will be simplified. Tides are present but the interaction with fine scale topography is smoothed. The diffusion we add is a simple, spatially uniform representation of unresolved stirring and is best treated as a tuning parameter rather than a directly measured quantity. Working at a single depth ignores vertical variability and vertical mixing of eDNA. The particles are passive and do not represent decay, settling or active biology. Our domain is a finite window around each reef, so water that truly leaves the area is deleted even if the broader ocean might bring it back days later. None of these issues invalidate the method, but they set the scale at which the numbers should be interpreted.

## What the files are and how to read the outputs

The stage 1 outputs are NetCDF files with hourly currents at one depth on a small curvilinear grid around each reef. They are named using the reef’s name and label and include attributes that document the grid indices and depth. The stage 2 outputs are weekly particle files and a yearly CSV per reef. The CSV contains the start time of each weekly release, the number of particles used, the daily fractions remaining inside the reef and the threshold based residency time in days. These tables are designed to be easy to summarise by month or to average across years to form a climatology. The animations give a quick visual check that the setup is behaving as expected.

## Why we subset and save locally

The raw eReefs dataset is very large. Streaming it over the network during every trial would be slow and brittle. By extracting a small grid aligned window once per reef and year, then saving it to a local NetCDF file, we reduce both data volume and complexity. Local files allow you to run the second stage repeatedly while adjusting parameters, and to compare results across different settings without waiting for downloads. This approach also improves reproducibility because the exact input fields used for a given set of runs are preserved on disk.

## Practical checks before you trust the numbers

A little time spent on basic validation pays off. After extraction, open a subcube and confirm that the time axis covers the target year with hourly spacing, the depth attribute is close to −2.5 m and the grid window looks sensible when plotted. Before running many weeks, do a single preview release and watch the animation. Check that the reef outline aligns with the grid and that particles are released over water rather than over land. Inspect the daily fraction for the first month and confirm it decays smoothly rather than jumping in odd ways. Keep a record of the random seed so repeated runs are comparable.

## Extending and adapting the approach

Once you are comfortable with the basics you can adapt the method. You might test two or three diffusion values to assess sensitivity, try a slightly deeper level if your sampling is subsurface, or change the threshold from 10 percent to a value that reflects your detection limit. You can form seasonal medians or compare residency during a known climate event. With more effort you could include vertical motion or add a simple exponential decay to mimic eDNA degradation, but start with the clear and defensible baseline described here.

## Summary

Residency time is a practical way to quantify flushing at the scale of individual reefs. By extracting compact local current fields from eReefs and using OceanParcels to simulate many passive particles, we can construct weekly survival curves and robust threshold based metrics. The method is designed to be transparent and fast enough to iterate. Its limitations come from the 1 km grid, the single depth and the simplicity of the diffusion, so results should be interpreted at the kilometre scale. With those caveats in mind the approach provides a consistent, comparable measure of how quickly water - and by extension eDNA - leaves each reef, and how that changes through the seasons and across years.
