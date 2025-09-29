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

## Running the scripts

### 01-download-reef-data.py
This downloads the reef boundary dataset used in the subsequent processing.
```bash
python 01-download-reef-data.py
```

### 02-extract-gbr1-subcubes.py
This extracts the eReefs GBR1 hourly hydrodynmaic data from NCI and saves the result as local NetCDF files. These files
are used by the particle movement simulation.

![Map showing how to get the LABEL_ID](media/reef-label-id.png)
To find out the LABEL_ID of the reefs you are interested in you can use the following [interactive map of GBR Features dataset](https://maps.eatlas.org.au/index.html?intro=false&z=9&ll=147.88826,-18.90488&l0=ea_nesp1%3ATS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features,ea_ea-be%3AWorld_Bright-Earth-e-Atlas-basemap,google_HYBRID,google_TERRAIN,google_SATELLITE,google_ROADMAP&v0=,,f,f,f,f).

The following is a minimal example downloading the 2016 currents for Davies Reef at -2.35 m with a 10 km buffer.
```bash
python 03-extract-gbr1-subcubes.py --config debug.toml
python 04-basic-particle-flow.py --config debug.toml
python 05-basic-particle-flow.py --config debug.toml 
```


