from data_downloader import DataDownloader

import os

downloader = DataDownloader(download_path="data/in-3p")
# --------------------------------------------------------
#Lawrey, E. P., Stewart M. (2016) Complete Great Barrier Reef (GBR) Reef and Island Feature boundaries including Torres Strait (NESP TWQ 3.13, AIMS, TSRA, GBRMPA) [Dataset]. Australian Institute of Marine Science (AIMS), Torres Strait Regional Authority (TSRA), Great Barrier Reef Marine Park Authority [producer]. eAtlas Repository [distributor]. https://eatlas.org.au/data/uuid/d2396b2c-68d4-4f4b-aab0-52f7bc4a81f5
direct_download_url = 'https://nextcloud.eatlas.org.au/s/xQ8neGxxCbgWGSd/download/TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.zip'
downloader.download_and_unzip(direct_download_url, 'GBR_AIMS_Complete-GBR-feat_V1b')

# Replace the .prj file for the GBR shapefile because it is using an older WKT format
# that fails to load correctly in stages 3.
gbr_shapefile = os.path.join(downloader.download_path, 'GBR_AIMS_Complete-GBR-feat_V1b', 'TS_AIMS_NESP_Torres_Strait_Features_V1b_with_GBR_Features.shp')
