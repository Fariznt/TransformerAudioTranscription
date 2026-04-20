Downloaded MAPS from https://zenodo.org/records/18160555

Tentative setup guide:
1. setup virtual environment, install requirements.txt
2. download MAPS dataset and put it (the whole zipped file) in ./datasets
3. run python scripts/build_maps_manifest.py
this will load the MAPS dataset as an index. the index's structure might change, but this should give you an idea of how i plan on using the dataset