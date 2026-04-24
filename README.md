Downloaded MAPS from https://zenodo.org/records/18160555

Tentative setup guide:
1. setup virtual environment, install requirements.txt
2. download MAPS dataset and put it (the whole zipped file) in ./datasets
3. run python scripts/build_maps_manifest.py
this will load the MAPS dataset as an index. the index's structure might change, but this should give you an idea of how i plan on using the dataset


Development Notes -- temporary section to guide collaboration
f: MAESTRO has an official test/valid/train split that is presumably used by the original paper. unfortunately my preprocessing doesn't current support this and was made to have a similar pipeline to MAPS which does not provide this. we can look into fixing this later, maybe. IMO since we cant do absolute comparisons due to differences in scale anyways, it is justifiable to use a custom split as that will not influence qualitative comparisons. maybe discuss this with our TA