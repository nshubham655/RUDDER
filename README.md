[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rudder-a-cross-lingual-video-and-text/video-retrieval-on-charades-sta)](https://paperswithcode.com/sota/video-retrieval-on-charades-sta?p=rudder-a-cross-lingual-video-and-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rudder-a-cross-lingual-video-and-text/video-retrieval-on-rudder)](https://paperswithcode.com/sota/video-retrieval-on-rudder?p=rudder-a-cross-lingual-video-and-text)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/rudder-a-cross-lingual-video-and-text/video-retrieval-on-didemo)](https://paperswithcode.com/sota/video-retrieval-on-didemo?p=rudder-a-cross-lingual-video-and-text)
# Instructions for setup

Please refer to https://github.com/albanie/collaborative-experts.

Make sure you have an active conda installation

The repository contains the conda environment file `environment.yml`

`conda env create -f environment.yml`

# Feature download

The feature are available to download from https://github.com/albanie/collaborative-experts.

Place them in the respective folder at this location `data/dataset`

# Text feature generation

Generate the text features for the augmented supervision using `data/dataset/data_creation_utility.py` script.

**Note: Change the folder name,  csv file name and caption count before running the script. More details about the caption count can be found in the supplementary doc.**

Training and testing the model

Follow the instructions as in https://github.com/albanie/collaborative-experts.

**Make sure to set the margin hyperparameters and loss function in the appropriate config files correctly before running the experiments**
