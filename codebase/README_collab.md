# Instructions for setup

Please refer to https://github.com/albanie/collaborative-experts.

Make sure you have an active conda installation

The repository contains the conda environment file `environment.yml`

`conda env create -f environment.yml`

# Feature download

The feature are available to download from https://github.com/albanie/collaborative-experts.

Place them in the respective folder at this location `data/dataset`

# Augemented Supervision

For generating csv file you need a pickle file corresponding to dataset which contain a dictionary. Each video(key) refer to list of captions. Use `data/dataset/label.py` script to generate csv.

**Note: Change the dataset file name, alpha_v(verb overlap) and alph_n(noun overlap) before running the script. More details about the alpha_v and alpha_n can be found in the supplementary doc.**

# Text feature generation

Generate the text features for the augmented supervision using `data/dataset/data_creation_utility.py` script.

**Note: Change the folder name,  csv file name and caption count before running the script. More details about the caption count can be found in the supplementary doc.**

Training and testing the model

Follow the instructions as in https://github.com/albanie/collaborative-experts.

**Make sure to set the margin hyperparameters and loss function in the appropriate config files correctly before running the experiments**
