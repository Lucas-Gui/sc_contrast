
# SC - Contrast

## Description

This repository contains the code associated with my master thesis for EPFL, accomplished at the  [Biomolecular Control Group](https://homepages.inf.ed.ac.uk/doyarzun/) at the University of Edinburgh.

It allows the training and evaluation of representation learning models for single-cell RNA seq data.
These models can be incorporated in a Multiple Instance Learning framework.

## Requirements

We do not provide a requirements file because some packages are only available in pip (not conda), and pytorch should be installed separately depending on your system requirements.

This code needs the following packages :

```
python=3.11
pytorch=2.1
seaborn
pandas
openpyxl 
scikit-learn
tqdm
tensorboard
jupyter
configargparse
sp1md # for Hotelling's t test 
```

Package versions are supplied as used, but later versions might work as well.
