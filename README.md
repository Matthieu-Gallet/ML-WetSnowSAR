# ML-WetSnowSAR

This repository contains the code for the evaluation of the WetSnowSAR dataset, using machine learning algorithms.

## 1. Structure

The repository is structured as follows:
```bash
.
├── README.md
├── LICENSE
├── data
│   ├── data.tar.gz.partaa
│   ...
│   └── data.tar.gz.partak
├── estimators
│   ├── __init__.py
│   ├── covariance.py
│   ├── statistical_descriptor.py
│   └── textural_features.py
├── evaluation
│   ├── __init__.py
│   ├── PE3_learning_models.py
│   ├── PE4_validate_models.py
│   └── main_comparisons.py
├── parameters
│   ├── comparison_algos.yml
│   ├── comparison_channels.yml
│   └── comparison_stats.yml
├── utils
│   ├── __init__.py
│   ├── dataset_management.py
│   ├── image_processing.py
│   ├── metrics.py
│   ├── files_management.py
│   └── geo_tools.py
└── plot
    ├── __init__.py
    ├── figure_main.py
    └── ... (other figures scripts)
```


## 2. Data
We provide the data in the form of a tarball. The data is split into 5 parts, each of which is 1GB in size. To reconstruct the tarball, run the following command:
```bash
cat data.tar.gz.part* | tar -xvz
```
The full dataset is also available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3906370.svg)](https://zenodo.org/record/8111485)