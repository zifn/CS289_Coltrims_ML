# CS289A - COLTRIMS ML
Project Title: Classification of Molecular States in COLTRIMS Scattering Experiments
Group: Richard Thurston, Larry Chen, Sajant Anand
Assignment: Project Family F, CS289A, UC Berkeley Fall 2020

## Summary

We implement analysis of COLTRIMS scattering data to cluster measurments into molecular states. We determine the fitness of the clustering by fitting each cluster to an angular distribution and calculating the cross entry between clustering label and predicted fitting label. The data analyzed in the final report was graciously provided by the [Atomic, Molecular, and Optical Sciences group](http://amo-csd.lbl.gov/home.php) at LBNL and has not yet been published. Thus it cannot be shared.
Instead we provide a randomly generated dataset 'RANDOM_DATASET.dat' so that our code can be tested.

Analysis file is 'analysis.py' with source files found in 'src/' and testing files found in 'src/tests/'.

## Dependencies

To install all required dependencies, run

```
python -m pip install -r requirements.txt
```

- matplotlib>=3.1.1
- numpy>=1.16.5
- scipy>=1.3.1
- pandas>=0.25.1
- scikit-learn>=0.23.2

## Quick Start

To run the analysis code:

```
python analysis.py

usage: analysis.py [-h] [-c CONFIG] [--cinit CLUSTERS_INIT]
                   [--cmin CLUSTERS_MIN] [--cmax CLUSTERS_MAX]
                   [--cstep CLUSTERS_STEP] [--bmin BINS_MIN] [--bmax BINS_MAX]
                   [--bstep BINS_STEP] [-L L]
                   datafile

Analyze a COLTRIMS dataset.

positional arguments:
  datafile              Path to the COLTRIMS datafile.

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Path to configuration file.
  --cinit CLUSTERS_INIT
                        The initial number of clusters to use.
  --cmin CLUSTERS_MIN   The minimum cluster size.
  --cmax CLUSTERS_MAX   The maximum cluster size.
  --cstep CLUSTERS_STEP
                        The step size for the cluster grid search.
  --bmin BINS_MIN       The minimum bin size.
  --bmax BINS_MAX       The maximum bin size.
  --bstep BINS_STEP     The step size for the bin size grid search
  -L L                  The largest Lmax to try.
```

To analysis the random data set with default settings:
```
user@machine$ python analysis.py RANDOM_DATASET.dat
```

## Pylint and Pytest
Tests of the source functions can be run from the root project directory by: 

```
user@machine$ pytest
```

We conform to Pylint specifications for the source and testing files.
