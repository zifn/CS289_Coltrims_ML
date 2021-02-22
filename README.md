
# CS289A - COLTRIMS ML

## RIMS - Group
Project Title: Classification of Molecular States in COLTRIMS Scattering Experiments  
Group Members: Richard Thurston, Larry Chen, Sajant Anand  
Assignment: Project Family F, CS289A, UC Berkeley Fall 2020

## Summary

This codebase implements an analysis of COLd Target Recoil Ion Momentum Spectroscopy (COLTRIMS) scattering data by clustering measurements into potential molecular states. We use the cross-entropy loss function as a measure of the quality of the clustering. This is computed by using the clustering label generated from the clustering algorithm, and the probabilities generated after fitting each cluster to an angular probability distribution to compute the cross-entropy of the clustered results. The data analyzed in the final report was graciously provided by the [Atomic, Molecular, and Optical Sciences group](http://amo-csd.lbl.gov/home.php) at LBNL and has not yet been published. Thus it cannot be shared.
Instead we provide a randomly generated dataset 'synthetic.dat' so that our code can be tested.

Analysis file is 'analysis.py' with source files found in 'src/' and testing files found in 'src/tests/'.

## Dependencies

To install all required dependencies, run

```
user@machine:~/project_repo$ python -m pip install -r requirements.txt
```

- matplotlib>=3.1.1
- numpy>=1.16.5
- scipy>=1.3.1
- pandas>=0.25.1
- scikit-learn>=0.23.2
- pyyaml>=5.3.0

## Quick Start

To run the analysis code:

```
user@machine:~/project_repo$ python analysis.py

usage: analysis.py [-h] [-c CONFIG] [--cinit CLUSTERS_INIT] [--cmin CLUSTERS_MIN] [--cmax CLUSTERS_MAX] [--cstep CLUSTERS_STEP] [--bmin BINS_MIN] [--bmax BINS_MAX] [--bstep BINS_STEP] [-L L]
                   [-s SAVE_DIR]
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
  -s SAVE_DIR, --save SAVE_DIR
                        Directory to save results to.
```

To analyze the generated data set with default settings:
```
user@machine:~/project_repo$ python analysis.py synthetic.dat
```

## Pylint and Pytest
Tests of the source functions can be run from the root project directory by: 

```
user@machine:~/project_repo$ pytest
```

Linter results using the included linter configuration file can be run from the root project directory by: 

```
user@machine:~/project_repo$ pylint --rcfile=.pylintrc src
```

