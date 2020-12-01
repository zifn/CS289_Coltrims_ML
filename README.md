# CS289 - COLTRIMS ML

Group: Richard Thurston, Larry Chen, Sajant Anand

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

To run the analysis code,

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

