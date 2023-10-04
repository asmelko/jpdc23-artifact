# Artifact Submission: cross-corr

This is a repliaction package containing code and experimental results related to a JPDC paper titled:  TBD

## Overview

The artifact comprises the following directories:

* `benchmark` -- Benchmarking scripts.
* `plots` -- Plotting scripts.
* `repo` -- Presented CUDA implementation of cross-correlation. It contains 2 subdirectories --- the main `cross-corr` repository containing all the kernel source files and its copy `one-to-one-s` with modified one-to-one grouped-overlap kernels to fully saturate a GPU used in some micro-benchmarks.

## Detailed artifact contents

TODO

## Requirements for running the experiments

Hardware requirements:

* A CUDA-compatible GPU

Software requirements:

* [CUDA toolkit 12.2 or later](https://developer.nvidia.com/cuda-downloads) and appropriate driver
* `boost` and `nlohmann` libraries
* `python` for benchmarking
* `R` software for plotting the graphs (see details below)

Installing all dependencies on Debian/Ubuntu:
```
sudo apt-get update && apt-get install -y r-base python3 python3-pip libboost-all-dev nlohmann-json3-dev
```

Afterwards, R packages need to be installed:
```
sudo R -e "install.packages(c('ggplot2', 'cowplot', 'sitools', 'viridis', 'dplyr'), repos='https://cloud.r-project.org')"
```

Then, python packages need to be installed:
```
pip3 install numpy pandas scipy ruamel.yaml
```

## Running the experiments

**Kick the tires:**

Just to see whether the code is working, run this from the root directory:
```
./kick-the-tires.sh
```
This should take ~10m to finish. The script will run a subset of one-to-one experiments. It will generate results in csv format in `results` directory. It should contain 5 files, one file for each algorithm. Also, it will plot 2 figures in `plots/one-to-one-fast` directory --- `one-to-one.pdf`, which shows the comparison of the best one-to-one cross-corr implementations with the naive `overlap-wise` approach, and `one-to-one-fft.pdf`, which compares the same implementations with fft.

**Full measurement:**

TODO