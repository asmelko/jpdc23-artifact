# Artifact Submission: cross-corr

This is a repliaction package containing code and experimental results related to a JPDC paper titled:  TBD

## Overview

The artifact comprises the following directories:

* `benchmark` -- Benchmarking scripts.
* `plots` -- Plotting scripts.
* `repo` -- CUDA implementation of cross-correlation. It contains 2 subdirectories --- the main `cross-corr` repository containing all the kernel source files and its copy `one-to-one-s` with modified one-to-one grouped-overlap kernels to fully saturate GPU for specific micro-benchmarks.

## Detailed artifact contents

`repo/cross-corr/src` directory contains the source files to the CUDA kernels. The names of kernels in this directory are in the original form. In the paper, we decided to rename them for the sake of better readability and simpler understanding. Here we provide the translation table for the kernel names:

| Optimization name in the paper | Kernel name in the source files |
| --------------------------- | ----------- |
| overlap-wise |  `cross_corr_naive_original` |
| warp-per-shift | `ccn_warp_per_shift`|
| split-row | `ccn_shuffle_work_distribution` |
| grouped-overlap | `ccn_shuffle_multirow_both` |
| multi-matrix-right | `ccn_shuffle_multimat_right` |
| multi-matrix-both | `ccn_shuffle_multimat_both` |

Note, that we omited the combinations of the aforementioned optimizations, which do have their separate kernel names. Also, source files include kernels that were not described in the paper; this is because they did not provide any advantage over the already comprehensive list of the presented optimizations. To see the full list, see the source [README](repo/cross-corr/src/README.md).

The compilation of the cross-correlation binary is governed by `cross-corr/CMakeLists.txt` cmake file. The compilation time can vary according to the *define parameters* passed to cmake during the configuration (also documented in the source [README](repo/cross-corr/src/README.md)). The cmake define parameters specify the maximum value of an argument of a specific kernel (e.g., the maximum number of *right matrices per thread* for multi-matrix-right algorithm). As a result, these parameters affect how many different specializations of the same kernel will be built and therefore how long the compilation will take. The full build required for performing all experiments takes around 6 hours.

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

Our experiments are designed to provide comprehensive analysis of the aforementioned algorithms running with various combinations of parameters computing different sizes of input instances. Therefore, the overall duration of running the experiments is quite long --- it is in a range of 2 to 3 days.

In order to provide a swift way to check the reproducibility of our experiments, we prepared a special script which runs only a subset of benchmarks.

**Kick the tires:**

Just to see whether the code is working, run this from the root directory:
```
./kick-the-tires.sh
```
This should take ~10m to finish. The script first builds the binary passing a specific combination of define parameters such that only kernels for one-to-one inputs are build. The script then runs a subset of one-to-one experiments. 

After the script runs, it will generate results in csv format in `results` directory. It should contain 5 files, one file for each measured algorithm. Each csv file contains self-documenting headers. Finally, the plotting script is run generating two plots in `plots/one-to-one-fast` directory. To see which exact combination of algorithm, parameters and inputs are run, open `benchmark/benchmark.yml` and go to benchmark group called `one_to_one_fast` starting line 477. To see the way how the results csv rows are processed into plots, see `plots/plot_fast.R`.

The two generated plots will be `one-to-one.pdf`, which shows the comparison of the best one-to-one cross-corr implementations with the naive `overlap-wise` approach, and `one-to-one-fft.pdf`, which compares the same implementations with fft.

**Full measurement:**

To run the full measurements, run:
```
./run-all.sh
```
The script performs the same steps as in kick-the-tires; the only difference is that it runs the full set of experiments in the overall duration of 2-3 days.