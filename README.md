# Artifact Submission: cross-corr

This is a replication package containing code and experimental results related to a JPDC paper titled: Efficient GPU-accelerated Parallel Cross-correlation.

## Overview

The artifact comprises the following directories:

* `benchmark` -- benchmarking scripts
* `plots` -- scripts for generating results plots	
* `repo` -- the CUDA implementation of cross-correlation
  - `cross-corr` -- main codebase containing all the kernel source files
  - `one-to-one-s` -- a copy of the codebase with modified one-to-one grouped-overlap kernels to fully saturate GPU for specific micro-benchmarks
* `results` -- containing plots (including those that were not published in the paper due to the page limit) and CSV data files with measurements (that were used to generate the plots)


## Detailed artifact contents

`repo/cross-corr/src` directory contains the source files to the CUDA kernels. The names of kernels in this directory are in the original form. In the paper, we decided to rename them for the sake of better readability and simpler understanding. Here we provide the translation table for the kernel names:

| Optimization name in the paper | Kernel name in the source files |
| --------------------------- | ----------- |
| overlap-wise |  `cross_corr_naive_original` |
| warp-per-overlap | `ccn_warp_per_shift`|
| split-row | `ccn_shuffle_work_distribution` |
| grouped-overlap | `ccn_shuffle_multirow_both` |
| multi-matrix-right | `ccn_shuffle_multimat_right` |
| multi-matrix-both | `ccn_shuffle_multimat_both` |

Note, that we omitted the combinations of the aforementioned optimizations, which do have their separate kernel names. Also, source files include kernels that were not described in the paper; this is because they did not provide any advantage over the already comprehensive list of the presented optimizations. To see the full list, see the source [README](repo/cross-corr/src/README.md).

The compilation of the cross-correlation binary is governed by `cross-corr/CMakeLists.txt` CMake file. The compilation time can vary according to the *defined parameters* passed to CMake during the configuration (also documented in the source [README](repo/cross-corr/src/README.md)). The CMake defines parameters that specify the maximum value of an argument of a specific kernel (e.g., the maximum number of *right matrices per thread* for the *multi-matrix-right* algorithm). As a result, these parameters affect how many different specializations of the same kernel will be built and therefore how long the compilation will take. **The full build required for performing all experiments takes around 6 hours** on our hardware.


## Requirements for running the experiments

Hardware requirements:

* A CUDA-compatible GPU

Software requirements:

* [CUDA toolkit 12.2 or later](https://developer.nvidia.com/cuda-downloads) and appropriate driver
* `cmake 3.18` or later 
* `boost` and `nlohmann` libraries
* `python` for benchmarking
* `R` software for plotting the graphs (see details below)

Let us present a few commands for your convenience that will allow you to set up the environment quickly:

Installing all dependencies (except CUDA and GPU driver) on Debian/Ubuntu:
```
sudo apt-get update && apt-get install -y g++ cmake r-base python3 python3-pip libboost-all-dev nlohmann-json3-dev
```

Installing all dependencies (except CUDA and GPU driver) on RHEL-like distribution:
```
sudo dnf install -y cmake gcc-c++ R python3 python3-pip boost-devel json-devel
```

Installing the Python packages required by our benchmarking suite:
```
sudo pip3 install numpy pandas scipy ruamel.yaml
```

R packages necessary for generating the plots:
```
sudo R -e "install.packages(c('ggplot2', 'cowplot', 'sitools', 'viridis', 'dplyr'), repos='https://cloud.r-project.org')"
```


> Note: If the version of cmake is greater than `3.24`, cmake is able to discover the native CUDA Compute Capability (CC), under which the kernels will be compiled. If the cmake version is smaller, the CUDA compilation is defaulted to generate CC 5.0 PTX code. In order to provide the best results, it is advised to change this attribute manually according to the hardware at hand. To do so, modify line 119 in `repo/cross-corr/CMakeLists.txt` and `repo/one-to-one-s/CMakeLists.txt`.


## Running the experiments

Our experiments are designed to provide a comprehensive analysis of the aforementioned algorithms running various combinations of parameters computing different sizes of input instances. Therefore, the overall duration of **running the experiments is quite long** (currently about **2 to 3 days** on our GPU cluster).

In order to provide a swift way to check the reproducibility of our experiments, we prepared a special script that runs only a subset of benchmarks.

**Kick the tires:**

Just to see whether the code is working, run the following from the root directory:
```
./kick-the-tires.sh
```
The script should take about 10 minutes to finish. First, it builds the binary passing a specific combination of macro parameters to ensure that only kernels for one-to-one inputs are built. The script then runs a subset of one-to-one experiments. 

After the script runs, it will generate results in a CSV format in the `results` directory. It should contain 5 CSV files, one file for each measured algorithm. Each CSV file contains self-documenting headers. Finally, the plotting script is executed generating a single plot in the `plots/one-to-one-fast` directory. The `benchmark/benchmark.yml` file specifies which combination of algorithms, parameters, and input sizes is benchmarked (note section `one_to_one_fast` starting on line 482).
More details on how the CSV results rows are processed into plots can be found in the `plots/plot_fast.R` script.

The generated plot file will be named `one-to-one.pdf` and it shows the comparison of the best one-to-one cross-correlation implementations with the baseline (naive `overlap-wise` approach) and FFT implementation.


**Complete set of measurements:**

To run the complete benchmark, execute
```
./run-all.sh
```

## Measured results

The results measured by `./run-all.sh` on our GPU cluster were stored in the `results` directory. The `results/data` directory is divided into three subdirectories, each containing results from different GPU architectures:
- `ada` -- NVIDIA Tesla L40 PCIe 48 GB
- `ampere` -- NVIDIA Tesla A100 PCIe 80 GB
- `volta` -- NVIDIA Tesla V100 SXM2 32 GB

The `results/plots` directory contains only figures generated for the `ampere` architecture. To generate the plots of other architectures, the plot scripts (in `plots` directory). Here we describe each figure in the `results/plots` directory:

| Plot | Description | Figure number in paper |
| --------------------------- | ----------- | -- |
| `one-to-one/grouped-overlap.pdf`|one-to-one grouped-overlap optimization | Figure 15
| `one-to-one/grouped-overlap-*.pdf`|one-to-one grouped-overlap optimization with different combinations of cached left rows, cached shifts and block size | not included 
| `one-to-one/split-row.pdf`|one-to-one split-row optimization | Figure 16 
| `one-to-one/one-to-one.pdf`|comparison of all one-to-one algorithms and optimizations | Figure 17
| `one-to-one/warp-per-overlap.pdf`|one-to-one warp-per-overlap optimization |  not included
| `one-to-one/warp-per-overlap-shared-memory.pdf`|one-to-one warp-per-overlap with shared memory optimization |   not included
| `one-to-one/one-to-one-warp-per-overlap-and-split-row.pdf`|comparison of one-to-one warp-per-overlap optimizations and split-row optimization |  not included
| `one-to-many/multimat-right-grouped-overlap.pdf`|one-to-many multi-matrix-right grouped-overlap optimization | Figure 18
| `one-to-many/multimat-right-grouped-overlap-*.pdf`|one-to-many multi-matrix-right grouped-overlap optimization with different combinations of cached left rows, cached shifts, right matrices per thread and inputs |  not included
| `one-to-many/multimat-right-split-row.pdf`|one-to-many multi-matrix-right split-row optimization | Figure 19
| `one-to-many/multimat-right-split-row-*.pdf`|one-to-many multi-matrix-right split-row optimization with different combinations of overlap rows, right matrices per thread and inputs |  not included
| `one-to-many/multimat-right.pdf`|one-to-many multi-matrix-right optimization |  not included
| `one-to-many/one-to-many.pdf`|comparison of all one-to-many algorithms and optimizations | Figure 20
| `n-to-mn/multimat-right-grouped-overlap.pdf`|n-to-mn multi-matrix-right grouped-overlap optimization |  not included
| `n-to-mn/multimat-right-grouped-overlap-*.pdf`|n-to-mn multi-matrix-right grouped-overlap optimization with different combinations of cached left rows, cached shifts, right matrices per thread, streams and inputs |  not included
| `n-to-mn/multimat-right-split-row.pdf`|n-to-mn multi-matrix-right split-row optimization |  not included
| `n-to-mn/multimat-right-split-row-*.pdf`|n-to-mn multi-matrix-right split-row optimization with different combinations of overlap rows, right matrices per thread and inputs |  not included
| `n-to-mn/multimat-right.pdf`|n-to-mn multi-matrix-right optimization |  not included
| `n-to-mn/n-to-mn.pdf`|comparison of all n-to-mn algorithms and optimizations |  not included
| `n-to-mn/n-to-mn-one-to-many-streams.pdf`|comparison of grouped-overlap and split-row optimizations on n-to-mn input running serial one-to-many kernels and parallel n-to-mn kernels using streams |  not included
| `n-to-m/multimat-both-grouped-overlap.pdf`|n-to-m multi-matrix-both grouped-overlap optimization | Figure 21
| `n-to-m/multimat-both-grouped-overlap-*.pdf`|n-to-m multi-matrix-both grouped-overlap optimization with different combinations of cached left rows, cached shifts, right matrices, left matrices per thread |  not included
| `n-to-m/multimat-both-split-row.pdf`|n-to-m multi-matrix-both split-row optimization | Figure 22 
| `n-to-m/multimat-both-split-row-*.pdf`|n-to-m multi-matrix-both split-row optimization with different combinations of overlap rows, left matrices, right matrices per thread and inputs |  not included
| `n-to-m/multimat-both.pdf`|n-to-m multi-matrix-both optimization |  not included
| `n-to-m/n-to-m.pdf`|comparison of all n-to-m algorithms and optimizations | Figure 23
