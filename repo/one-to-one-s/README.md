# Beware!

The sources under this directory (`one-to-one-s`) contain modified implementation of cross-correlation.
Certain kernels were modified for the purposes of benchmarking and these changes should be disregarded.

Please, navigate to the repository under `repo/cross-corr` directory for the original kernels.

### Code Changes

The code change introduced in this source directory is in the implementation of `shuffle_multirow_both_impl` kernel. It is hardcoded to spawn an artificially big grid to utilize all GPU resources to get reasonable benchmarking results.

The changed files are:
- `naive_shuffle_multirow_both.cu`
- `one_to_one.hpp`
- `types.cuh`