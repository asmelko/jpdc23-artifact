#!/bin/bash

# build one-to-one
cd repo
./build-fast.sh cross-corr
cd -

# benchmark subset of one-to-one experiments
cd benchmark
./benchmark.sh ../repo/cross-corr/build/cross one_to_one_fast
cd -

# plot subset of one-to-one experiments
cd plots
rm -rf one-to-one-fast; mkdir one-to-one-fast
Rscript plot_fast.R
cd -