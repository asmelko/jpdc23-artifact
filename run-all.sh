#!/bin/bash

set -x

# build 
cd repo
./build.sh cross-corr
./build.sh one-to-one-s
cd -

# benchmark 
cd benchmark
./benchmark.sh ../repo/cross-corr/build/cross one_to_one
./benchmark.sh ../repo/one-to-one-s/build/cross one_to_one_saturated

./benchmark.sh ../repo/cross-corr/build/cross one_to_many
./benchmark.sh ../repo/cross-corr/build/cross one_to_many_saturated

./benchmark.sh ../repo/cross-corr/build/cross n_to_mn
./benchmark.sh ../repo/cross-corr/build/cross n_to_mn_saturated

./benchmark.sh ../repo/cross-corr/build/cross n_to_m
./benchmark.sh ../repo/cross-corr/build/cross n_to_m_saturated
cd -

# plot 
cd plots
rm -rf one-to-one one-to-many n-to-m n-to-mn
mkdir one-to-one one-to-many n-to-m n-to-mn
Rscript plot_one_to_one.R
Rscript plot_one_to_many.R
Rscript plot_n_to_mn.R
Rscript plot_n_to_m.R
cd -