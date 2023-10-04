./build.sh one-to-one
./benchmark.sh one-to-one/build/cross one_to_one_fast

cd plots
mkdir one-to-one-fast
Rscript plot_fast.R
cd -