./build.sh one-to-one
./benchmark.sh one-to-one/build/cross one_to_one_fast

cd plots
rm -rf one-to-one-fast
mkdir one-to-one-fast
Rscript plot_fast.R
cd -