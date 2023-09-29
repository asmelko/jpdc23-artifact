#!/bin/bash

./benchmark/benchmarking.py -e "$1" benchmark -c -o ./out ./benchmark/benchmark.yml $2

cd benchmark
./preprocess.py
cd -