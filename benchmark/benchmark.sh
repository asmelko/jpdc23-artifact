#!/bin/bash

./benchmarking.py -e "$1" benchmark -c -o ./out ./benchmark.yml $2

./preprocess.py