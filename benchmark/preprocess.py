#!/usr/bin/env python3

import os
import shutil

from shared import Benchmark

from pathlib import Path

benchmark = Benchmark.load(Path.cwd() / "out")
results_path = "../results"

if os.path.exists(results_path):
    shutil.rmtree(results_path)
os.mkdir(results_path)

for group_name, group in benchmark.groups.items():
    os.mkdir(f"{results_path}/{group_name}")
    headers = set()
    for run in group.runs:
        header = run.name not in headers
        headers.add(run.name)
        run.data.to_csv(f"{results_path}/{group_name}/{run.name}.csv", mode="a+", header=header)
