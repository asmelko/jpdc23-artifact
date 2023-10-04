#!/usr/bin/env python3

import os
import shutil

from shared import Benchmark

from pathlib import Path

benchmark = Benchmark.load(Path.cwd() / "out")

if os.path.exists("results"):
    shutil.rmtree("results")
os.mkdir("results")

for group_name, group in benchmark.groups.items():
    os.mkdir(f"results/{group_name}")
    headers = set()
    for run in group.runs:
        header = run.name not in headers
        headers.add(run.name)
        run.data.to_csv(f"results/{group_name}/{run.name}.csv", mode="a+", header=header)
