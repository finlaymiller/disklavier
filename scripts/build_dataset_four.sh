#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py --data_dir "data/datasets/trimmed midi" --output_dir "data/datasets/all-fourbar" -t -s 12 -n 16