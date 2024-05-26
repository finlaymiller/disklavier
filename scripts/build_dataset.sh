#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py --data_dir "data/datasets/trimmed midi" --output_dir "data/datasets/all" -t -s 12