#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py --data_dir "inputs/trimmed midi" --output_dir "inputs/all-fourbar" -t -s 12 -n 16