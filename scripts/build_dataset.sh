#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py --data_dir "inputs/trimmed midi" --output_dir "inputs/all" -t -s 12