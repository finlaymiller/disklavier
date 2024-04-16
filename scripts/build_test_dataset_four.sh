#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py --data_dir "inputs/test datasets" --output_dir "inputs/test" -t -s -n 16