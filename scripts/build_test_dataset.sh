#!/bin/bash
cd "$(dirname "$0")"/..
python build_dataset.py --data_dir "data/datasets/test datasets" --output_dir "data/datasets/test" -t -s