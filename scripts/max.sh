#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/all-fourbar-tmp" --param_file "params/max.yaml" --output_dir "outputs" --tempo 80 -k