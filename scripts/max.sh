#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/all" --param_file "params/max.yaml" --output_dir "outputs" -k --tempo 90