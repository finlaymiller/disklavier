#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/all-fourbar" --param_file "params/max_energy.yaml" --output_dir "outputs" --tempo 80 -k