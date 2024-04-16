#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/all-fourbar" --param_file "params/disklavier_blur.yaml" --output_dir "data/outputs" --tempo $1