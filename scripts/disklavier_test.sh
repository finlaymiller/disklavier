#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/test" --param_file "params/disklavier.yaml" --output_dir "outputs" -f --tempo $1