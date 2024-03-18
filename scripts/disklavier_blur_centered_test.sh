#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/test" --param_file "params/disklavier_blur_centered.yaml" --output_dir "outputs" --tempo $1