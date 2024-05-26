#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "data/datasets/test" --param_file "params/disklavier_blur_centered.yaml" --output_dir "data/outputs" --tempo $1