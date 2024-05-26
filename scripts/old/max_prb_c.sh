#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "data/datasets/all-fourbar" --param_file "params/max_prb_c.yaml" --output_dir "data/outputs" --tempo 80 -k