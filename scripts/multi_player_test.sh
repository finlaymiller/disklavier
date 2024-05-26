#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --data_dir "data/datasets/test"\
    --param_file "params/multi_player.yaml"\
    --output_dir "data/outputs"\
    --tempo $1