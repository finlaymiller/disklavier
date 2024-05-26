#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --data_dir "data/datasets/careful"\
    --param_file "params/max.yaml"\
    --output_dir "data/outputs"\
    --kickstart "data/datasets/careful/20231220-80-01_0000-0008.mid"\
    --tempo $1