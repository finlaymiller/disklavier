#!/bin/bash
cd "$(dirname "$0")"/..
python main.py\
    --data_dir "data/datasets/careful"\
    --param_file "params/disklavier.yaml"\
    --output_dir "data/outputs"\
    --kickstart "data/datasets/careful/20240117-64-01_0160-0168.mid"\
    -p \
    --tempo $1
    # --kickstart "data/datasets/careful/20240213-100-02_1944-1952.mid"\