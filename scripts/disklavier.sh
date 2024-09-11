#!/bin/bash
DATASET_NAME="20240621"
cd "$(dirname "$0")"/..
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/play"\
    --params "params/disklavier.yaml"\
    --output "data/outputs"\
    --tables "data/tables/$DATASET_NAME"\
    --bpm $1