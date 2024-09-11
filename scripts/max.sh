#!/bin/bash
DATASET_NAME="test"
cd "$(dirname "$0")"/..
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/synthetic"\
    --params "params/max.yaml"\
    --output "data/outputs"\
    --tables "data/tables/$DATASET_NAME"\
    --bpm $1