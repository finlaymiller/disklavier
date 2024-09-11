#!/bin/bash
DATASET_NAME="test"
cd "$(dirname "$0")"/..
python src/main.py\
    --dataset "data/datasets/$DATASET_NAME/synthetic"\
    --params "params/disklavier_rewrite.yaml"\
    --output "data/outputs"\
    --tables "data/tables/$DATASET_NAME"\
	--kickstart "baba-060-02_0005-0011.mid"\
    --bpm $1