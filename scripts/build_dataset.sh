#!/bin/bash
DATASET="test_micro"

cd "$(dirname "$0")"/..
python build_dataset.py\
    --data_dir "data/datasets/$DATASET"\
    --dataset_name "$DATASET"\
    -art

python build_tables.py\
    --data_dir "data/datasets/$DATASET"\
    --dataset_name "$DATASET"\
    --metric "pitch_histogram"\
    -nrst
