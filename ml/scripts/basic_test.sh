#!/bin/bash
cd "$(dirname "$0")"/../..
python train.py --data_dir "inputs/no tempo" --param_file "ml/params/basic.yaml" --model "autoencoder" --output_dir "ml/outputs"