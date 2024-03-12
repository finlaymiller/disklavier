#!/bin/bash
cd "$(dirname "$0")"/../..
python train.py --data_dir "inputs/all" --param_file "ml/params/basic.yaml" --model "autoencoder" --output_dir "ml/outputs"