#!/bin/bash
cd "$(dirname "$0")"/../..
python train_simple.py --data_dir "data/outputs" --param_file "ml/params/vanilla.yaml" --output_dir "ml/outputs"