#!/bin/bash
cd "$(dirname "$0")"/../..

python train.py --data_dir "data/all_data_prs.npz" --param_file "ml/params/bad_overfit.yaml" --output_dir "ml/outputs" --device "cuda:0"
# python train.py --data_dir "data/all_data_prs.npz" --param_file "ml/params/overfit_base_10.yaml" --output_dir "ml/outputs" --device "cuda:0"
# python train.py --data_dir "data/all_data_prs.npz" --param_file "ml/params/overfit_base_100.yaml" --output_dir "ml/outputs" --device "cuda:0"