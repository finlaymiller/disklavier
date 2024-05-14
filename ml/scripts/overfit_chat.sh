#!/bin/bash
cd "$(dirname "$0")"/../..

python train.py --data_dir "data/all_data_prs.npz" --param_file "ml/params/overfit_chat_1.yaml" --output_dir "ml/outputs" --device "cuda:1"
# python train.py --data_dir "data/all_data_prs.npz" --param_file "ml/params/overfit_chat_10.yaml" --output_dir "ml/outputs" --device "cuda:1"
# python train.py --data_dir "data/all_data_prs.npz" --param_file "ml/params/overfit_chat_100.yaml" --output_dir "ml/outputs" --device "cuda:1"