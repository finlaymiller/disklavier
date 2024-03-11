#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "input_data" --param_file "params/basic.yaml" --model "autoencoder"