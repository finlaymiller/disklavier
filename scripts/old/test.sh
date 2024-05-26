#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "data/datasets/test" --param_file "params/disklavier.yaml" --output_dir "test-outputs" --log_config "logging.conf" -f --tempo $1