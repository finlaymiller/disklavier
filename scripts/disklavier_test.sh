#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "inputs/test" --param_file "params/disklavier.yaml" --output_dir "outputs" --log_config "logging.conf" -f --tempo $1