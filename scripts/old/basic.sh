#!/bin/bash
cd "$(dirname "$0")"/..
python main.py --data_dir "input_data" --param_file "params/basic.yaml" --log_dir "logs" --record_dir "recordings"