#!/usr/bin/env python3

from MIDP import DataLoader
import argparse
import yaml
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--loader-config',
    required=True,
    help='loader config'
)
parser.add_argument(
    '--output-dir',
    default='output',
    help='output dir'
)
args = parser.parse_args()

with open(args.loader_config) as f:
    loader_config = yaml.safe_load(f)

loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)

data_idx = data_loader.data_list[0]

timer = time.time()
label = data_loader.get_label(data_idx)
print('Resampling and loading data took:', time.time()-timer)

timer = time.time()
data_loader.save_prediction(
    data_idx,
    label,
    args.output_dir
)
print('Resampling and saving data took:', time.time()-timer)

print(data_idx)
print(label.shape)
