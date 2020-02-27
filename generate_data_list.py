#!/usr/bin/env python3

import argparse
import random
from MIDP import DataLoader
import yaml


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--loader-config',
    required=True,
    help='specify the data parsers in the data loader'
)
arg_parser.add_argument(
    '--output',
    default='./data_list.yaml',
    help='output data list'
)
arg_parser.add_argument(
    '--split-ratio',
    default=0.7,
    type=float,
    nargs='+',
    help='data split ratio of train/valid/test'
)
args = arg_parser.parse_args()

# load config
with open(args.loader_config) as f:
    loader_config = yaml.safe_load(f)
raw_loader_config = loader_config.copy()
loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)
n_data = len(data_loader.data_list)
assert n_data > 0

# random sample
data_list = random.sample(data_loader.data_list, n_data)

# split ratio
if type(args.split_ratio) is list:
    split_ratio = args.split_ratio
else:
    split_ratio = [args.split_ratio]

# partition data list
if len(split_ratio) == 1:
    n_train = int(n_data * split_ratio[0])
    n_valid = n_data - n_train
    n_test = 0
elif len(split_ratio) == 2:
    n_train = int(n_data * split_ratio[0])
    n_valid = int(n_data * split_ratio[1])
    n_test = n_data - n_train - n_valid
else:
    raise ValueError('split ratio should be of length 1 or 2.')

# setup data list dict
data_list_dict = {
    'loader': raw_loader_config,
    'amount': {
        'total': n_data,
        'train': n_train,
        'valid': n_valid,
        'test': n_test
    },
    'list': {
        'train': sorted(data_list[:n_train]),
        'valid': sorted(data_list[n_train:n_train+n_valid]),
        'test': sorted(data_list[n_train+n_valid:n_train+n_valid+n_test]),
    }
}

with open(args.output, 'w') as f:
    yaml.dump(data_list_dict, f)
print('Saved data list into ', args.output)
