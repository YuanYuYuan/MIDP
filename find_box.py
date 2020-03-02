#!/usr/bin/env python3
from MIDP import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--loader-config',
    required=True,
    help='loader config'
)
args = parser.parse_args()

with open(args.loader_config) as f:
    loader_config = yaml.safe_load(f)

loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)

box = {
    'corner1': np.ones(3) * np.Inf,
    'corner2': np.zeros(3),
}
for data_idx in tqdm(data_loader.data_list):
    indices = np.where(data_loader.get_label(data_idx) > 0)
    corner1 = np.array([min(idx) for idx in indices])
    corner2 = np.array([max(idx) for idx in indices])
    box['corner1'] = np.minimum(box['corner1'], corner1)
    box['corner2'] = np.maximum(box['corner2'], corner2)
print(box)
print(box['corner2'] - box['corner1'])
