#!/usr/bin/env python3
from MIDP import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
import yaml
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    required=True,
    help='config'
)
parser.add_argument(
    '--output',
    default='box.json',
    help='output bbox'
)
args = parser.parse_args()

# load config
with open(args.config) as f:
    config = yaml.safe_load(f)
data_list = config['list']
loader_config = config['loader']

loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)
if data_list is not None:
    data_loader.set_data_list(data_list)

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
box['center'] = (box['corner2'] + box['corner1']) / 2
box['size'] = box['corner2'] - box['corner1']
for key in box:
    box[key] = box[key].tolist()
print('Bounding box:', box)
with open(args.output, 'w') as f:
    json.dump(box, f, indent=2)
print('has been saved to', args.output)
