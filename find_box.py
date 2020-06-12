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
    default='bbox.json',
    help='output bbox'
)
parser.add_argument(
    '--padding',
    type=int,
    default=20,
    help='padding of bbox'
)
args = parser.parse_args()

# load config
with open(args.config) as f:
    config = yaml.safe_load(f)

if 'loader' in config:
    data_list = config['list']
    loader_config = config['loader']

    loader_name = loader_config.pop('name')
    data_loader = DataLoader(loader_name, **loader_config)
    if data_list is not None:
        data_loader.set_data_list(data_list)
else:
    loader_name = config.pop('name')
    data_loader = DataLoader(loader_name, **config)


def find_center(box):
    assert len(box) == 6
    return [(i+j) / 2 for (i, j) in zip(box[:3], box[3:])]


def find_size(box):
    assert len(box) == 6
    return [(j-i) for (i, j) in zip(box[:3], box[3:])]


def find_box(target, padding=20):
    assert isinstance(padding, int)
    assert len(target.shape) == 3

    indices = np.where(target)
    corner_1 = [
        int(max(min(idx) - padding, 0))
        for idx in indices
    ]
    corner_2 = [
        int(min(max(idx) + padding, margin))
        for (idx, margin) in zip(indices, target.shape)
    ]
    return corner_1 + corner_2


boxes = {}
for idx in tqdm(data_loader.data_list):
    boxes[idx] = {
        'box': find_box(
            data_loader.get_label(idx) > 0,
            padding=args.padding
        )
    }
    boxes[idx].update({
        'center': find_center(boxes[idx]['box']),
        'shape': find_size(boxes[idx]['box'])
    })

with open(args.output, 'w') as f:
    json.dump(boxes, f, indent=2)
print('has been saved to', args.output)
