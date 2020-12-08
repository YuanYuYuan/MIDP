#!/usr/bin/env python3

from MIDP import DataGenerator, DataLoader, save_nifti
from tqdm import tqdm
import numpy as np
import argparse
import os
import yaml
import time
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    '--generator-config',
    required=True,
    help='generator config'
)
parser.add_argument(
    '--loader-config',
    required=True,
    help='loader config'
)
parser.add_argument(
    '--output-dir',
    default=None,
    help='directory to store ouptut images'
)
args = parser.parse_args()

timer = time.time()
with open(args.loader_config) as f:
    loader_config = yaml.safe_load(f)
loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)

with open(args.generator_config) as f:
    generator_config = yaml.safe_load(f)
data_generator = DataGenerator(data_loader, generator_config)


if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)

results = []
for i in range(10):
    for data in tqdm(
        data_generator,
        total=len(data_generator),
        dynamic_ncols=True,
        desc='[Generating batch data]'
    ):
        # batch = data['label'][:, 48, 48, 48].flatten()
        batch = data['label'].flatten()
        (uniq, counts) = torch.unique(batch, return_counts=True)
        freq = np.zeros(16)
        for (u, c) in zip(uniq, counts):
            freq[u] = c
        # print(np.sum(freq[1:])/np.sum(freq))
        freq /= np.sum(freq)
        results.append(freq)

data = np.array(results)
np.savetxt('data.csv', data, delimiter=',')

print('Total time:', time.time()-timer)
