#!/usr/bin/env python3

from MIDP import DataGenerator, DataLoader, save_nifti
import numpy as np
import argparse
import os
import yaml
import time

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
    default='outputs',
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


os.makedirs(args.output_dir, exist_ok=True)
for idx, data in enumerate(data_generator):
    for batch_idx, image in enumerate(data['image']):
        image = np.squeeze(image.numpy())
        file_path = os.path.join(
            args.output_dir,
            f'idx-{idx:03}-batch-{batch_idx:02}.nii.gz'
        )
        save_nifti(image, file_path)
        print(f'Outputs file to {file_path}.')

print('Total time:', time.time()-timer)
