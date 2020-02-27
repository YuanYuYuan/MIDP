#!/usr/bin/env python3

from glob import glob
import os
import yaml
import numpy as np
import nrrd
from scipy import ndimage
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default='./configs/nrrd2nifti.yaml',
)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)


data_list = [
    fn.split('/')[-2] for fn in
    glob(os.path.join(config['data_dir'], '*/'))
]

os.makedirs(config['output_dir'], exist_ok=True)
for key in ['images', 'labels']:
    os.makedirs(os.path.join(config['output_dir'], key), exist_ok=True)


def get_spacing(nrrd_data):
    return tuple(np.diag(nrrd_data[1]['space directions']))


def load_image(data_idx):
    nrrd_data = nrrd.read(os.path.join(
        config['data_dir'],
        data_idx,
        'img.nrrd'
    ))
    image = nrrd_data[0]
    spacing = get_spacing(nrrd_data)
    return image.astype(np.int16), spacing


def load_label(data_idx):
    label: np.Array = None
    spacing = None

    for roi, value in config['roi_map'].items():
        nrrd_data = nrrd.read(os.path.join(
            config['data_dir'],
            data_idx,
            'structures',
            roi + '.nrrd'
        ))
        if label is None:
            label = nrrd_data[0]
            spacing = get_spacing(nrrd_data)
        else:
            label = np.maximum(label, nrrd_data[0] * value)
            assert spacing == get_spacing(nrrd_data)

    return label.astype(np.int16), spacing


def convert(data_idx, order=2):
    image, spacing_1 = load_image(data_idx)
    label, spacing_2 = load_label(data_idx)
    assert spacing_1 == spacing_2
    assert image.shape == label.shape
    zoom = tuple(s / config['spacing'] for s in spacing_1)

    image = ndimage.zoom(image, zoom, order=order, mode='nearest')
    image = image.astype(np.int16)
    nib.save(
        nib.Nifti1Image(image, affine=np.eye(4)),
        os.path.join(config['output_dir'], 'images', data_idx + '.nii.gz')
    )

    label = ndimage.zoom(label, zoom, order=0, mode='nearest')
    label = label.astype(np.int16)
    nib.save(
        nib.Nifti1Image(label, affine=np.eye(4)),
        os.path.join(config['output_dir'], 'labels', data_idx + '.nii.gz')
    )


with Pool(os.cpu_count()) as pool:
    list(tqdm(pool.imap(convert, data_list), total=len(data_list)))


with open(os.path.join(config['output_dir'], 'info.json'), 'w') as f:
    json.dump({'list': data_list, 'roi_map': config['roi_map']}, f, indent=4)

print('Ouputs have been stored in %s.' % config['output_dir'])
