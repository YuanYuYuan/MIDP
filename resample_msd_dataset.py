#!/usr/bin/env python3

import argparse
import nibabel as nib
import os
from glob import glob
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    '--data-dir',
    required=True,
    help='data directory'
)
parser.add_argument(
    '--spacing',
    default=1,
    type=int,
    help='resampling spacing'
)
parser.add_argument(
    '--image-dir',
    default='imagesTr',
    help='image dir'
)
parser.add_argument(
    '--label-dir',
    default='labelsTr',
    help='label dir'
)
parser.add_argument(
    '--output-dir',
    default='output',
    help='output directory'
)
args = parser.parse_args()

os.makedirs(os.path.join(args.output_dir, args.image_dir), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, args.label_dir), exist_ok=True)


def convert(data_idx):
    image_nib = nib.load(os.path.join(
        args.data_dir,
        args.image_dir,
        data_idx + '.nii.gz'
    ))
    image_zoom = np.diag(image_nib.affine)[:3]

    slices = [
        slice(-1, None, -1) if image_zoom[axis] < 0
        else slice(None)
        for axis in range(3)
    ]

    resampled_image = ndimage.zoom(
        image_nib.get_data()[slices[0], slices[1], slices[2]],
        np.abs(image_zoom),
        order=1,
        mode='nearest'
    ).astype(np.int16)
    nib.save(
        nib.Nifti1Image(resampled_image, affine=np.eye(4)),
        os.path.join(
            args.output_dir,
            args.image_dir,
            data_idx + '.nii.gz'
        )
    )

    # label
    label_nib = nib.load(os.path.join(
        args.data_dir,
        args.label_dir,
        data_idx + '.nii.gz'
    ))
    label_zoom = np.diag(label_nib.affine)[:3]
    slices = [
        slice(-1, None, -1) if label_zoom[axis] < 0
        else slice(None)
        for axis in range(3)
    ]
    resampled_label = ndimage.zoom(
        label_nib.get_data()[slices[0], slices[1], slices[2]],
        np.abs(label_zoom),
        order=0,
        mode='nearest'
    ).astype(np.uint8)
    nib.save(
        nib.Nifti1Image(resampled_label, affine=np.eye(4)),
        os.path.join(
            args.output_dir,
            args.label_dir,
            data_idx + '.nii.gz'
        )
    )

    return data_idx


data_list = [
    fn.split('/')[-1].split('.nii.gz')[0]
    for fn in glob(os.path.join(
        args.data_dir,
        args.image_dir,
        '*.nii.gz'
    ))
]
data_list = sorted(data_list)

with Pool() as pool:
    progress_bar = tqdm(
        pool.imap(convert, data_list),
        total=len(data_list)
    )
    for data_idx in progress_bar:
        progress_bar.set_description(data_idx)
