#!/usr/bin/env python3

from glob import glob
import os
import nrrd
from scipy import ndimage
import nibabel as nib
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    '--nrrd-dir',
    required=True,
    help='source directory containing original NRRD files'
)
parser.add_argument(
    '--nifti-dir',
    required=True,
    help='input directory containing NIfTI files'
)
parser.add_argument(
    '--output-dir',
    default='output',
    help='output directory storing the restored NRRD files'
)
args = parser.parse_args()


file_list = [
    fn.split('/')[-1].split('.')[0]
    for fn in glob(os.path.join(args.nifti_dir, '*.nii.gz'))
]
os.makedirs(args.output_dir, exist_ok=True)


def convert(data_idx):
    data = nib.load(os.path.join(
        args.nifti_dir,
        data_idx + '.nii.gz'
    )).get_data()
    header = nrrd.read_header(os.path.join(
        args.nrrd_dir,
        data_idx,
        'img.nrrd'
    ))
    zoom = tuple(
        (s1 / s2) for s1, s2
        in zip(header['sizes'], data.shape)
    )
    data = ndimage.zoom(
        data,
        zoom,
        order=0,
        mode='nearest'
    )
    assert data.shape == tuple(header['sizes'])
    nrrd.write(
        os.path.join(
            args.output_dir,
            data_idx + '.nrrd'
        ),
        data,
        header=header
    )


with Pool(os.cpu_count()) as pool:
    list(tqdm(pool.imap(convert, file_list), total=len(file_list)))
print('Ouputs have been stored in %s.' % args.output_dir)
