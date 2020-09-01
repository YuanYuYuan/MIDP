#!/usr/bin/env python3
import SimpleITK as sitk
import os
import nibabel as nib
import numpy as np
from scipy import ndimage
from multiprocessing import Pool
from tqdm import tqdm
from glob import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--unlabeled', default=False, action='store_true')
parser.add_argument('--data-dir', default='data')
parser.add_argument('--output-dir', default='output')
parser.add_argument('--spacing', default=None, type=float)
args = parser.parse_args()

postfix = {
    'ct': '_ct.mha',
    't1': '_t1.mha',
    't2': '_t2.mha',
}
if not args.unlabeled:
    postfix.update({
        'task1': '_labelmap_task1.mha',
        'task2': '_labelmap_task2.mha',
    })

data_list = [
    elm.split('/')[-1].split('_ct.mha')[0]
    for elm in glob(os.path.join(args.data_dir, '*_ct.mha'))
]
data_list = sorted(data_list)
assert len(data_list) > 0

os.makedirs(args.output_dir, exist_ok=True)
for dtype in postfix:
    os.makedirs(os.path.join(args.output_dir, dtype), exist_ok=True)

def job(data_idx):
    for dtype in postfix:
        sitk_img = sitk.ReadImage(os.path.join(
            args.data_dir,
            data_idx + postfix[dtype]
        ))

        img = sitk.GetArrayFromImage(sitk_img)

        # Roll axes from z,y,x,c to x,y,z,c
        if img.ndim == 4:
            img = np.moveaxis(img, -1, 0)
        img = img.T

        # Orientation: RAI -> RAS
        # flip x, y axes
        img = img[::-1, ::-1, :]

        affine = np.eye(4)
        if args.spacing is not None:
            zoom = sitk_img.GetSpacing()
            zoom = tuple(z / args.spacing for z in zoom)
            if dtype in ['task1', 'task2']:
                zoomed_img = ndimage.zoom(img, zoom, order=0, mode='nearest')
            else:
                zoomed_img = ndimage.zoom(img, zoom, order=2, mode='nearest')
            assert zoomed_img.dtype == img.dtype
            img = zoomed_img
            for i in range(3):
                affine[i, i] *= args.spacing
        else:
            zoom = sitk_img.GetSpacing()
            for i in range(3):
                affine[i, i] = zoom[i]

        origin = np.array(sitk_img.GetOrigin())
        affine[0, 3] = (origin[0] + img.shape[0]) * -1
        affine[1, 3] = (origin[1] + img.shape[1]) * -1
        affine[2, 3] = origin[2]

        nib.save(
            nib.Nifti1Image(img, affine=affine),
            os.path.join(args.output_dir, dtype, data_idx + '.nii.gz')
        )

# job(data_list[0])

with Pool(os.cpu_count()) as pool:
    list(tqdm(
        pool.imap(job, data_list),
        total=len(data_list)
    ))
