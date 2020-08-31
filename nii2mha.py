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
parser.add_argument('--nii-dir', default='nii')
parser.add_argument('--mha-dir', default='mha')
parser.add_argument('--output-dir', default='output')
args = parser.parse_args()

data_list = [
    elm.split('/')[-1].split('.nii.gz')[0]
    for elm in glob(os.path.join(args.nii_dir, '*.nii.gz'))
]

assert len(data_list) > 0
os.makedirs(args.output_dir, exist_ok=True)

def job(data_idx):
    img = nib.load(os.path.join(args.nii_dir, data_idx + '.nii.gz')).get_data()

    # Orientation: RAS -> RAI
    # flip x, y axes
    img = img[::-1, ::-1, :]

    # Roll axes from x,y,z to z,y,x
    img = img.T

    source_itk = sitk.ReadImage(os.path.join(args.mha_dir, data_idx + '_labelmap_task1.mha'))

    img = sitk.GetImageFromArray(img)
    img.CopyInformation(source_itk)

    sitk.WriteImage(img, os.path.join(args.output_dir, data_idx + '.mha'))

# job('001')

with Pool(os.cpu_count()) as pool:
    list(tqdm(
        pool.imap(job, data_list),
        total=len(data_list)
    ))
