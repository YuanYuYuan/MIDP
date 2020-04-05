#!/usr/bin/env python3

import argparse
import nrrd
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
    help='resampling spacing'
)
parser.add_argument(
    '--output-dir',
    default='output',
    help='output directory'
)
args = parser.parse_args()


def convert(data_idx):

    input_dir = os.path.join(args.data_dir, data_idx)
    output_dir = os.path.join(args.output_dir, data_idx)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'structures'), exist_ok=True)

    image_nrrd = nrrd.read(os.path.join(input_dir, 'img.nrrd'))
    header = image_nrrd[1]
    scale = tuple(
        header['space directions'][i, i] /
        args.spacing for i in range(3)
    )

    nrrd.write(
        os.path.join(output_dir, 'img.nrrd'),
        ndimage.zoom(image_nrrd[0], scale, order=2, mode='nearest'),
        header=header
    )

    for match in glob(os.path.join(
        args.data_dir,
        data_idx,
        'structures',
        '*.nrrd'
    )):
        target = match.split('.nrrd')[0].split('/')[-1]
        label_nrrd = nrrd.read(os.path.join(
            input_dir,
            'structures',
            '%s.nrrd' % target
        ))
        assert np.all(
            header['space directions'] == label_nrrd[1]['space directions']
        )
        nrrd.write(
            os.path.join(output_dir, 'structures', '%s.nrrd' % target),
            ndimage.zoom(label_nrrd[0], scale, order=0, mode='nearest'),
            header=header
        )


data_list = []
for match in sorted(glob(os.path.join(args.data_dir, '*/img.nrrd'))):
    data_idx = match.split('/img.nrrd')[0].split('/')[-1]
    data_list.append(data_idx)

# convert(data_idx)
with Pool() as pool:
    list(tqdm(
        pool.imap(convert, data_list),
        total=len(data_list)
    ))
