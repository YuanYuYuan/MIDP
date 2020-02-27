#!/usr/bin/env python3

from MIDP import DataLoader, ImagesSlicer
import matplotlib.pyplot as plt
import numpy as np
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    '--loader-config',
    required=True,
    help='loader config'
)
args = parser.parse_args()

with open(args.loader_config) as f:
    loader_config = yaml.safe_load(f)

loader_name = loader_config.pop('name')
data_loader = DataLoader(loader_name, **loader_config)

idx = data_loader.data_list[0]

images = [
    data_loader.get_image(idx),
    data_loader.get_label(idx)
]

idx_containing_ROI = np.where(images[1] != 0)[2][0]

fig, axes = plt.subplots(
    1,
    len(images),
    constrained_layout=True,
    figsize=(10, 5)
)

if len(images) == 1:
    axes = [axes]

images_slicer = ImagesSlicer(axes, images, idx=idx_containing_ROI)
fig.suptitle(f'ID: {idx}, ROIs: {data_loader.ROIs}')
fig.canvas.mpl_connect('scroll_event', images_slicer.on_scroll)
plt.show()
