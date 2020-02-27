#!/usr/bin/env python3

from MIDP import NIfTILoader, ImagesSlicer
from MIDP.generators import BlockSampler
import matplotlib.pyplot as plt
from typing import List
import numpy as np

data_loader = NIfTILoader(data_dir='nifti')

idx = data_loader.data_list[0]
for data_idx in data_loader.data_list:
    print(data_loader.get_label(idx).shape)

quit()

gen = BlockSampler(
    shuffle=False,
    block_shape=(96, 96, 96),
    out_shape=None,
    n_samples=64,
    ratios=None
)(data_loader)

for data in gen:
    print(data.shape)


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
