import yaml
from .loaders import DataLoader, NIfTILoader
from .generators import DataGenerator
from .generators.reverter import Reverter
import nibabel as nib
import numpy as np


class ImagesSlicer:

    def __init__(self, axes, images, idx=None):
        self.axes = axes
        self.images = images
        self.slices = images[0].shape[-1]
        if idx is None:
            self.idx = self.slices // 2
        else:
            self.idx = idx
        self.imshows = list()

        self.imshows = [
            ax.imshow(image[:, :, self.idx], cmap='gray')
            for (ax, image) in zip(self.axes, self.images)
        ]
        axes[0].set_ylabel('Slice: %s' % self.idx)

    def on_scroll(self, event):
        if event.button == 'up':
            self.idx += 1
        else:
            self.idx -= 1
        self.idx = self.idx % self.slices
        self.axes[0].set_ylabel('Slice: %s' % self.idx)

        for im, image in zip(self.imshows, self.images):
            im.set_data(image[:, :, self.idx])
            im.axes.figure.canvas.draw()


def get_data_generator(data_config_file, generator_config_file, test=False):

    with open(data_config_file) as f:
        data_config = yaml.safe_load(f)

    with open(data_config['loader_config']) as f:
        loader_config = yaml.safe_load(f)

    with open(generator_config_file) as f:
        generator_config = yaml.safe_load(f)

    # construct data_loader for each train/valid/test data
    data_loader = dict()
    for key, lst in data_config['list'].items():
        data_loader[key] = DataLoader(*loader_config)
        if lst:
            data_loader[key].set_data_list(lst if not test else lst[:4])

    data_generator = {
        key: DataGenerator(data_loader[key], cfg)
        for key, cfg in generator_config.items()
    }

    return data_generator


def save_nifti(image, nifti_file):
    assert len(image.shape) == 3, image.shape
    if np.issubdtype(image.dtype, np.floating):
        image *= 255
    image = image.astype(np.int16)
    nifti_image = nib.Nifti1Image(image, affine=np.eye(4))
    nib.save(nifti_image, nifti_file)
