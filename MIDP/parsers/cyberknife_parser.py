from glob import glob
import nibabel as nib
from .parser_template import ParserTemplate
import numpy as np
import os
from .. import preprocessings
from ruamel.yaml import YAML
yaml = YAML()


class CyberknifeParser(ParserTemplate):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # export class variables
        self.n_labels = 2
        self.image_dim = (256, 256)

    def get_data_list(self, data_list):
        file_list = [
            fn.split('/')[-2]
            for fn in glob(os.path.join(self.data_dir, '*/'))
        ]
        assert len(file_list) > 0
        file_list = sorted(file_list)

        if data_list is None:
            return file_list
        elif data_list == 'test':
            return file_list[:4]
        else:
            assert type(data_list) == list
            assert set(data_list).issubset(file_list), data_list
            assert len(data_list) > 0
            return data_list

    def get_image_shape(self, data_idx):
        nib_data = nib.load(os.path.join(self.data_dir, data_idx, 'CT.nii.gz'))
        shape = nib_data.shape

        # TODO: improve implementation
        # after rescale and crop
        shape = (256, 256, int(round(shape[2] * nib_data.affine[2, 2])))

        return shape

    def get_label_shape(self, data_idx):
        nib_data = nib.load(os.path.join(
            self.data_dir,
            data_idx,
            'BrainStem.nii.gz'
        ))
        shape = nib_data.shape

        # TODO: improve implementation
        # after rescale and crop
        shape = (256, 256, int(round(shape[2] * nib_data.affine[2, 2])))
        return shape

    def get_image(self, data_idx, **kwargs):

        # check valid kwargs
        for arg in kwargs.keys():
            assert arg in [
                'preprocess',
                'expand_dim',
                'window_width',
                'window_level'
            ], '%s is not a valid keyword argument' % arg

        nib_data = nib.load(os.path.join(self.data_dir, data_idx, 'CT.nii.gz'))
        data = nib_data.get_data()

        # TODO: more preprocessing
        if kwargs.get('preprocess', self.preprocess_image):
            data = preprocessings.window(
                data,
                width=kwargs.get('window_width', self.window_width or 100),
                level=kwargs.get('window_level', self.window_level or 50),
            )

        # TODO: improve implementation
        # rescaling
        data = preprocessings.rescale_and_crop(
            data,
            scale=np.diag(nib_data.affine[:3, :3]),
            crop_shape=(256, 256, -1),
        )

        if kwargs.get('expand_dim', self.expand_image_dim):
            data = np.expand_dims(data, -1)

        return data

    def get_label(self, data_idx, **kwargs):
        '''
        orig shape: [512, 512, n_slices]
        rescaled shape: [256, 256, rescaled_n_slices]
        '''

        # check valid kwargs
        for arg in kwargs.keys():
            assert arg in [
                'one_hot',
            ], '%s is not a valid keyword argument' % arg

        # TODO: add more channels
        nib_data = nib.load(os.path.join(
            self.data_dir,
            data_idx,
            'BrainStem.nii.gz'
        ))
        data = nib_data.get_data()
        data = data.astype(np.int16)

        # TODO: improve implementation
        # rescaling
        data = preprocessings.rescale_and_crop(
            data,
            scale=np.diag(nib_data.affine[:3, :3]),
            crop_shape=(256, 256, -1),
        )

        if kwargs.get('one_hot', self.one_hot_label):
            data = np.eye(self.n_labels, dtype=np.int16)[data]

        return data
