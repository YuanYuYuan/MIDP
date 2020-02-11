from glob import glob
import numpy as np
from .parser_template import ParserTemplate
import nrrd
from .. import preprocessings
import os


class PDDCAParser(ParserTemplate):

    def __init__(self, ROIs=['BrainStem'], **kwargs):
        super().__init__(ROIs=ROIs, **kwargs)

        # TODO: improve implementation
        self.image_dim = (256, 256)

    @property
    def all_ROIs(self):
        return [
            'Mandible',         # 1
            'BrainStem',        # 2
            'Parotid_L',        # 3
            'Parotid_R',        # 4
            'Submandibular_L',  # 5
            'Submandibular_R',  # 6
            'OpticNerve_L',     # 7
            'OpticNerve_R',     # 8
            'Chiasm',           # 9
        ]

    @property
    def file_list(self):
        return [
            fn.split('/')[-2]
            for fn in glob(os.path.join(self.data_dir, '*/'))
        ]

    def get_image_shape(self, data_idx):
        nrrd_info = nrrd.read_header(os.path.join(
            self.data_dir,
            data_idx,
            'img.nrrd'
        ))
        shape = tuple(nrrd_info['sizes'].tolist())

        # TODO: improve implementation
        # after rescale and crop
        z_scale = nrrd_info['space directions'][2, 2]
        shape = (256, 256, int(round(shape[2] * z_scale)))
        return shape

    def get_label_shape(self, data_idx):

        file_match = os.path.join(
            self.data_dir,
            data_idx,
            'structures',
            '*.nrrd'
        )
        shape = None
        for f in glob(file_match):
            if shape is None:
                # get shape
                nrrd_info = nrrd.read_header(f)
                shape = tuple(nrrd_info['sizes'].tolist())
                z_scale = nrrd_info['space directions'][2, 2]
            else:
                # check shape
                nrrd_info = nrrd.read_header(f)
                assert shape == tuple(nrrd_info['sizes'].tolist())
                assert z_scale == nrrd_info['space directions'][2, 2]

        shape = (256, 256, int(round(shape[2] * z_scale)))
        return shape

    def get_scale(self, data_idx):
        scale_array = nrrd.read(os.path.join(
            self.data_dir,
            data_idx,
            'img.nrrd'
        ))[1]['space directions']
        scale_array = np.abs(scale_array)
        return np.diag(scale_array)

    def get_image(self, data_idx, **kwargs):

        # TODO: improve implementation
        # check valid kwargs
        for arg in kwargs.keys():
            assert arg in [
                'preprocess',
                'expand_dim',
                'window_width',
                'window_level'
            ], '%s is not a valid keyword argument' % arg

        # shape: [512, 512, n_slices]
        nrrd_data = nrrd.read(os.path.join(
            self.data_dir,
            data_idx,
            'img.nrrd'
        ))
        data = nrrd_data[0]

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
            scale=np.diag(nrrd_data[1]['space directions']),
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

        data = None
        for idx, ROI in enumerate(self.ROIs):
            nrrd_path = os.path.join(
                self.data_dir,
                data_idx,
                'structures',
                ROI + '.nrrd'
            )
            nrrd_data = nrrd.read(nrrd_path)
            if data is None:
                data = nrrd_data[0]
            else:
                data = np.maximum(data, nrrd_data[0] * (idx+1))

        # TODO: improve implementation
        # rescaling
        data = preprocessings.rescale_and_crop(
            data,
            scale=np.diag(nrrd_data[1]['space directions']),
            crop_shape=(256, 256, -1),
        )

        if kwargs.get('one_hot', self.one_hot_label):
            data = np.eye(self.n_labels, dtype=np.int16)[data]
        return data
