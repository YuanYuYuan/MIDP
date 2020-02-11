from glob import glob
import nibabel as nib
from .parser_template import ParserTemplate
import numpy as np
from .. import preprocessings
import os


class PancreasParser(ParserTemplate):

    def __init__(self, ROIs=['organ'], **kwargs):
        super().__init__(ROIs=ROIs, **kwargs)

        # TODO: improve implementation
        self.image_dim = (256, 256)

    @property
    def all_ROIs(self):
        return [
            'organ',   # 1
            'tumor',   # 2
        ]

    @property
    def file_list(self):
        return [
            fn.split('/')[-1].split('.')[0]
            for fn in glob(os.path.join(self.data_dir, 'images', '*.nii.gz'))
        ]

    def _get_shape(self, data_idx, data_type='image'):
        keyword = {
            'image': 'images',
            'label': 'labels',
        }
        file_path = os.path.join(
            self.data_dir,
            keyword[data_type],
            data_idx + '.nii.gz'
        )
        nib_data = nib.load(file_path)
        shape = nib_data.shape

        # TODO: improve implementation
        # after rescale and crop
        shape = (256, 256, int(round(shape[2] * nib_data.affine[2, 2])))

        return shape

    def get_image_shape(self, data_idx):
        return self._get_shape(data_idx, data_type='image')

    def get_label_shape(self, data_idx):
        return self._get_shape(data_idx, data_type='label')

    def get_image(self, data_idx, **kwargs):

        # check valid kwargs
        for arg in kwargs.keys():
            assert arg in [
                'preprocess',
                'expand_dim',
                'window_width',
                'window_level'
            ], '%s is not a valid keyword argument' % arg

        file_path = os.path.join(self.data_dir, 'images', data_idx + '.nii.gz')
        nib_data = nib.load(file_path)
        data = nib_data.get_data()

        # TODO: more preprocessing
        if kwargs.get('preprocess', self.preprocess_image):
            data = preprocessings.window(
                data,
                width=kwargs.get('window_width', self.window_width or 350),
                level=kwargs.get('window_level', self.window_level or 40),
            )

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
        shape: [256, 256, n_slices]
        '''

        # check valid kwargs
        for arg in kwargs.keys():
            assert arg in [
                'one_hot',
            ], '%s is not a valid keyword argument' % arg

        file_path = os.path.join(self.data_dir, 'labels', data_idx + '.nii.gz')
        nib_data = nib.load(file_path)
        data = nib_data.get_data()

        if self.ROIs == ['organ']:
            data[data > 1] = 1
        elif self.ROIs == ['tumor']:
            data[data == 1] = 0
            data[data > 1] = 1
        elif self.ROIs == ['organ', 'tumor']:
            pass
        else:
            raise ValueError('Wrong ROIs')

        # if self.ROIs == ['organ']:
        #     data[data > 1] = 1
        # elif self.ROIs == ['tumor']:
        #     data[data == 1] = 0
        #     data[data > 1] = 1
        # elif self.ROIs == ['organ', 'tumor']:
        #     data[data > 1] = 1
        #     data[data == 1] = 2
        # else:
        #     raise ValueError('Wrong ROIs')

        if kwargs.get('one_hot', self.one_hot_label):
            data = np.eye(self.n_labels, dtype=np.int16)[data]

        data = preprocessings.rescale_and_crop(
            data,
            scale=np.diag(nib_data.affine[:3, :3]),
            crop_shape=(256, 256, -1),
        )

        return data
