import json
import os
import nibabel as nib
from .. import preprocessings


class NIfTILoader:

    def __init__(self, data_dir, test=False):

        with open(os.path.join(data_dir, 'info.json')) as f:
            info = json.load(f)

        self.data_dir = data_dir
        if test:
            self._data_list = info['list'][:2]
        else:
            self._data_list = info['list']
        self.ROIs = list(info['roi_map'].keys())
        self.roi_map = info['roi_map']

    def get_image(self, data_idx, window_width=100, window_level=50):
        data = nib.load(os.path.join(
            self.data_dir,
            'images',
            data_idx + '.nii.gz'
        )).get_data()
        data = preprocessings.window(data, window_width, window_level)
        return data

    def get_label(self, data_idx):
        return nib.load(os.path.join(
            self.data_dir,
            'labels',
            data_idx + '.nii.gz'
        )).get_data()

    @property
    def n_data(self):
        return len(self._data_list)

    @property
    def data_list(self):
        return self._data_list

    def set_data_list(self, new_list):
        assert set(new_list).issubset(self._data_list), new_list
        self._data_list = new_list