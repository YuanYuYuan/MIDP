import os
from glob import glob
import nrrd
from scipy import ndimage
import numpy as np

# TODO: correct ROIs to classes


class NRRDLoader:

    def __init__(
        self,
        data_dir,
        roi_map={},
        spacing=1,
        resample=False,
        test=False
    ):

        self.data_dir = data_dir
        self._data_list = sorted([
            fn.split('/')[-2]
            for fn in glob(os.path.join(self.data_dir, '*/'))
        ])
        if test:
            self._data_list = self._data_list[:2]
        self.ROIs = list(roi_map.keys())
        self.roi_map = roi_map
        self.spacing = spacing
        self.resample = resample

        # include backgrounds
        # TODO change to n_classes
        self.n_labels = len(self.ROIs) + 1

    def _get_shape(self, nrrd_path):
        header = nrrd.read_header(nrrd_path)
        if self.resample:
            return tuple(
                int(
                    header['sizes'][i] *
                    header['space directions'][i, i] /
                    self.spacing
                ) for i in range(3)
            )
        else:
            return tuple(
                int(header['sizes'][i])
                for i in range(3)
            )

    def get_image_shape(self, data_idx):
        return self._get_shape(
            os.path.join(
                self.data_dir,
                data_idx,
                'img.nrrd'
            )
        )

    def get_label_shape(self, data_idx):

        def get_each_shape(roi):
            return self._get_shape(
                os.path.join(
                    self.data_dir,
                    data_idx,
                    'structures',
                    roi + '.nrrd'
                )
            )

        shape = None
        for roi in self.ROIs:
            if shape is None:
                shape = get_each_shape(roi)
            else:
                assert shape == get_each_shape(roi)

        return shape

    def _get_data(
        self,
        nrrd_path,
        mode='nearest',
        order=2
    ):
        nrrd_data = nrrd.read(nrrd_path)
        if self.resample:
            scale = tuple(
                nrrd_data[1]['space directions'][i, i] /
                self.spacing for i in range(3)
            )
            return ndimage.zoom(
                nrrd_data[0],
                scale,
                order=order,
                mode=mode
            )
        else:
            return nrrd_data[0]

    def get_image(self, data_idx):
        return self._get_data(
            os.path.join(
                self.data_dir,
                data_idx,
                'img.nrrd'
            ),
            mode='nearest',
            order=2
        )

    def get_label(self, data_idx):

        def get_each_data(roi):
            return self._get_data(
                os.path.join(
                    self.data_dir,
                    data_idx,
                    'structures',
                    roi + '.nrrd'
                ),
                mode='nearest',
                order=0
            )
        data = None
        for roi, idx in self.roi_map.items():
            if data is None:
                data = get_each_data(roi)
            else:
                data = np.maximum(data, get_each_data(roi) * idx)

        return data

    # TODO check consistent spacing
    def save_prediction(self, data_idx, prediction, output_dir):
        assert isinstance(prediction, np.ndarray), type(prediction)
        os.makedirs(
            os.path.join(output_dir, data_idx, 'structures'),
            exist_ok=True
        )
        header = nrrd.read_header(
            os.path.join(
                self.data_dir,
                data_idx,
                'img.nrrd'
            )
        )
        for roi, idx in self.roi_map.items():
            nrrd.write(
                os.path.join(
                    output_dir,
                    data_idx,
                    'structures',
                    roi + '.nrrd'
                ),
                (prediction == idx).astype(np.int),
                header=header
            )

    @property
    def n_data(self):
        return len(self._data_list)

    @property
    def data_list(self):
        return self._data_list

    def set_data_list(self, new_list):
        assert set(new_list).issubset(self._data_list), new_list
        self._data_list = new_list
