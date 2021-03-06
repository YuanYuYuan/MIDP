import json
import os
import nibabel as nib
import numpy as np
from ..metrics import dice_score
from ..preprocessings import box_crop

# TODO: correct ROIs to classes


class NIfTILoader:

    def __init__(
        self,
        data_dir,
        test=False,
        bbox=None,
    ):

        with open(os.path.join(data_dir, 'info.json')) as f:
            info = json.load(f)

        self.data_dir = data_dir
        if test:
            self._data_list = info['list'][:2]
        else:
            self._data_list = info['list']
        self.ROIs = list(info['roi_map'].keys())
        self.roi_map = info['roi_map']

        # bounding box of each data
        if bbox is not None:
            with open(bbox) as f:
                self.bbox = json.load(f)
            for idx in self._data_list:
                assert idx in self.bbox
            self.use_bbox = True
        else:
            self.use_bbox = False
            self.bbox = None

        # include backgrounds
        # TODO: change to n_classes
        self.n_labels = len(self.ROIs) + 1

    def get_image_shape(self, data_idx):
        if self.use_bbox:
            return self.bbox[data_idx]['shape']
        else:
            return nib.load(os.path.join(
                self.data_dir,
                'images',
                data_idx + '.nii.gz'
            )).shape

    def get_image(self, data_idx):
        data = nib.load(os.path.join(
            self.data_dir,
            'images',
            data_idx + '.nii.gz'
        )).get_data()

        if self.use_bbox:
            data = box_crop(data, self.bbox[data_idx]['bbox'])
        return data.astype(np.int16)

    def get_label(self, data_idx):
        data = nib.load(os.path.join(
            self.data_dir,
            'labels',
            data_idx + '.nii.gz'
        )).get_data()

        if self.use_bbox:
            data = box_crop(data, self.bbox[data_idx]['bbox'])
        return data.astype(np.uint8)

    def get_label_shape(self, data_idx):
        if self.use_bbox:
            return self.bbox[data_idx]['shape']
        else:
            return nib.load(os.path.join(
                self.data_dir,
                'labels',
                data_idx + '.nii.gz'
            )).shape

    @property
    def n_data(self):
        return len(self._data_list)

    @property
    def data_list(self):
        return self._data_list

    def set_data_list(self, new_list):
        assert set(new_list).issubset(self._data_list), new_list
        self._data_list = new_list

    # assume the spacing has been converted into 1
    def save_prediction(self, data_idx, prediction, output_dir):
        assert isinstance(prediction, np.ndarray), type(prediction)
        os.makedirs(output_dir, exist_ok=True)
        nib.save(
            nib.Nifti1Image(prediction.astype(np.uint8), affine=np.eye(4)),
            os.path.join(output_dir, data_idx + '.nii.gz')
        )

    def evaluate(self, data_idx, prediction):
        return {
            roi: dice_score(
                (prediction == val).astype(int),
                (self.get_label(data_idx) == val).astype(int)
            )
            for roi, val in self.roi_map.items()
        }
