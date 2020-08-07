import json
import os
import nibabel as nib
import numpy as np
from ..metrics import dice_score
from ..preprocessings import box_crop
from glob import glob

# TODO: correct ROIs to classes

TASK_1 = {
    'Cerebellum': 1,
    'Falx': 2,
    'Sinuses': 3,
    'Tentorium': 4,
    'Ventricles': 5,
}

TASK_2 = {
    'Brainstem': 1,
    'Chiasm': 2,
    'Cochlea_L': 3,
    'Cochlea_R': 4,
    'Eye_L': 5,
    'Eye_R': 6,
    'Lacrimal_L': 7,
    'Lacrimal_R': 8,
    'OpticNerve_L': 9,
    'OpticNerve_R': 10
}

class ABCSLoader:

    def __init__(
        self,
        data_dir,
        test=False,
        bbox=None,
        # modalities=['ct', 't1', 't2'],
        modalities=['ct'],
        task=1,
    ):

        self.data_dir = data_dir
        self.modalities = modalities

        assert task == 1 or task == 2
        self.task = task
        if task == 1:
            self.roi_map = TASK_1
        elif task == 2:
            self.roi_map = TASK_2
        else:
            raise ValueError('Task should be either 1 or 2.')
        self.ROIs = list(self.roi_map.keys())

        self._data_list = [
            elm.split('/')[-1].split('.')[0] for elm in glob(
                os.path.join(self.data_dir, 'task' + str(task), '*.nii.gz')
            )
        ]
        assert len(self._data_list) > 0
        if test:
            self._data_list = self._data_list[:2]

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
            shape = None
            for mod in self.modalities:
                if shape is None:
                    shape = nib.load(os.path.join(
                        self.data_dir,
                        mod,
                        data_idx + '.nii.gz'
                    )).shape
                else:
                    assert shape == nib.load(os.path.join(
                        self.data_dir,
                        mod,
                        data_idx + '.nii.gz'
                    )).shape

            return shape

    # FIXME introduce more modalities
    def get_image(self, data_idx):
        if len(self.modalities) == 1:
            data = nib.load(os.path.join(
                self.data_dir,
                'ct',
                data_idx + '.nii.gz'
            )).get_data()
        else:
            data = np.stack((
                nib.load(os.path.join(
                    self.data_dir,
                    mod,
                    data_idx + '.nii.gz'
                )).get_data()
                for mod in self.modalities
            ), axis=-1)

        if self.use_bbox:
            data = box_crop(data, self.bbox[data_idx]['bbox'])
        return data

    def get_label(self, data_idx):
        data = nib.load(os.path.join(
            self.data_dir,
            'task'+str(self.task),
            data_idx + '.nii.gz'
        )).get_data()

        if self.use_bbox:
            data = box_crop(data, self.bbox[data_idx]['bbox'])
        return data

    def get_label_shape(self, data_idx):
        if self.use_bbox:
            return self.bbox[data_idx]['shape']
        else:
            return nib.load(os.path.join(
                self.data_dir,
                'task'+str(self.task),
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
        affine = nib.load(os.path.join(
            self.data_dir,
            self.modalities[0],
            data_idx + '.nii.gz'
        )).affine
        nib.save(
            nib.Nifti1Image(prediction.astype(np.int16), affine=affine),
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
