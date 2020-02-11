from pathos.multiprocessing import ProcessingPool as Pool
import pydicom as dicom
from glob import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
from ruamel.yaml import YAML
import os
yaml = YAML()


class LungParser:

    def __init__(
        self,
        data_dir,
        image_subdir='images',
        label_subdir='labels',
        inference=False,
    ):

        # input variables
        self.data_dir = data_dir
        self.image_subdir = image_subdir
        self.label_subdir = label_subdir

        image_dir = os.path.join(data_dir, image_subdir)
        label_dir = os.path.join(data_dir, label_subdir)

        if inference:
            image_dir = image_dir if os.path.isdir(image_dir) else data_dir

        # data list
        image_list = sorted(os.listdir(image_dir))
        assert len(image_list) > 0

        if not inference:
            label_list = sorted([
                fn.split('.')[0]
                for fn in os.listdir(label_dir)
            ])
            assert image_list == label_list

        data_list = image_list

        # export class variables
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.data_list = data_list
        self.n_data = len(data_list)
        self.n_labels = 2
        self.image_dim = (512, 512)
        self.inference = inference

    def get_image_shape(self, data_idx):
        dcm_dir = os.path.join(self.image_dir, data_idx)
        dcm_files = self.get_dcm_files(dcm_dir)
        dcm = dicom.read_file(dcm_files[0])
        shape = (dcm.Rows, dcm.Columns, len(dcm_files))
        return shape

    def get_label_shape(self, data_idx):
        fn = os.path.join(self.label_dir, data_idx+'.nii.gz')
        shape = nib.load(fn).shape
        return shape

    def get_shapes(self):

        def get_shape(data_idx):
            return {
                'image': self.get_image_shape(data_idx),
                'label': self.get_label_shape(data_idx)
            }

        print('Generating shape dict ...')
        with Pool() as pool:
            shape_gen = tqdm(
                pool.imap(get_shape, self.data_list),
                total=self.n_data
            )
        shapes = {
            idx: shape for (idx, shape)
            in zip(self.data_list, shape_gen)
        }

        return shapes

    def window(self, image):
        MIN = -1000.0
        MAX = 400.0
        image = (image - MIN) / (MAX - MIN)
        image = np.clip(image, 0., 1.0)
        return image

    def get_dcm_files(self, dcm_dir):
        file_match = os.path.join(dcm_dir, '*.dcm')
        file_list = glob(file_match)
        assert len(file_list) > 0, 'file match: ' + file_match
        return file_list

    def get_image(self, data_idx, preprocess=False, expand_dim=False, **kwargs):

        # read dicom files
        dcm_files = self.get_dcm_files(os.path.join(self.image_dir, data_idx))
        dcm_list = [dicom.read_file(f) for f in dcm_files]
        dcm_list.sort(key=lambda dcm: dcm.SliceLocation)

        # hounsfield unit transform
        if kwargs.get('hu', preprocess):
            scan_list = list()
            for dcm in dcm_list:
                intercept = dcm.RescaleIntercept
                slope = dcm.RescaleSlope
                if slope != 1:
                    scan = slope * dcm.pixel_array.astype(np.float64)
                    scan = scan.astype(np.int16)
                else:
                    scan = dcm.pixel_array.astype(np.int16)
                scan += np.int16(intercept)
                scan_list.append(scan)
        else:
            scan_list = [dcm.pixel_array.astype(np.int16) for dcm in dcm_list]

        # stack scans
        data = np.stack(scan_list, axis=-1)

        # window
        data = self.window(data) if kwargs.get('window', preprocess) else data

        if expand_dim:
            data = np.expand_dims(data, -1)

        # do a x-y transporse to match the orientation of label(nifti)
        data = np.moveaxis(data, 0, 1)

        return data

    def get_label(self, data_idx, one_hot=False):
        file_path = os.path.join(self.label_dir, data_idx+'.nii.gz')
        data = nib.load(file_path).get_data().astype(np.int16)

        # one hot
        data = np.eye(self.n_labels, dtype=np.int16)[data] if one_hot else data

        return data
