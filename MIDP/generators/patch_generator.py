'''
Read data from DataLoader and then partition
the data into 2D patches z * [x, y]
'''

import random
import numpy as np
from .multi_thread_queue_generator import MultiThreadQueueGenerator


class PatchGenerator:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data_loader):
        return _PatchGenerator(data_loader, *self.args, **self.kwargs)


class _PatchGenerator(MultiThreadQueueGenerator):

    def __init__(
        self,
        data_loader,
        shuffle=False,
        include_label=True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # init variables
        self.shuffle = shuffle
        self.data_loader = data_loader

        # data list
        self.data_list = data_loader.data_list

        # export class variables
        # TODO improve implementation
        # self.shapes = {'image': data_loader.image_dim}
        image_dim = data_loader.get_image_shape(self.data_list[0])[:2]
        self.shapes = {'image': image_dim}
        self.data_types = ['image']
        # TODO improve implementation
        if include_label:
            self.shapes['label'] = self.shapes['image']
            self.data_types.append('label')

    @property
    def partition(self):
        return [
            self.data_loader.get_label_shape(data_idx)[2]
            for data_idx in self.data_list
        ]

    def __len__(self):
        total = sum(self.partition)
        assert total > 0
        return total

    def _init_jobs(self):
        # create work list
        self.jobs = self.data_list.copy()
        if self.shuffle:
            random.shuffle(self.jobs)

    def _producer_work(self):
        data_idx = self.jobs.pop(0)
        data = dict()
        for key in self.data_types:
            if key == 'image':
                image = self.data_loader.get_image(data_idx)
                # move z axis to first
                image = np.moveaxis(image, 2, 0)
                data['image'] = image

            elif key == 'label':
                label = self.data_loader.get_label(data_idx)
                label = np.moveaxis(label, 2, 0)
                data['label'] = label

            else:
                raise KeyError('Key should be either image or label.')

        return data

    def _consumer(self):
        for _ in range(len(self.data_list)):
            data = self.queue.get()
            each_data = [data[key] for key in self.data_types]

            for data_slice in zip(*each_data):
                yield {
                    key: ds for (key, ds)
                    in zip(self.data_types, data_slice)
                }

    # def restore(self, data_idx, patches):
    #     orig_shape = self.data_loader.get_label_shape(data_idx)

    #     if isinstance(patches, np.ndarray):
    #         restored_data = np.moveaxis(patches, 0, 2)
    #     else:
    #         import torch
    #         assert torch.is_tensor(patches)
    #         restored_data = patches.permute(1, 2, 0)

    #     assert orig_shape == restored_data.shape
    #     return restored_data

    def revert(self, data_idx, patches, output_threshold=0.35):

        # if output contains channel axis: [N, C, ...]
        contains_channel = len(patches.shape) == 4
        if contains_channel:
            n_channels = patches.shape[1]
            for i in range(1, n_channels):
                patches[:, i, ...] += \
                        (patches[:, i, ...] >= output_threshold).astype(np.float)
            # [D, C, W, H] -> [C, W, H, D]
            restoration = np.moveaxis(patches, 0, -1)
            restoration = np.argmax(restoration, 0)
        else:
            assert len(patches.shape) == 3
            # [D, W, H] -> [W, H, D]
            restoration = np.moveaxis(patches, 0, -1)

        return restoration
