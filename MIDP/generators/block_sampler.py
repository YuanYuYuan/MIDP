'''
Read data from DataLoader and then sample
3D blocks from the data.
For training only.
'''

import random
import numpy as np
from .multi_thread_queue_generator import MultiThreadQueueGenerator
from ..preprocessings import get_crop_idx


class BlockSampler:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data_loader):
        return _BlockSampler(data_loader, *self.args, **self.kwargs)


class _BlockSampler(MultiThreadQueueGenerator):

    def __init__(
        self,
        data_loader,
        shuffle=False,
        block_shape=(128, 128, 30),
        out_shape=None,
        n_samples=64,
        target_ratio=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # init variables
        self.shuffle = shuffle
        self.data_loader = data_loader
        self.out_shape = block_shape if out_shape is None else out_shape
        self.block_shape = block_shape
        self.n_samples = n_samples

        # in case of not tuple
        self.out_shape = tuple(self.out_shape)
        self.block_shape = tuple(self.block_shape)

        # data list
        self.data_list = data_loader.data_list

        # export class variables
        self.shapes = {
            'image': self.block_shape,
            'label': self.out_shape,
        }
        self.data_types = self.shapes.keys()
        if target_ratio == None:
            self.uniform_sample = True
        else:
            self.uniform_sample = False
            if not isinstance(target_ratio, list):
                target_ratio = [target_ratio]
            self.prob_interval = [sum(target_ratio[:k+1]) for k in range(len(target_ratio))]
            assert len(self.prob_interval) < data_loader.n_labels
        self.total = n_samples * data_loader.n_data


    def __len__(self):
        return self.total

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
                data[key] = self.data_loader.get_image(data_idx)
            elif key == 'label':
                data[key] = self.data_loader.get_label(data_idx)
            else:
                raise KeyError('Key should be either image or label.')
        return self.sample_blocks(data)

    def sample_blocks(self, data):

        # create a proper sampling range
        sampling_range = dict()
        sampling_range['min'] = tuple(bs//2 for bs in self.block_shape)
        sampling_range['max'] = tuple(
            (ls - bs//2) for (ls, bs) in
            zip(data['label'].shape, self.block_shape)
        )

        # sanity check
        for a, b in zip(sampling_range['min'], sampling_range['max']):
            assert a < b

        sampling_range['idx'] = tuple(
            slice(_min, _max) for (_min, _max)
            in zip(sampling_range['min'], sampling_range['max'])
        )

        def arbitrarily_sample():
            return tuple(
                np.random.randint(min, max) for (min, max)
                in zip(sampling_range['min'], sampling_range['max'])
            )

        def random_sample():
            # sample an arbitrary target index in range
            fail_counter = 0
            random_sample_fail = False
            while True:
                idx = tuple(
                    np.random.randint(min, max) for (min, max)
                    in zip(sampling_range['min'], sampling_range['max'])
                )

                # target conditions
                is_target = data['label'][idx] == target
                non_empty = data['image'][idx] > 0

                if is_target and non_empty:
                    return idx
                else:
                    fail_counter += 1
                    if fail_counter > 1000:
                        random_sample_fail = True
                        return None

        def random_sample_from_idx(idx_of_target):
            if len(idx_of_target[0]) == 0:
                return arbitrarily_sample()
            else:
                # randomly sample an idx
                sample_an_idx = np.random.randint(0, len(idx_of_target[0]))
                return tuple(
                    (i[sample_an_idx] + shift) for i, shift
                    in zip(idx_of_target, sampling_range['min'])
                )

        # sample n_samples blocks
        idx_of_target_dict = dict()
        for _ in range(self.n_samples):
            if self.uniform_sample:
                target_idx = arbitrarily_sample()

            else:
                # set default target = 0(brackground)
                target= 0

                # random change to foreground ROIs(>=1) with given probabilities
                rand = np.random.rand()
                for (i, threshold) in enumerate(self.prob_interval):
                    if rand <= threshold:
                        target = i + 1

                if target in idx_of_target_dict:
                    target_idx = random_sample_from_idx(idx_of_target_dict[target])
                else:
                    target_idx = random_sample()
                    if target_idx == None:
                        idx_of_target = np.where(data['label'][sampling_range['idx']] == 0)
                        target_idx = random_sample_from_idx(idx_of_target)
                        idx_of_target_dict[target] = idx_of_target

            # do cropping
            block = dict()
            for key in self.data_types:
                crop_idx = get_crop_idx(target_idx, self.shapes[key])
                block[key] = data[key][crop_idx]

                # sanity check
                assert block[key].shape == self.shapes[key]

            yield block
