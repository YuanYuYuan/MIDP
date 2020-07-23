'''
Read data from DataLoader and then sample
3D blocks from the data.
For training only.
'''

import random
import numpy as np
from .multi_thread_queue_generator import MultiThreadQueueGenerator
# from ..preprocessings import get_crop_idx
from ..preprocessings import center_crop, pad_to_shape


class TargetSampler:

    def __init__(self, n_labels, ratios):
        '''
        n_labels: the number of labels, including the background
        ratios: the sampling ratio of each label
        '''

        # ensure in list type
        if not isinstance(ratios, list):
            ratios = [ratios]

        # e.g. ratios: [bg, fg1, fg2] = [0.1, 0.4, 0.5],
        # then prob_interval: [bg, fg1, fg2] = [0.1, 0.5, 1.0]
        total = sum(ratios)
        self.prob_interval = [
            float(sum(ratios[:i+1])) / float(total)
            for i in range(n_labels)
        ]

        # sanity check
        assert self.prob_interval[-1] == 1.0, self.prob_interval
        assert len(ratios) == n_labels

    def sample(self):
        # return the idx if a random number in its corresponding interval
        rand = np.random.rand()
        for (idx, threshold) in enumerate(self.prob_interval):
            if rand <= threshold:
                return idx


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
        pad_first=False,
        shift=None,
        n_samples=64,
        ratios=None,
        collapse_label=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # init variables
        self.shuffle = shuffle
        self.data_loader = data_loader
        self.out_shape = block_shape if out_shape is None else out_shape
        self.block_shape = block_shape
        self.n_samples = n_samples
        self.shift = tuple(shift) if isinstance(shift, list) else shift
        self.pad_first = pad_first
        if self.shift is not None:
            assert len(self.shift) == len(self.block_shape)
            # for s, o in zip(self.shift, self.out_shape):
            #     assert s <= (o // 2)

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
        self.data_types = ['image', 'label']
        if ratios is None:
            self.uniform_sample = True
        else:
            self.uniform_sample = False
            self.target_sampler = TargetSampler(data_loader.n_labels, ratios)
        self.total = n_samples * data_loader.n_data

        # FIXME
        self.collapse_label = collapse_label
        if self.collapse_label:
            self.shapes['label'] = (1,)
            self.shapes['anchor'] = (3,)

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

        raw_shape = data['label'].shape

        if self.pad_first:
            bigger_shape = tuple(
                bs // 2 + ds for (bs, ds)
                in zip(self.block_shape, data['label'].shape)
            )
            for key in data:
                data[key] = pad_to_shape(data[key], bigger_shape)

        need_padding = False
        for a, b in zip(data['label'].shape, self.block_shape):
            if a <= b:
                need_padding = True
                break

        if need_padding:
            bigger_shape = tuple(
                max(bs + 10, ds) for (bs, ds)
                in zip(self.block_shape, data['label'].shape)
            )
            for key in data:
                data[key] = pad_to_shape(data[key], bigger_shape)

        # create a proper sampling range
        sampling_range = dict()
        if self.pad_first:
            sampling_range['min'] = tuple(bs//2 for bs in self.block_shape)
        else:
            sampling_range['min'] = tuple(bs//2 for bs in self.block_shape)

        sampling_range['max'] = tuple(
            (ls - bs//2) for (ls, bs) in
            zip(data['label'].shape, self.block_shape)
        )

        # sanity check
        for a, b in zip(sampling_range['min'], sampling_range['max']):
            assert a < b, (a, b, data['label'].shape)

        sampling_range['idx'] = tuple(
            slice(_min, _max) for (_min, _max)
            in zip(sampling_range['min'], sampling_range['max'])
        )

        def arbitrarily_sample():
            return tuple(
                np.random.randint(min, max) for (min, max)
                in zip(sampling_range['min'], sampling_range['max'])
            )

        def random_sample(target):
            # sample an arbitrary target index in range
            fail_counter = 0
            while True:
                idx = tuple(
                    np.random.randint(min, max) for (min, max)
                    in zip(sampling_range['min'], sampling_range['max'])
                )

                # target conditions
                is_target = data['label'][idx] == target
                # non_empty = target == 0 or data['image'][idx] > 0
                non_empty = True

                if is_target and non_empty:
                    return idx
                else:
                    fail_counter += 1
                    if fail_counter > 1000:
                        return None

        def random_sample_from_target_range(target_range):
            if len(target_range[0]) == 0:
                return arbitrarily_sample()
            else:
                # randomly sample an idx
                sample_an_idx = np.random.randint(0, len(target_range[0]))

                # shift the target idx since the target_range is obtained from the sampling range
                # not the original range of label data
                return tuple(
                    s + t[sample_an_idx] for (t, s)
                    in zip(target_range, sampling_range['min'])
                )

        # use variable target_range_cache to cache the idx of sampled target
        # in next n_samples blocks
        target_range_cache = dict()

        # sample n_samples blocks
        for _ in range(self.n_samples):
            if self.uniform_sample:
                target_idx = arbitrarily_sample()

            else:
                target = self.target_sampler.sample()
                if target in target_range_cache:
                    target_idx = random_sample_from_target_range(target_range_cache[target])
                else:
                    target_idx = random_sample(target)
                    if target_idx is None:
                        target_range = np.where(data['label'][sampling_range['idx']] == target)
                        target_idx = random_sample_from_target_range(target_range)
                        target_range_cache[target] = target_range

            if self.shift:
                target_idx = tuple(
                    t + int(np.clip(np.random.normal(scale=1.5), -1, 1) * s)
                    for t, s in zip(target_idx, self.shift)
                )

            # do cropping
            block = dict()
            for key in self.data_types:
                shape = self.block_shape if key == 'image' else self.out_shape
                block[key] = center_crop(data[key], target_idx, shape)

                # sanity check
                assert block[key].shape == shape

            # FIXME
            if self.collapse_label:
                padding = tuple((x - y)//2 for x,y in zip(data['image'].shape, raw_shape))
                anchor = tuple(x - y - z//2 for x,y,z in zip(target_idx, padding, self.block_shape))
                block['anchor'] = anchor
                block['label'] = (np.sum(block['label']) > 0).astype(np.int)

            yield block
