'''
Read data from DataLoader and then partition
the data into 3D blocks
'''


import random
import numpy as np
from .multi_thread_queue_generator import MultiThreadQueueGenerator
from itertools import product
from ..preprocessings import crop_to_shape, pad_to_shape


class BlockGenerator:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, data_loader):
        return _BlockGenerator(data_loader, *self.args, **self.kwargs)


class _BlockGenerator(MultiThreadQueueGenerator):

    def __init__(
        self,
        data_loader,
        shuffle=False,
        block_shape=(128, 128, 30),
        stride=None,
        out_shape=None,
        crop_shape=None,
        include_label=True,
        ordered=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # init variables
        self.shuffle = shuffle
        self.data_loader = data_loader
        self.out_shape = block_shape if out_shape is None else out_shape
        self.block_shape = block_shape
        self.ordered = ordered
        if ordered:
            self.shuffle = False

        # in case of not tuple
        self.out_shape = tuple(self.out_shape)
        self.block_shape = tuple(self.block_shape)

        # set stride for block partition
        if stride is None:
            self.strides = self.out_shape
        elif isinstance(stride, int):
            self.strides = (stride,) * 3
        elif isinstance(stride, (list, tuple)):
            self.strides = tuple(s for s in stride)
        else:
            raise ValueError(stride)

        # check overlap caused by strides
        self.overlap = (self.strides != self.out_shape)
        if self.overlap:
            for i in range(3):
                assert self.strides[i] <= self.out_shape[i]

        # crop
        self.crop_shape = tuple(crop_shape) \
            if isinstance(crop_shape, list) else crop_shape
        if self.crop_shape:
            assert len(self.crop_shape) == 3

        # data list
        self.data_list = data_loader.data_list

        # count partition if cropping due to fixed data shape
        if self.crop_shape:
            steps = tuple(
                int(np.ceil(i/s)) for (i, s)
                in zip(self.crop_shape, self.strides)
            )
            self.steps_dict = {idx: steps for idx in self.data_list}
            self.partition = [np.prod(steps)] * len(self.data_list)
        else:
            self.steps_dict = dict()
            self.partition = list()

        # collect image shape and count partition if not cropping
        self.img_shape_dict = dict()
        for data_idx in self.data_list:
            if include_label:
                img_shape = data_loader.get_label_shape(data_idx)
            else:
                img_shape = data_loader.get_image_shape(data_idx)
            self.img_shape_dict[data_idx] = img_shape

            # check valid cropping
            if self.crop_shape:
                for i in range(3):
                    assert self.crop_shape[i] < img_shape[i], \
                        (self.crop_shape, img_shape)
            else:
                steps = tuple(
                    int(np.ceil(i/s)) for (i, s)
                    in zip(img_shape, self.strides)
                )
                self.steps_dict[data_idx] = steps
                self.partition.append(np.prod(steps))

        self.total = sum(self.partition)
        assert self.total > 0

        # gap/padding
        self.gap = tuple(
            (b - o) // 2 for (b, o) in
            zip(self.block_shape, self.out_shape)
        )

        # export class variables
        self.shapes = {'image': self.block_shape}
        self.data_types = ['image']
        if include_label:
            self.shapes['label'] = self.out_shape
            self.data_types.append('label')
        self.zeros = {key: np.zeros(self.shapes[key]) for key in self.shapes}

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
        return self.extract_blocks(data, data_idx)

    # TODO: implement the case of cropping case
    def extract_blocks(self, data, data_idx):

        if self.crop_shape:
            data = {
                key: crop_to_shape(data[key], self.crop_shape)
                for key in data
            }
            img_shape = self.crop_shape
        else:
            img_shape = self.img_shape_dict[data_idx]

        steps = self.steps_dict[data_idx]
        for nth_partition, base_idx in enumerate(product(*map(range, steps))):
            '''
            for steps = (1, 2, 3),
            base_idx will loop over [
                (0, 0, 0),
                (0, 0, 1),
                (0, 0, 2),
                (0, 1, 0),
                (0, 1, 1),
                (0, 1, 2)
            ]
            '''
            block = dict()
            for key in self.data_types:

                block_slice_idx = tuple()
                data_slice_idx = tuple()

                for ax in range(3):
                    # NOTE
                    # origin = base_idx[ax] * self.out_shape[ax]
                    origin = base_idx[ax] * self.strides[ax]
                    anchor = origin - self.gap[ax]

                    if key == 'image':
                        block_slice_idx += (slice(
                            max(-anchor, 0),
                            min(img_shape[ax]-anchor, self.block_shape[ax]),
                        ),)
                        data_slice_idx += (slice(
                            max(anchor, 0),
                            min(img_shape[ax], anchor+self.block_shape[ax]),
                        ),)

                    elif key == 'label':
                        block_slice_idx += (slice(
                            0,
                            min(img_shape[ax]-origin, self.out_shape[ax]),
                        ),)
                        data_slice_idx += (slice(
                            origin,
                            min(img_shape[ax], origin+self.out_shape[ax]),
                        ),)

                    else:
                        raise KeyError('Key should be either image or label.')

                block[key] = self.zeros[key].copy()
                block[key][block_slice_idx] = data[key][data_slice_idx]

            if self.ordered:
                idx = {
                    'data': data_idx,
                    'partition': nth_partition
                }
                yield idx, block
            else:
                yield block

    def _consumer(self):
        if self.ordered:
            return self._yield_data_in_order()
        else:
            return self._fast_yield()

    def _yield_data_in_order(self):

        tmp_idx = list()
        tmp_data = list()

        for data_idx, n_partitions in zip(self.data_list, self.partition):
            for partition_idx in range(n_partitions):
                ordered_data = None

                # check the oreder data in tmp or not
                for i in range(len(tmp_idx)):
                    if (tmp_idx[i]['data'], tmp_idx[i]['partition']) == (data_idx, partition_idx):
                        _, ordered_data = tmp_idx.pop(i), tmp_data.pop(i)
                        break

                # if not in tmp, then pull new data from queue
                if ordered_data is None:
                    while True:
                        idx, data = self.queue.get()

                        # if new data is the right one then yield
                        if (idx['data'], idx['partition']) == (data_idx, partition_idx):
                            ordered_data = data
                            break

                        # otherwise store them into tmp
                        else:
                            tmp_idx.append(idx)
                            tmp_data.append(data)
                yield ordered_data

    def restore(self, data_idx, blocks, output_threshold=0.35):

        steps = self.steps_dict[data_idx]
        '''
        Prepares a container to put blocks together.
        Note that this shape may not equal to raw image shape,
        since there will be padding and trimming later.
        '''
        zeros_shape = tuple(
            ((s-1) * st + os) for (s, st, os)
            in zip(steps, self.strides, self.out_shape)
        )

        # if output contains channel axis: [N, C, ...]
        contains_channel = len(blocks.shape) == 5
        if contains_channel:
            n_channels = blocks.shape[1]
            zeros_shape = (n_channels,) + zeros_shape
        else:
            assert len(blocks.shape) == 4

        # construct restoration array
        if isinstance(blocks, np.ndarray):
            restoration = np.zeros(zeros_shape)
        else:
            # FIXME torch.Tensor is not allowed anymore due to the later restoration
            import torch
            assert torch.is_tensor(blocks)
            restoration = torch.zeros(zeros_shape, dtype=torch.long)
            restoration = restoration.to(blocks.device)

        # do averaging if overlap and in shape [C, H, W, D]
        if self.overlap and contains_channel:
            count = np.zeros(zeros_shape)

        # insert each block into the restoration array
        base_indices = product(*map(range, steps))
        for (block, base_idx) in zip(blocks, base_indices):
            if len(block.shape) == 3:
                assert block.shape == self.out_shape
            else:
                assert len(block.shape) == 4
                assert block.shape[1:] == self.out_shape

            restoration_idx = tuple(
                slice(bi * st, bi * st + os) for (bi, st, os)
                in zip(base_idx, self.strides, self.out_shape)
            )
            if self.overlap:
                if contains_channel:
                    # do summation for blocks of shape [C, D_1, D_2, D_3]
                    restoration_idx = (slice(None),) + restoration_idx
                    restoration[restoration_idx] += block
                    count[restoration_idx] += 1
                else:
                    # do elementwise max for blocks of shape [D_1, D_2, D_3]
                    restoration[restoration_idx] = np.maximum(
                        restoration[restoration_idx],
                        block
                    )
            else:
                if contains_channel:
                    restoration_idx = (slice(None),) + restoration_idx
                restoration[restoration_idx] = block

        if contains_channel:
            if self.overlap:
                restoration /= count

            for i in range(1, restoration.shape[0]):
                restoration[i, ...] += \
                    (restoration[i, ...] >= output_threshold).astype(np.float)

            # restoration[1:, ...] = (restoration[1:, ...] >= output_threshold).astype(np.float)
            restoration = np.argmax(restoration, 0)

        # reshape the restoration to the original size
        orig_shape = self.img_shape_dict[data_idx]
        if self.crop_shape:
            # FIXME
            # # trim redundant voxels
            # trim_shape = (n_channels,) + orig_shape \
            #     if contains_channel else self.crop_shape
            trim_shape = self.crop_shape
            output = restoration[tuple(slice(ts) for ts in trim_shape)]

            # pad to original shape
            output = pad_to_shape(output, orig_shape)

        else:
            # FIXME
            # # trim redundant voxels
            # trim_shape = (n_channels,) + orig_shape \
            #     if contains_channel else orig_shape
            trim_shape = orig_shape
            output = restoration[tuple(slice(ts) for ts in trim_shape)]

        return output
