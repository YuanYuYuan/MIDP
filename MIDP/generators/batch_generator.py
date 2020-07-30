import numpy as np
from .thread_safe_generator import ThreadSafeGenerator as TSG
from .multi_thread_queue_generator import MultiThreadQueueGenerator
import torch

'''
For training only
'''


class BatchGenerator:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, generator):
        return _BatchGenerator(generator, *self.args, **self.kwargs)


class _BatchGenerator(MultiThreadQueueGenerator):

    def __init__(
        self,
        generator,
        batch_size,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # init variables
        self.generator = generator
        self.batch_size = batch_size

        # count batches
        self.n_batches = np.ceil(len(generator) / batch_size)
        self.n_batches = self.n_batches.astype(np.int16)
        assert self.n_batches > 0

        self.zero_batch = dict()
        for key in generator.shapes:
            if key == 'idx':
                self.zero_batch[key] = ['',] * batch_size
            else:
                self.zero_batch[key] = np.zeros((batch_size,) + generator.shapes[key])

        # self.zero_batch = {
        #     key: np.zeros((batch_size,) + generator.shapes[key])
        #     for key in generator.shapes
        # }

    def __len__(self):
        return self.n_batches

    def _init_jobs(self):
        self.jobs = TSG(iter(self.generator))

    def _producer_work(self):
        batch_data = {
            key: self.zero_batch[key].copy()
            for key in self.zero_batch
        }
        for batch_idx in range(self.batch_size):
            try:
                data = next(self.jobs)
                for key in batch_data:
                    batch_data[key][batch_idx] = data[key]
            except StopIteration:
                if batch_idx == 0:
                    # End of iteration
                    raise StopIteration
                else:
                    # use zero as padding
                    pass

        # process batch data
        for key in batch_data:
            if key == 'image':
                images = batch_data['image']

                # add a channel axis
                n_dim = len(images.shape)
                if n_dim == 5:
                    images = np.moveaxis(images, -1, 1)
                else:
                    assert n_dim == 3 or n_dim == 4
                    images = np.expand_dims(images, 1)

                images = torch.from_numpy(images)
                images = images.type(torch.FloatTensor)
                batch_data['image'] = images

            elif key == 'label':
                labels = batch_data['label']
                labels = torch.from_numpy(labels)
                labels = labels.type(torch.LongTensor)
                batch_data['label'] = labels

            elif key == 'anchor':
                anchor = batch_data['anchor']
                anchor = torch.from_numpy(anchor)
                anchor = anchor.type(torch.LongTensor)
                batch_data['anchor'] = anchor

            elif key == 'idx':
                pass

            else:
                raise KeyError

        return batch_data
