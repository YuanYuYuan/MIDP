import random
from .thread_safe_generator import ThreadSafeGenerator as TSG
from .multi_thread_queue_generator import MultiThreadQueueGenerator
import numpy as np

'''
For training only
'''


def random_factor(_range):
    a, b = _range
    return random.random() * (b-a) + a


class Augmentor:

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, generator):
        return _Augmentor(generator, *self.args, **self.kwargs)


class _Augmentor(MultiThreadQueueGenerator):

    '''
    [input] generator produces dictionary ('image': image, 'label': label)
    [output] augmentor returns dictionary ('image': image, 'label': label)
    '''

    def __init__(
        self,
        generator,
        zoom_range=None,
        filter_range=None,
        flip=False,
        transpose=False,
        noise=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.shapes = generator.shapes

        # augmenting methods
        self.methods = []

        if noise:
            def _noise(data):
                for key in data:
                    data[key] += np.random.normal(
                        loc=0.0,
                        scale=0.05,
                        size=data[key].shape
                    )
                return data
            self.methods.append(_noise)

        if zoom_range is not None:
            from ..preprocessings import zoom

            def _zoom(data):
                zoom_factor = random_factor(zoom_range)
                for key in data:
                    data[key] = zoom(data[key], zoom_factor)
                return data
            self.methods.append(_zoom)

        if filter_range is not None:
            from scipy import ndimage

            def _filter(data):
                sigma = random_factor(filter_range)
                for key in data:
                    ndim = len(data[key].shape)
                    data[key] = ndimage.gaussian_filter(
                        data[key],
                        sigma=(sigma, sigma) + (0,) * (ndim - 2),
                    )
                return data
            self.methods.append(_filter)

        if flip:
            def _flip(data):
                if random.random() > 0.5:
                    for key in data:
                        data[key] = data[key][::-1, :, ...]
                else:
                    for key in data:
                        data[key] = data[key][:, ::-1, ...]
                return data
            self.methods.append(_flip)

        if transpose:
            def _transpose(data):
                if random.random() > 0.5:
                    for key in data:
                        data[key] = np.moveaxis(data[key], 0, 1)
                return data
            self.methods.append(_transpose)

    def __len__(self):
        return len(self.generator)

    def _init_jobs(self):
        self.jobs = TSG(iter(self.generator))

    def _producer_work(self):
        data = next(self.jobs)
        for method in self.methods:
            data = method(data)
        return data

    def _consumer(self):
        for _ in range(self.__len__()):
            data = self.queue.get()
            yield data
