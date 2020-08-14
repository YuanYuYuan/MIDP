import random
from .thread_safe_generator import ThreadSafeGenerator as TSG
from .multi_thread_queue_generator import MultiThreadQueueGenerator
import numpy as np
import torch


# TODO: Add n_samples
# NOTE: For training only


def random_factor(_range):
    assert len(_range) == 2
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
        normalization=False,
        minmax=False,
        affine=False,
        window_width=None,
        window_level=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.generator = generator
        self.shapes = generator.shapes

        # augmenting methods
        self.methods = []

        if normalization:
            def _norm(data):
                axes = tuple(range(len(data['image'].shape)))
                mean = np.mean(data['image'], axis=axes)
                std = np.std(data['image'], axis=axes)
                data['image'] = (data['image'] - mean) / std
                return data

            self.methods.append(_norm)

        if minmax:
            def _minmax(data):
                lower_percentile = 0.2,
                upper_percentile = 99.8

                foreground = data['image'] != data['image'][(0,) * len(data['image'].shape)]
                min_val = np.percentile(data['image'][foreground].ravel(), lower_percentile)
                max_val = np.percentile(data['image'][foreground].ravel(), upper_percentile)
                data['image'][data['image'] > max_val] = max_val
                data['image'][data['image'] < min_val] = min_val
                data['image'] = (data['image'] - min_val) / (max_val - min_val)
                data['image'][~foreground] = 0

                return data

            self.methods.append(_minmax)


        # affine
        if affine:
            import torchio
            from torchio.transforms import (
                RandomAffine,
                RandomElasticDeformation,
                OneOf,
            )
            transform = OneOf(
                {
                    RandomAffine(translation=10): 0.5,
                    RandomElasticDeformation(): 0.5
                },
                p=0.75,
            )

            def _affine(data):

                for key in data:
                    data[key] = torch.Tensor(data[key])

                transformed = transform(torchio.Subject(
                    image=torchio.Image(
                        tensor=data['image'],
                        type=torchio.INTENSITY
                    ),
                    label=torchio.Image(
                        tensor=data['label'],
                        type=torchio.LABEL
                    )
                ))

                data['image'] = transformed.image.numpy()
                data['label'] = transformed.label.numpy()
                return data

            self.methods.append(_affine)

        # convert image to float
        def _to_float(data):
            data['image'] = data['image'].astype(np.float)
            return data
        self.methods.append(_to_float)

        # adjust contrast/window
        if window_width or window_level:
            from ..preprocessings import window
            window_width = window_width if window_width else 100
            window_level = window_level if window_level else 50

            def _window(data):
                if isinstance(window_width, (tuple, list)):
                    _window_width = random_factor(window_width)
                else:
                    _window_width = window_width
                if isinstance(window_level, (tuple, list)):
                    _window_level = random_factor(window_level)
                else:
                    _window_level = window_level
                data['image'] = window(
                    data['image'],
                    width=_window_width,
                    level=_window_level
                )
                return data
            self.methods.append(_window)

        if noise:
            def _noise(data):
                data['image'] += np.random.normal(
                    loc=0.0,
                    scale=0.05,
                    size=data['image'].shape
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

        # TODO: deprecated
        # if filter_range is not None:
        #     from scipy import ndimage

        #     def _filter(data):
        #         sigma = random_factor(filter_range)
        #         ndim = len(data['image'].shape)
        #         data['image'] = ndimage.gaussian_filter(
        #             data['image'],
        #             sigma=(sigma, sigma) + (0,) * (ndim - 2),
        #         )
        #         return data
        #     self.methods.append(_filter)

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
