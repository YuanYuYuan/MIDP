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
        flip=None,
        transpose=False,
        noise=None,
        normalization=False,
        minmax=False,
        affine=None,
        window_width=None,
        window_level=None,
        window_vmax=1.0,
        window_vmin=0.0,
        intensity_shift=0.0,
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
        if affine is not None:
            import torchio
            from torchio.transforms import (
                RandomAffine,
                RandomElasticDeformation,
                OneOf,
            )

            if affine == 'strong':
                transform = OneOf(
                    {
                        RandomAffine(
                            translation=10,
                            degrees=10,
                            scales=(0.9, 1.1),
                            default_pad_value='otsu',
                            image_interpolation='bspline'
                        ): 0.5,
                        RandomElasticDeformation(): 0.5
                    },
                    p=0.75,
                )
            else:
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

                subjs = {'label': torchio.Image(tensor=data['label'], type=torchio.LABEL)}
                shape = data['image'].shape

                # We need to seperate out the case of 4D image
                if len(shape) == 4:
                    n_channels = shape[-1]
                    for i in range(n_channels):
                        subjs.update({
                            f'ch{i}': torchio.Image(
                                tensor=data['image'][..., i],
                                type=torchio.INTENSITY
                            )
                        })

                else:
                    assert len(shape) == 3
                    subjs.update({
                        'image': torchio.Image(
                            tensor=data['image'],
                            type=torchio.INTENSITY
                        )
                    })

                transformed = transform(torchio.Subject(**subjs))

                if 'image' in subjs.keys():
                    data['image'] = transformed.image.numpy()

                else:
                    # if image contains multiple channels,
                    # then aggregate the transformed results into one
                    data['image'] = np.stack(tuple(
                        getattr(transformed, ch).numpy()
                        for ch in subjs.keys() if 'ch' in ch
                    ), axis=-1)
                data['label'] = transformed.label.numpy()

                for key in data:
                    data[key] = data[key].squeeze()

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
                    level=_window_level,
                    vmin=window_vmin,
                    vmax=window_vmax,
                )
                return data
            self.methods.append(_window)

        if noise is not None:
            assert isinstance(noise, float)
            assert noise > 0.
            def _noise(data):
                data['image'] += np.random.normal(
                    loc=0.0,
                    scale=noise,
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

        if flip is not None:
            assert isinstance(flip, (list, tuple))
            for f in flip:
                assert f >= 0 and f <= 1

            def flip_img(img, flip_x, flip_y, flip_z):
                if flip_x:
                    img = img[::-1, :, :, ...]

                if flip_y:
                    img = img[:, ::-1, :, ...]

                if flip_z:
                    img = img[:, :, ::-1, ...]

                return img

            def _flip(data):
                to_flip_x = random.random() < flip[0]
                to_flip_y = random.random() < flip[1]
                to_flip_z = random.random() < flip[2]
                for key in data:
                    data[key] = flip_img(data[key], flip_x=to_flip_x, flip_y=to_flip_y, flip_z=to_flip_z)
                return data
            self.methods.append(_flip)

        if transpose:
            def _transpose(data):
                if random.random() > 0.5:
                    for key in data:
                        data[key] = np.moveaxis(data[key], 0, 1)
                return data
            self.methods.append(_transpose)

        if intensity_shift > 0.:
            def _shift(data):
                data['image'] += (np.random.rand() * 2.0 - 1.0) * intensity_shift
                return data
            self.methods.append(_shift)


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
