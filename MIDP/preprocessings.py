import numpy as np
import cv2
from scipy import ndimage


def rescale_and_crop(data, scale, crop_shape):
    # sanity check
    scale = tuple(abs(s) for s in scale)

    return crop_to_shape(ndimage.zoom(
        data,
        scale,
        order=0,
        mode='nearest',
    ), crop_shape)


def crop_to_shape(data, crop_shape):
    '''
    The crop_shape should be a tuple of same dim with data.shape.
    Value -1 in crop_shape means no cropping along this axis.
    '''
    center = tuple(s//2 for s in data.shape)
    shape = tuple(
        ds if cs == -1 else cs for (ds, cs)
        in zip(data.shape, crop_shape)
    )
    return data[get_crop_idx(center, shape)]


def window(image, width=100, level=50):
    image = (image - level + width/2) / width
    image = np.clip(image, 0., 1.0)
    return image


def get_crop_idx(center, shape):
    anchor = [max(0, int(c-s//2)) for (c, s) in zip(center, shape)]
    crop_idx = tuple(slice(a, a+s) for (a, s) in zip(anchor, shape))
    return crop_idx


def zoom(image, factor):
    '''
    zoom w.r.t center
    support 2D zoom only
    if image is in 3D, then z-axis will remain
    '''

    n_dim = len(image.shape)
    assert n_dim == 2 or n_dim == 3, image.shape

    original_shape = image.shape[:2]
    center = tuple(o//2 for o in original_shape)
    if type(factor) is not list:
        factor = [factor] * 2

    # cropping
    cropped_shape = tuple(
        int(o/f) for (o, f)
        in zip(original_shape, factor)
    )
    cropped_image = image[get_crop_idx(center, cropped_shape)]

    # resizing, note that x-y is transposed in opencv
    resized_shape = tuple(
        min(o, int(o*f)) for (o, f)
        in zip(original_shape, factor)
    )
    output = cv2.resize(
        cropped_image.astype(float),
        tuple(reversed(resized_shape))
    )

    # prevent the case cv2 removing last axis if it has only one channel
    if image.shape[-1] == 1:
        output = output.reshape(output.shape+(1,))

    # padding
    pad_width = tuple(
        ((o-r)//2, o - r - (o-r)//2) for (o, r)
        in zip(original_shape, resized_shape)
    )
    # no padding if there is z-axis
    pad_width += ((0, 0),) * (n_dim - 2)
    output = np.pad(output, pad_width, 'constant', constant_values=0)

    return output


def crop_and_resize(image, center, shape=(300, 300)):

    assert type(center) == tuple
    assert type(shape) == tuple
    assert len(center) == len(shape)
    assert len(image.shape) <= 3

    raw_shape = image.shape[:2]
    crop_idx = get_crop_idx(center, shape)
    output = cv2.resize(image[crop_idx], raw_shape)
    return output
