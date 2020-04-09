import numpy as np

def dice_score(x, y):
    assert x.shape == y.shape, (x.shape, y.shape)
    return 2 * np.sum(x * y) / np.sum(x + y)
