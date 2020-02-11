import numpy as np
import time


class TestParser:

    def __init__(
        self,
        shape=(256, 256, 150),
        n_data=50,
        n_labels=3,
        **kwargs
    ):

        self.image_dim = shape[:2]
        self.data_list = list(range(n_data))
        self.shape = shape
        self.n_labels = n_labels
        self.ROIs = list(range(n_labels))
        self.n_data = n_data

    def get_image_shape(self, data_idx):
        return self.shape

    def get_label_shape(self, data_idx):
        return self.shape

    def get_image(self, data_idx, **kwargs):
        # time.sleep(0.1)
        return np.random.rand(*self.shape)

    def get_label(self, data_idx, **kwargs):
        # time.sleep(0.1)
        return np.random.randint(self.n_labels, size=self.shape)
