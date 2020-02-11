import numpy as np
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool


class SimpleParser:

    def __init__(self):
        self.crop_dict = dict()

    def window(self, image, width=100, level=50):
        image = (image - level + width/2) / width
        image = np.clip(image, 0., 1.0)
        return image

    def get_shapes(self):

        def get_shape(data_idx):
            return {
                'image': self.get_image_shape(data_idx),
                'label': self.get_label_shape(data_idx)
            }

        print('Generating shape dict ...')
        with Pool() as pool:
            shape_gen = tqdm(
                pool.imap(get_shape, self.data_list),
                total=self.n_data
            )
        shapes = {
            idx: shape for (idx, shape)
            in zip(self.data_list, shape_gen)
        }

        return shapes
