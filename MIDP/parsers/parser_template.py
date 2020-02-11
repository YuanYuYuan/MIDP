from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool


class ParserTemplate:

    def __init__(
        self,
        data_dir=None,
        data_list=None,
        ROIs=None,
        sort_ROIs=True,
        preprocess_image=False,
        expand_image_dim=False,
        one_hot_label=False,
        window_width=None,
        window_level=None,
    ):

        # set data dir
        assert data_dir is not None, 'Required to specify the data dir.'
        self.data_dir = data_dir

        # set data list
        assert len(self.file_list) > 0, 'Wrong data dir. Empty file list.'
        file_list = sorted(self.file_list)
        if data_list is None:
            self._data_list = file_list
        elif data_list == 'test':
            self._data_list = file_list[:3]
        else:
            assert type(data_list) == list
            assert set(data_list).issubset(file_list), data_list
            assert len(data_list) > 0
            self._data_list = data_list

        # check and sort ROIs
        assert set(ROIs).issubset(self.all_ROIs), ROIs
        if sort_ROIs:
            self.ROIs = [ROI for ROI in self.all_ROIs if ROI in ROIs]
        else:
            self.ROIs = ROIs

        self.preprocess_image = preprocess_image
        self.expand_image_dim = expand_image_dim
        self.one_hot_label = one_hot_label
        self.window_width = window_width
        self.window_level = window_level
        self.crop_dict = dict()

    @property
    def n_labels(self):
        return len(self.ROIs) + 1

    @property
    def all_ROIs(self):
        raise NotImplementedError

    @property
    def file_list(self):
        raise NotImplementedError

    @property
    def data_list(self):
        return self._data_list

    @property
    def n_data(self):
        return len(self._data_list)

    def get_image(self, data_idx):
        raise NotImplementedError

    def get_label(self, data_idx):
        raise NotImplementedError

    def get_image_shape(self, data_idx):
        raise NotImplementedError

    def get_label_shape(self, data_idx):
        raise NotImplementedError

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
