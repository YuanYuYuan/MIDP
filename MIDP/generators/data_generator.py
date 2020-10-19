'''
This is a simple data generator for training and inference which
contains mainly 3 parts:
    1. Partition Generator (PG)
    2. Augmentation Generator (AG)
    3. Batch Generator (BG)
'''

from .. import generators
from ..loaders import DataLoader
from typing import Dict
import yaml


def load_gen(gen_config):
    assert 'data' in gen_config
    with open(gen_config.pop('data')) as f:
        data_config = yaml.safe_load(f)
    loader_config = data_config['loader']
    loader_name = loader_config.pop('name')
    data_loader = DataLoader(loader_name, **loader_config)
    data_loader.set_data_list(data_config['list'])
    return DataGenerator(data_loader, gen_config)


class DataGenerator:

    def __init__(
        self,
        data_loader: DataLoader,
        generators_config: Dict[str, dict]
    ):
        '''
        generators_config: {
            generator_name: {
                generator_config_parameters
            }
        }
        '''

        self._generator = data_loader
        self._struct = {'DL': data_loader}

        for gen_type in ['PG', 'AG', 'BG']:
            for gen_name in getattr(generators, gen_type):
                if gen_name in generators_config:

                    # get the next generator
                    gen_config = generators_config[gen_name]
                    gen = getattr(generators, gen_name)(**gen_config)

                    # compose the next and previous one
                    self._generator = gen(self._generator)

                    # update generator structure
                    self._struct[gen_type] = self._generator

    @property
    def done(self):
        return self._generator.done

    @property
    def struct(self):
        return self._struct

    @property
    def batch_size(self):
        return self._struct['BG'].batch_size

    def __len__(self):
        return len(self._generator)

    def __iter__(self):
        return iter(self._generator)

    def __enter__(self):
        return iter(self._generator)

    def __exit__(self, *args):
        for name in ['BG', 'AG', 'PG']:
            if name in self._struct:
                self._struct[name].reset_threads()
