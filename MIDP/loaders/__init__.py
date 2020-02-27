from .parsing_loader import ParsingLoader
from .nifti_loader import NIfTILoader

LOADERS = {
    'NIfTILoader': NIfTILoader,
    'ParsingLoader': ParsingLoader
}


class DataLoader:

    def __new__(self, name, **kwargs):
        return LOADERS[name](**kwargs)
